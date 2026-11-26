//! Layer 2's n-gram table: the hash of each token's row ids, and the mapped table a pass gathers its rows from.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");
const ple_pool = @import("cuda_ple_pool.zig");
const ple_reader = @import("cuda_ple_reader.zig");
const ple_fp8 = @import("cuda_ple_fp8.zig");
const host = @import("ple_host.zig");

pub const max_n = host.max_n;
pub const max_heads = host.max_heads;
const max_shards = host.max_shards;
pub const layer = host.layer;
pub const Hash = host.Hash;
pub const lastIds = host.lastIds;
pub const toBf16 = host.toBf16;
pub const sumGroups = host.sumGroups;
pub const Open = host.Open;
pub const readI64 = host.readI64;
pub const weightName = host.weightName;
pub const partName = host.partName;
pub const partNameFor = host.partNameFor;
pub const findOpen = host.findOpen;
pub const copyRow = host.copyRow;
pub const tableScale = host.tableScale;
const readScale = host.readScale;
pub const readSpec = host.readSpec;
pub const eosOf = host.eosOf;
pub const Format = host.Format;

/// The n-gram table held open for an engine's life: the hash, the scale, and each shard's slices of its mapped file.
pub const Ngram = struct {
    gpa: std.mem.Allocator,
    hash: Hash,
    scale: f32,
    files: std.ArrayList(Open) = .empty,
    w: [max_shards][]const u8 = undefined,
    s: [max_shards][]const u8 = undefined,
    b: [max_shards][]const u8 = undefined,
    bits: [max_shards]u8 = @splat(4),
    format: Format = .mlx,
    tables: [max_shards]core.nvfp4_rows.Table = undefined,
    tables8: [max_shards]ple_fp8.Table = undefined,
    readers: [max_shards]ple_reader.Reader = undefined,
    read_mode: ple_reader.Mode = .mmap,
    warm_ahead: bool = false,
    pool: ?*ple_pool.Pool = null, // the gather's parked workers (null: every row on the caller's thread)
    warmer: ?std.Thread = null, // warm's read-ahead thread until joined (waitWarm, deinit)

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Ngram {
        const cfg = try readSpec(gpa, io, model_dir);
        var g: Ngram = .{ .gpa = gpa, .hash = try Hash.init(.{
            .vocab = cfg.vocab,
            .ngram = cfg.ngram,
            .per = cfg.per,
            .base = cfg.base,
            .divisor = cfg.divisor,
            .shards = cfg.shards,
            .seed = cfg.seed,
            .eos = cfg.eos,
            .ple_dim = cfg.ple_dim,
            .ple_index = cfg.ple_index,
        }), .scale = 1 };
        errdefer g.deinit();
        const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
        defer gpa.free(index_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const wm = try core.safetensors.index.weightMap(parsed.value);
        const first = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        const modelopt = wm.contains(first);
        g.format = if (modelopt) try ple_fp8.format(gpa, io, model_dir, wm, first) else .mlx;
        if (modelopt) g.read_mode = try ple_reader.mode(if (std.c.getenv("TF_FLASHNEXT_PLE_IO")) |v| std.mem.span(v) else null);
        g.warm_ahead = try @import("sys").environment.read(bool, "TF_FLASHNEXT_WARM_AHEAD", false);
        g.scale = switch (g.format) {
            .mlx => try tableScale(gpa, io, model_dir, wm),
            .nvfp4 => try readScale(gpa, io, model_dir, wm, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2", true),
            .fp8 => try readScale(gpa, io, model_dir, wm, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale", true),
        };
        const dh = g.hash.dh;
        var key: [180]u8 = undefined;
        for (0..g.hash.shards) |shard| {
            var nested = false;
            const mapped = if (wm.get(try partNameFor(&key, shard, false, "weight", modelopt))) |v|
                v.string
            else blk: {
                nested = true;
                break :blk (wm.get(try partNameFor(&key, shard, true, "weight", modelopt)) orelse return error.MissingTensor).string;
            };
            const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
            defer gpa.free(path);
            const file = findOpen(g.files.items, path) orelse file: {
                var o = try Open.load(gpa, io, path);
                errdefer o.close(gpa);
                if (@import("builtin").os.tag != .windows) std.posix.madvise(o.file.map.memory.ptr, o.file.map.memory.len, std.posix.MADV.RANDOM) catch {};
                try g.files.append(gpa, o);
                break :file &g.files.items[g.files.items.len - 1];
            };
            const dtype: core.safetensors.DType = switch (g.format) {
                .mlx => .u32,
                .nvfp4 => .u8,
                .fp8 => .f8_e4m3,
            };
            const w = try file.tensor(try partNameFor(&key, shard, nested, "weight", modelopt), dtype);
            const expected_rows: usize = @intCast(g.hash.starts[shard + 1] - g.hash.starts[shard]);
            if (w.rank != 2 or w.dim(0) != expected_rows) return error.UnexpectedTensor;
            if (modelopt) {
                try ple_fp8.openShard(&g, file, shard, nested, w);
            } else {
                const sc = try file.tensor(try partNameFor(&key, shard, nested, "scales", false), .bf16);
                const bias = try file.tensor(try partNameFor(&key, shard, nested, "biases", false), .bf16);
                const b: u8 = if (w.dim(1) == dh * 3 / 32) 3 else if (w.dim(1) == dh / 8) 4 else if (w.dim(1) == dh / 4) 8 else return error.UnexpectedTensor;
                if (!sc.is(.bf16, &.{ expected_rows, dh / 32 }) or !bias.is(.bf16, &.{ expected_rows, dh / 32 })) return error.UnexpectedTensor;
                g.w[shard] = w.bytes;
                g.s[shard] = sc.bytes;
                g.b[shard] = bias.bytes;
                g.bits[shard] = b;
            }
        }
        g.pool = try ple_pool.Pool.create(gpa, io);
        return g;
    }

    pub fn deinit(g: *Ngram) void {
        g.waitWarm();
        if (g.pool) |p| p.destroy();
        for (g.files.items) |*o| o.close(g.gpa);
        g.files.deinit(g.gpa);
    }

    /// Values a row's embedding holds (heads x head width).
    pub fn width(g: *const Ngram) usize {
        return g.hash.heads * g.hash.dh;
    }

    /// With TF_FLASHNEXT_TABLE_WARM=1, the table read ahead into the page cache after load, so a cold prompt reads RAM.
    pub fn warm(g: *Ngram) void {
        // the startup check refused every malformed value already, so a plain default stands here
        const on = @import("sys").environment.read(bool, "TF_FLASHNEXT_TABLE_WARM", false) catch false;
        if (@import("builtin").os.tag != .linux or !on) return;
        g.warmer = std.Thread.spawn(.{}, warmMain, .{g.*}) catch null;
    }

    /// Returns once warm's read-ahead is queued.
    pub fn waitWarm(g: *Ngram) void {
        if (g.warmer) |t| t.join();
        g.warmer = null;
    }

    fn warmMain(g: Ngram) void {
        var bytes: usize = 0;
        for (0..g.hash.shards) |shard| switch (g.format) {
            .nvfp4 => bytes += advise(g.tables[shard].codes) + advise(g.tables[shard].scales),
            .mlx => bytes += advise(g.w[shard]) + advise(g.s[shard]) + advise(g.b[shard]),
            .fp8 => bytes += advise(g.tables8[shard].codes),
        };
        std.debug.print("n-gram table: {d:.1} GiB read ahead into the page cache\n", .{@as(f64, @floatFromInt(bytes)) / (1 << 30)});
    }

    /// MADV_WILLNEED over the pages of `bytes`, a 128 KiB readahead window a call (the most the kernel reads a call).
    fn advise(bytes: []const u8) usize {
        if (bytes.len == 0) return 0;
        const page = std.heap.pageSize();
        const step = 128 << 10;
        const lo = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
        const hi = std.mem.alignForward(usize, @intFromPtr(bytes.ptr) + bytes.len, page);
        var at = lo;
        while (at < hi) : (at += step) _ = std.os.linux.madvise(@ptrFromInt(at), @min(step, hi - at), std.os.linux.MADV.WILLNEED);
        return bytes.len;
    }

    /// Rows [lo, hi) of a chunk's token-major row ids `ids`, each dequantized into its head's slice of `out`.
    pub fn fill(g: *const Ngram, ids: []const i64, out: []u16, lo: usize, hi: usize) !void {
        const dh = g.hash.dh;
        for (lo..hi) |i| {
            const loc = try g.hash.locate(ids[i]);
            const row = out[i * dh ..][0..dh];
            switch (g.format) {
                .mlx => try @import("cuda_tfq3.zig").dequantPleRow(g.bits[loc.shard], g.w[loc.shard], g.s[loc.shard], g.b[loc.shard], dh, loc.local, row),
                .nvfp4, .fp8 => if (g.read_mode == .pread)
                    try g.readers[loc.shard].read(g.gpa, loc.local, row)
                else if (g.format == .fp8)
                    try g.tables8[loc.shard].gather(&.{loc.local}, row)
                else
                    try g.tables[loc.shard].gather(&.{loc.local}, row),
            }
        }
    }

    /// One call's rows split into `parts` equal ranges, part i filling [i * per, (i + 1) * per).
    pub const Split = struct {
        g: *const Ngram,
        ids: []const i64,
        out: []u16,
        per: usize,

        pub fn part(ctx: *const anyopaque, i: usize) anyerror!void {
            const s: *const Split = @ptrCast(@alignCast(ctx));
            const lo = @min(s.ids.len, i * s.per);
            try s.g.fill(s.ids, s.out, lo, @min(s.ids.len, lo + s.per));
        }
    };

    pub const warmAhead = @import("cuda_ple_ahead.zig").warm;

    /// `embedding` for each token after `history` (moved past the chunk) into out and xs, read by up to 128 threads.
    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16, xs: []f32) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w or xs.len != out.len / 32) return error.UnexpectedTensor;
        const items = tokens.len * heads;
        const ids = try g.gpa.alloc(i64, items);
        defer g.gpa.free(ids);
        var hist: [max_n]i64 = undefined;
        @memcpy(hist[0..history.len], history);
        for (tokens, 0..) |token, t| {
            try lastIds(g.hash, hist[0..history.len], @intCast(token), ids[t * heads ..][0..heads]);
            for (0..history.len - 1) |i| hist[i] = hist[i + 1];
            hist[history.len - 1] = token;
        }
        const n_threads = @min(ple_pool.max_parts, items / 4);
        if (g.format != .mlx and items > 1) {
            try @import("cuda_ple_gather.zig").run(g, ids, out);
        } else if (n_threads <= 1 or g.pool == null) {
            try g.fill(ids, out, 0, items);
        } else {
            const split: Split = .{ .g = g, .ids = ids, .out = out, .per = (items + n_threads - 1) / n_threads };
            try g.pool.?.run(n_threads, .{ .ctx = &split, .run = Split.part });
        }
        for (0..tokens.len) |t| {
            const row = out[t * w ..][0..w];
            if (g.format == .mlx and g.scale != 1) {
                for (row) |*v| {
                    const f: f32 = @bitCast(@as(u32, v.*) << 16);
                    v.* = toBf16(f * g.scale);
                }
            }
            sumGroups(row, xs[t * (w / 32) ..][0 .. w / 32]);
        }
        @memcpy(history, hist[0..history.len]);
    }
};

test {
    _ = @import("cuda_ple_gather_lanes.zig");
    _ = @import("cuda_ple_nvfp4_test.zig");
    _ = @import("cuda_ple_pool.zig");
    _ = @import("cuda_ple_pool_gather_test.zig");
}

test {
    _ = @import("cuda_ple_gather.zig");
}

test {
    _ = host;
    _ = ple_reader;
    _ = ple_fp8;
    _ = @import("cuda_ple_ahead.zig");
}
