//! The n-gram tables in host memory (the checkpoint's mapped shards): a window's rows gathered for the device.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");
const host_pages = @import("../host_pages.zig");
/// The parked workers a gather spreads over: no backend in them, so the CUDA n-gram reader shares them.
const pool = @import("../host_pool.zig");
const nvfp4 = @import("tables_nvfp4.zig");

/// One shard's packed rows and their group tables, as mapped.
pub const Shard = struct { words: []const u8, scales: []const u8, biases: []const u8 };

/// A gather copies the checkpoint's rows in id order: a staging too small, or a row past the last shard.
const Error = error{ StagingTooSmall, NgramRowOutOfRange };

/// Parts a gather spreads over at most (all the pool's: its page reads wait together), and rows a part takes first.
const max_parts = pool.max_parts;
const rows_per_part = 2;

/// The shards' rows as mapped: what a gather reads, apart from the hasher that names the rows.
pub const Rows = struct {
    shards: []const Shard,
    /// The first row of each shard, and one past the last.
    starts: []const usize,
    /// The first `extra` shards hold `wide` rows and the rest `narrow`; `wide` is 0 when the sizes do not fit.
    wide: usize,
    narrow: usize,
    extra: usize,
    dims: usize,
    bits: u8,
    group: usize,

    /// Bytes of a row's packed words, and of its scales (as of its biases).
    pub fn wordBytes(r: *const Rows) usize {
        return r.dims * r.bits / 8;
    }

    pub fn tableBytes(r: *const Rows) usize {
        return r.dims / r.group * 2;
    }

    /// The shard sizes the reference splits rows into: `wide` rows for the first `extra` shards, then `narrow`.
    pub const Split = struct { wide: usize, narrow: usize, extra: usize };

    /// The sizes `starts` shows, or zeros when they follow no such run and a row must be searched for.
    pub fn split(starts: []const usize) Split {
        const n = starts.len - 1;
        if (n == 0 or starts[1] - starts[0] == 0) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        const wide = starts[1] - starts[0];
        var extra: usize = 1;
        while (extra < n and starts[extra + 1] - starts[extra] == wide) extra += 1;
        const narrow = if (extra < n) starts[extra + 1] - starts[extra] else wide;
        if (narrow == 0) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        for (extra..n) |i| {
            if (starts[i + 1] - starts[i] != narrow) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        }
        return .{ .wide = wide, .narrow = narrow, .extra = extra };
    }

    /// Row `id`'s shard and its row in it: a shard's rows are a contiguous run, so the sizes need no search.
    pub fn locate(r: *const Rows, id: u32) Error!struct { shard: usize, row: usize } {
        const at: usize = id;
        if (at >= r.starts[r.shards.len]) return error.NgramRowOutOfRange;
        if (r.wide != 0) {
            const first = r.wide * r.extra;
            if (at < first) return .{ .shard = at / r.wide, .row = at % r.wide };
            return .{ .shard = r.extra + (at - first) / r.narrow, .row = (at - first) % r.narrow };
        }
        const s = std.sort.upperBound(usize, r.starts, at, struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        return .{ .shard = s, .row = at - r.starts[s] };
    }

    /// Every byte a gather reads (a shard's three arrays): the pages a checkpoint release over the weights keeps.
    pub fn spans(r: *const Rows, a: std.mem.Allocator) ![]const []const u8 {
        const out = try a.alloc([]const u8, r.shards.len * 3);
        for (r.shards, 0..) |s, i| {
            out[i * 3] = s.words;
            out[i * 3 + 1] = s.scales;
            out[i * 3 + 2] = s.biases;
        }
        return out;
    }

    /// Each id's packed words, scales and biases into the staging slices in id order; `w`'s parts warm first.
    pub fn gather(r: *const Rows, w: *Workers, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        const wb = r.wordBytes();
        const tb = r.tableBytes();
        if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
            return error.StagingTooSmall;
        const parts = @min(max_parts, @max(1, ids.len / rows_per_part));
        const call: Parts = .{
            .r = r,
            .w = w,
            .ids = ids,
            .words = words,
            .scales = scales,
            .biases = biases,
            .per = (ids.len + parts - 1) / parts,
        };
        // a gather of pages asked for before is a copy alone: the caller takes every part, waking no worker
        const p = if (w.cold(r, ids)) w.pool else null;
        const run = p orelse {
            for (0..parts) |i| try call.part(i);
            return;
        };
        try run.run(parts, .{ .ctx = &call, .run = Parts.task });
    }
};

/// The parked workers a gather spreads its page faults over (none: the caller gathers alone) and the pages it warms.
pub const Workers = struct {
    pool: ?*pool.Pool = null,
    warmer: host_pages.Warmer = .{},

    /// Whether a row of `ids` holds a page no gather asked for yet (a warm gather copies alone, waking no worker).
    pub fn cold(w: *Workers, r: *const Rows, ids: []const u32) bool {
        for (ids) |id| {
            const at = r.locate(id) catch continue;
            const sh = r.shards[at.shard];
            const wb = r.wordBytes();
            const tb = r.tableBytes();
            const region = at.shard * 3;
            if (w.warmer.cold(region, sh.words[at.row * wb ..][0..wb])) return true;
            if (w.warmer.cold(region + 1, sh.scales[at.row * tb ..][0..tb])) return true;
            if (w.warmer.cold(region + 2, sh.biases[at.row * tb ..][0..tb])) return true;
        }
        return false;
    }
};

/// One gather in parts of `per` ids: a part asks for its rows' pages, then copies them into its own staging rows.
const Parts = struct {
    r: *const Rows,
    w: *Workers,
    ids: []const u32,
    words: []u8,
    scales: []u8,
    biases: []u8,
    per: usize,

    fn task(ctx: *const anyopaque, i: usize) anyerror!void {
        const self: *const Parts = @ptrCast(@alignCast(ctx));
        return self.part(i);
    }

    fn part(self: *const Parts, i: usize) Error!void {
        const first = @min(self.ids.len, i * self.per);
        const last = @min(self.ids.len, first + self.per);
        const wb = self.r.wordBytes();
        const tb = self.r.tableBytes();
        // every page of the part is asked for before its first byte is read: the reads wait together, not one by one
        for (self.ids[first..last]) |id| self.ask(id);
        for (self.ids[first..last], first..) |id, n| {
            const at = try self.r.locate(id);
            const sh = self.r.shards[at.shard];
            @memcpy(self.words[n * wb ..][0..wb], sh.words[at.row * wb ..][0..wb]);
            @memcpy(self.scales[n * tb ..][0..tb], sh.scales[at.row * tb ..][0..tb]);
            @memcpy(self.biases[n * tb ..][0..tb], sh.biases[at.row * tb ..][0..tb]);
        }
    }

    /// Row `id`'s three pages, best effort: a row past the last shard is refused by the copy.
    fn ask(self: *const Parts, id: u32) void {
        const at = self.r.locate(id) catch return;
        const sh = self.r.shards[at.shard];
        const wb = self.r.wordBytes();
        const tb = self.r.tableBytes();
        // a shard's words, scales and biases are its three warmer regions, in this order
        const region = at.shard * 3;
        self.w.warmer.warm(region, sh.words[at.row * wb ..][0..wb]);
        self.w.warmer.warm(region + 1, sh.scales[at.row * tb ..][0..tb]);
        self.w.warmer.warm(region + 2, sh.biases[at.row * tb ..][0..tb]);
    }
};

/// The tables' rows: MLX affine as mapped (gathered packed), or FP4-Local's NVFP4 read from disk (gathered as bf16).
pub const Store = union(enum) { mlx: Rows, nvfp4: nvfp4.Rows };

pub const Tables = struct {
    store: Store,
    hasher: ngram.Hasher,
    /// The gather's parked workers and its pages' advice; freed on deinit.
    workers: Workers,
    /// The spans the warmer covers, on the arena `init` was given.
    live: []const []const u8,

    /// Values a row holds (one head's width).
    pub fn dims(t: Tables) usize {
        return switch (t.store) {
            .mlx => |r| r.dims,
            .nvfp4 => |r| r.hash.dh,
        };
    }

    /// Bytes of a gathered row: packed words (MLX) or its bf16 values (NVFP4), and of its scales and biases (MLX).
    pub fn wordBytes(t: Tables) usize {
        return switch (t.store) {
            .mlx => |r| r.wordBytes(),
            .nvfp4 => |r| r.hash.dh * 2,
        };
    }

    pub fn tableBytes(t: Tables) usize {
        return switch (t.store) {
            .mlx => |r| r.tableBytes(),
            .nvfp4 => 0,
        };
    }

    /// The tables of PLE layer `layer` as the reference splits them, with the gather's workers and page advice.
    pub fn init(
        a: std.mem.Allocator,
        ck: *core.checkpoint.Checkpoint,
        io: std.Io,
        c: *const config.Config,
        layer: usize,
        spelling: []const u8,
    ) !Tables {
        const heads = try c.ngramHeadCount();
        const ref = try config.pleRef(c);
        var total: usize = 0;
        for (ref.sizes) |s| total += @intCast(s);
        const rows = try std.math.divCeil(usize, total, c.ngram_divisor) * c.ngram_divisor;
        const base = rows / c.ngram_shards;
        const extra = rows % c.ngram_shards;
        const starts = try a.alloc(usize, c.ngram_shards + 1);
        starts[0] = 0;
        for (0..c.ngram_shards) |i| starts[i + 1] = starts[i] + base + @intFromBool(i < extra);
        if (c.format == .modelopt) {
            var t: Tables = .{ .store = .{ .nvfp4 = try nvfp4.Rows.init(a, ck, io, c, layer, spelling, starts) }, .hasher = try ngram.Hasher.init(c), .workers = .{}, .live = &.{} };
            t.live = try t.spans(a);
            t.workers.pool = try pool.Pool.create(a, io);
            t.store.nvfp4.pool = t.workers.pool;
            return t;
        }
        const shards = try a.alloc(Shard, c.ngram_shards);
        var spec: ?@import("../spec.zig").Spec = null;
        for (shards, 0..) |*s, i| {
            var lb: [128]u8 = undefined;
            var nb: [256]u8 = undefined;
            const stem = try a.dupe(u8, try tensors.ple.table(&nb, try tensors.layer(&lb, layer), spelling, i));
            const n = starts[i + 1] - starts[i];
            const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
            const sc = try ck.get(try std.fmt.allocPrint(a, "{s}.scales", .{stem}));
            const bi = try ck.get(try std.fmt.allocPrint(a, "{s}.biases", .{stem}));
            if (spec == null)
                spec = (try c.declared(stem, core.quant.keys.canonical)) orelse return error.UnsupportedQuantization;
            if (w.shape[0] != n or sc.dtype != .bf16 or bi.dtype != .bf16) return error.UnexpectedNgramTable;
            // a gather reads these rows in no order: readahead per row would fetch a window for it
            for ([_][]const u8{ "weight", "scales", "biases" }) |part|
                try ck.adviseRandom(try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, part }));
            s.* = .{ .words = w.bytes, .scales = sc.bytes, .biases = bi.bytes };
        }
        const width = c.ple_dim / heads;
        if (shards[0].words.len != (starts[1] - starts[0]) * width * spec.?.bits / 8) return error.UnexpectedNgramTable;
        const sizes = Rows.split(starts);
        var r: Rows = .{
            .shards = shards,
            .starts = starts,
            .wide = sizes.wide,
            .narrow = sizes.narrow,
            .extra = sizes.extra,
            .dims = width,
            .bits = spec.?.bits,
            .group = spec.?.group,
        };
        const live = try r.spans(a);
        errdefer a.free(live);
        var t: Tables = .{
            .store = .{ .mlx = r },
            .hasher = try ngram.Hasher.init(c),
            .workers = .{ .warmer = try host_pages.Warmer.init(a, live) },
            .live = live,
        };
        errdefer t.workers.warmer.deinit(a);
        t.workers.pool = try pool.Pool.create(a, io);
        return t;
    }

    /// The parked workers freed and the advice's pages forgotten.
    pub fn deinit(t: *Tables, a: std.mem.Allocator) void {
        if (t.workers.pool) |p| p.destroy();
        t.workers.warmer.deinit(a);
        a.free(t.live);
        t.* = undefined;
    }

    /// Each id's row into the staging slices in id order: packed words, scales and biases, or bf16 values in `words`.
    pub fn gather(t: *Tables, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        switch (t.store) {
            .mlx => |*r| return r.gather(&t.workers, ids, words, scales, biases),
            .nvfp4 => |*r| return r.gather(ids, @alignCast(std.mem.bytesAsSlice(u16, words[0 .. ids.len * t.wordBytes()]))),
        }
    }

    /// Every byte a gather reads at runtime: the pages a checkpoint release must leave mapped.
    pub fn spans(t: *const Tables, a: std.mem.Allocator) ![]const []const u8 {
        return switch (t.store) {
            .mlx => |*r| r.spans(a),
            .nvfp4 => |*r| r.spans(a),
        };
    }
};

test {
    _ = @import("tables_test.zig");
}
