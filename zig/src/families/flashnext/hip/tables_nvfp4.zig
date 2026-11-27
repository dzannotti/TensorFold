//! FP4-Local's NVFP4 n-gram table on disk: a gather's distinct rows read and decoded once each to bf16, as on CUDA.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const names = @import("../names_modelopt.zig");
const distinct = @import("../cuda_ple_gather.zig");
const pool = @import("../host_pool.zig");
const Split = @import("tables.zig").Rows.Split;
const split = @import("tables.zig").Rows.split;

const Table = core.nvfp4_rows.Table;
const Allocator = std.mem.Allocator;

/// TF_FLASHNEXT_PLE_IO as the CUDA reader takes it: pread through the page cache (Linux's default) or the mapping.
pub const Mode = enum { mmap, pread };

pub fn mode(text: ?[]const u8) !Mode {
    const v = text orelse return if (@import("builtin").os.tag == .linux) .pread else .mmap;
    return std.meta.stringToEnum(Mode, v) orelse error.BadPleIoMode;
}

/// One shard's rows read by position from its open file: a row's codes, then its scales.
const Positional = struct {
    file: std.Io.File,
    io: std.Io,
    codes_at: u64,
    scales_at: u64,

    fn read(p: Positional, t: Table, row: usize, out: []u16) !void {
        var bytes: [512]u8 = undefined;
        const cb = t.width / 2;
        const sb = t.width / 16;
        if (row >= t.rows or out.len != t.width or cb + sb > bytes.len) return error.BadNvfp4Rows;
        if (try p.file.readPositionalAll(p.io, bytes[0..cb], p.codes_at + row * cb) != cb) return error.PleShortRead;
        if (try p.file.readPositionalAll(p.io, bytes[cb..][0..sb], p.scales_at + row * sb) != sb) return error.PleShortRead;
        const local: Table = .{ .codes = bytes[0..cb], .scales = bytes[cb..][0..sb], .rows = 1, .width = t.width, .global = t.global };
        try local.gather(&.{0}, out);
    }
};

/// The shards' rows as stored (codes and e4m3 scales, one global), read by pread (TF_FLASHNEXT_PLE_IO=mmap: mapped).
pub const Rows = struct {
    /// Thread-safe: a gather's index buffers come from it on the caller's thread.
    gpa: Allocator,
    pool: ?*pool.Pool,
    /// A row's bf16 values (one head's width), named as the CUDA reader's hash names them.
    hash: struct { dh: usize },
    starts: []const usize,
    sizes: Split,
    tables: []Table,
    readers: []Positional,
    mode: Mode,

    /// PLE layer `layer`'s shards under `spelling` from the mapped checkpoint; nothing of the table is read here.
    pub fn init(a: Allocator, ck: *core.checkpoint.Checkpoint, io: std.Io, c: *const config.Config, layer: usize, spelling: []const u8, starts: []const usize) !Rows {
        const heads = try c.ngramHeadCount();
        const dh = c.ple_dim / heads;
        var lb: [128]u8 = undefined;
        var sb: [160]u8 = undefined;
        const stem = try names.exported(try tensors.layer(&lb, layer), &sb);
        const global_name = try std.fmt.allocPrint(a, "{s}.ple.ple_embedding.ngram_embedding.weight_scale_2", .{stem});
        const g = try ck.get(global_name);
        if (!g.is(.f32, &.{}) and !g.is(.f32, &.{1})) return error.UnexpectedNgramTable;
        const global: f32 = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little));
        if (!std.math.isFinite(global) or global <= 0) return error.UnexpectedNgramTable;
        const n = c.ngram_shards;
        const tables = try a.alloc(Table, n);
        const readers = try a.alloc(Positional, n);
        for (0..n) |i| {
            var nb: [256]u8 = undefined;
            const shard = try a.dupe(u8, try tensors.ple.table(&nb, stem, spelling, i));
            const wname = try std.fmt.allocPrint(a, "{s}.weight", .{shard});
            const sname = try std.fmt.allocPrint(a, "{s}.weight_scale", .{shard});
            const w = try ck.get(wname);
            const s = try ck.get(sname);
            if (w.dim(0) != starts[i + 1] - starts[i] or w.dim(1) * 2 != dh) return error.UnexpectedNgramTable;
            tables[i] = try Table.init(w, s, global);
            const f = fileOf(ck, wname) orelse return error.UnexpectedNgramTable;
            if (fileOf(ck, sname) != f) return error.UnexpectedNgramTable;
            const base = @intFromPtr(f.map.memory.ptr);
            readers[i] = .{ .file = f.file, .io = io, .codes_at = @intFromPtr(w.bytes.ptr) - base, .scales_at = @intFromPtr(s.bytes.ptr) - base };
            // a gather reads these rows in no order: readahead per row would fetch a window for it
            try ck.adviseRandom(wname);
            try ck.adviseRandom(sname);
        }
        const how = try mode(@import("sys").environment.rawText("TF_FLASHNEXT_PLE_IO"));
        return .{ .gpa = std.heap.page_allocator, .pool = null, .hash = .{ .dh = dh }, .starts = starts, .sizes = split(starts), .tables = tables, .readers = readers, .mode = how };
    }

    /// Row `id`'s shard and its row in it.
    pub fn locate(r: *const Rows, id: usize) !struct { shard: usize, row: usize } {
        if (id >= r.starts[r.tables.len]) return error.NgramRowOutOfRange;
        const z = r.sizes;
        if (z.wide != 0) {
            const first = z.wide * z.extra;
            if (id < first) return .{ .shard = id / z.wide, .row = id % z.wide };
            return .{ .shard = z.extra + (id - first) / z.narrow, .row = (id - first) % z.narrow };
        }
        var s: usize = 0;
        while (r.starts[s + 1] <= id) s += 1;
        return .{ .shard = s, .row = id - r.starts[s] };
    }

    /// Rows [lo, hi) of `ids` decoded into their bf16 rows of `out` (cuda_ple_gather's reader).
    pub fn fill(r: *const Rows, ids: []const i64, out: []u16, lo: usize, hi: usize) !void {
        const dh = r.hash.dh;
        for (lo..hi) |i| {
            const at = try r.locate(@intCast(ids[i]));
            const row = out[i * dh ..][0..dh];
            switch (r.mode) {
                .pread => try r.readers[at.shard].read(r.tables[at.shard], at.row, row),
                .mmap => try r.tables[at.shard].gather(&.{at.row}, row),
            }
        }
    }

    /// Each id's bf16 row into `out` in id order: distinct rows decoded once on the parked pool, duplicates copied.
    pub fn gather(r: *const Rows, ids: []const u32, out: []u16) !void {
        if (out.len != ids.len * r.hash.dh) return error.StagingTooSmall;
        if (ids.len == 0) return;
        const wide = try r.gpa.alloc(i64, ids.len);
        defer r.gpa.free(wide);
        for (wide, ids) |*w, id| w.* = id;
        return distinct.run(r, wide, out);
    }

    /// Every byte a mapped gather reads: the codes and scales of each shard.
    pub fn spans(r: *const Rows, a: Allocator) ![]const []const u8 {
        const out = try a.alloc([]const u8, r.tables.len * 2);
        for (r.tables, 0..) |t, i| {
            out[2 * i] = t.codes;
            out[2 * i + 1] = t.scales;
        }
        return out;
    }
};

/// The open shard file that holds `name`.
fn fileOf(ck: *core.checkpoint.Checkpoint, name: []const u8) ?*const core.safetensors.File {
    for (ck.files.items) |*f| if (f.get(name) != null) return f;
    return null;
}

test "TF_FLASHNEXT_PLE_IO: pread or mmap by name, the platform's default unset, anything else refused" {
    try std.testing.expectEqual(Mode.mmap, try mode("mmap"));
    try std.testing.expectEqual(Mode.pread, try mode("pread"));
    for ([_][]const u8{ "", "auto", "0" }) |bad| try std.testing.expectError(error.BadPleIoMode, mode(bad));
}

test "a positional row read decodes the bytes the mapped table holds" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var codes: [64 * 16]u8 = undefined;
    var scales: [64 * 2]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @truncate(i * 37 + 5);
    for (&scales, 0..) |*c, i| c.* = @truncate(0x30 + i % 16);
    var data: [3 + codes.len + scales.len]u8 = undefined;
    @memset(data[0..3], 0x42);
    @memcpy(data[3..][0..codes.len], &codes);
    @memcpy(data[3 + codes.len ..], &scales);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "table", .data = &data });
    const file = try tmp.dir.openFile(t.io, "table", .{});
    defer file.close(t.io);
    const table: Table = .{ .codes = &codes, .scales = &scales, .rows = 64, .width = 32, .global = 0.75 };
    const p: Positional = .{ .file = file, .io = t.io, .codes_at = 3, .scales_at = 3 + codes.len };
    for (0..64) |row| {
        var want: [32]u16 = undefined;
        var got: [32]u16 = undefined;
        try table.gather(&.{row}, &want);
        try p.read(table, row, &got);
        try t.expectEqualSlices(u16, &want, &got);
    }
}
