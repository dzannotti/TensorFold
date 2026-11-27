//! FP4-Local's baked NVFP4 and row-stitched MXFP8 as core/quant formats: host planes, device buffers, handles.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const nvfp4_format = @import("nvfp4_format.zig");
const mxfp8_format = @import("mxfp8_format.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;

/// The baked planes are cut by the loader before the bake: a cut of a baked projection is refused by name.
fn uncut(what: []const u8) slice.Error {
    std.log.err("{s}: FP4-Local's baked planes are not cut after the bake (tensor parallelism is refused on load)", .{what});
    return error.UnevenSplit;
}

/// NVFP4 baked (nvfp4_format): words [(E,) n, k / 8], e4m3 scales [(E,) n, k / 16], f32 globals [(E,) per].
pub const nvfp4 = struct {
    pub const id: types.Format = .nvfp4;
    pub const decoder: types.Decoder = .nvfp4;
    pub const Config = void;

    pub const Host = struct { words: Tensor, scales: Tensor, globals: Tensor };

    /// One baked projection, borrowed: its global the one entry of `globals`.
    pub fn one(p: *const nvfp4_format.Projection) Host {
        return .{
            .words = types.view(.u32, &.{ p.n, p.k / 8 }, std.mem.sliceAsBytes(p.words)),
            .scales = types.view(.u8, &.{ p.n, p.k / nvfp4_format.group }, p.scales),
            .globals = types.view(.f32, &.{1}, std.mem.asBytes(&p.global)),
        };
    }

    /// A baked stack, borrowed: `per` globals an expert (2: gate | up, 1: down).
    pub fn stacked(s: nvfp4_format.Stack) Host {
        const per = s.globals.len / s.count;
        return .{
            .words = types.view(.u32, &.{ s.count, s.rows, s.k / 8 }, std.mem.sliceAsBytes(s.words)),
            .scales = types.view(.u8, &.{ s.count, s.rows, s.k / nvfp4_format.group }, s.scales),
            .globals = types.view(.f32, &.{ s.count, per }, std.mem.sliceAsBytes(s.globals)),
        };
    }

    pub fn rows(h: Host) usize {
        return h.words.dim(h.words.rank - 2);
    }

    pub fn bytes(h: Host) usize {
        return h.words.bytes.len + h.scales.bytes.len + h.globals.bytes.len;
    }

    pub fn halves(_: Allocator, _: Host) ![2]Host {
        return uncut("NVFP4 halves");
    }

    pub fn sliceRows(_: Allocator, _: Host, _: []const slice.Span) slice.Error!Host {
        return uncut("NVFP4 rows");
    }

    pub fn sliceCols(_: Allocator, _: Host, _: slice.Rank, what: []const u8) slice.Error!Host {
        return uncut(what);
    }

    pub fn sliceStack(_: Allocator, _: Host, _: slice.Rank, _: usize) slice.Error!Host {
        return uncut("NVFP4 experts");
    }

    pub const Device = struct { words: types.Buf, scales: types.Buf, globals: types.Buf };

    pub fn upload(u: types.Uploader, h: Host) !Device {
        return .{ .words = try u.tensor(h.words), .scales = try u.tensor(h.scales), .globals = try u.tensor(h.globals) };
    }

    /// The planes' addresses, and the globals an expert holds (gate | up: 2, the gate's rows first).
    pub const Matrix = struct { words: u64, scales: u64, globals: u64, per: u32 };

    pub const View = struct { n: u32, k: u32, matrix: Matrix };

    pub fn view(d: Device, is_stack: bool) error{UnsupportedTables}!View {
        const lead: usize = @intFromBool(is_stack);
        if (d.words.rank != 2 + lead or d.globals.rank != 1 + lead or d.scales.rank != d.words.rank) return error.UnsupportedTables;
        return .{
            .n = @intCast(d.words.dim(lead)),
            .k = @intCast(d.words.dim(lead + 1) * 8),
            .matrix = .{ .words = d.words.ptr, .scales = d.scales.ptr, .globals = d.globals.ptr, .per = @intCast(d.globals.dim(lead)) },
        };
    }
};

/// MXFP8 as stored and row-stitched (mxfp8_format): e4m3 codes [n, k], e8m0 scales [n, k / 32].
pub const mxfp8 = struct {
    pub const id: types.Format = .mxfp8;
    pub const decoder: types.Decoder = .mxfp8;
    pub const Config = void;

    pub const Host = struct { codes: Tensor, scales: Tensor };

    /// Planes [n, k] and [n, k / 32] over `codes` and `scales`, borrowed; admitted (mxfp8_format.check) by the caller.
    pub fn of(n: usize, k: usize, codes: []const u8, scales: []const u8) Host {
        return .{ .codes = types.view(.f8_e4m3, &.{ n, k }, codes), .scales = types.view(.u8, &.{ n, k / mxfp8_format.block }, scales) };
    }

    pub fn rows(h: Host) usize {
        return h.codes.shape[0];
    }

    pub fn bytes(h: Host) usize {
        return h.codes.bytes.len + h.scales.bytes.len;
    }

    pub fn halves(_: Allocator, _: Host) ![2]Host {
        return uncut("MXFP8 halves");
    }

    pub fn sliceRows(_: Allocator, _: Host, _: []const slice.Span) slice.Error!Host {
        return uncut("MXFP8 rows");
    }

    pub fn sliceCols(_: Allocator, _: Host, _: slice.Rank, what: []const u8) slice.Error!Host {
        return uncut(what);
    }

    pub fn sliceStack(_: Allocator, _: Host, _: slice.Rank, _: usize) slice.Error!Host {
        return uncut("MXFP8 experts");
    }

    pub const Device = struct { codes: types.Buf, scales: types.Buf };

    pub fn upload(u: types.Uploader, h: Host) !Device {
        return .{ .codes = try u.tensor(h.codes), .scales = try u.tensor(h.scales) };
    }

    pub const Matrix = struct { codes: u64, scales: u64 };

    pub const View = struct { n: u32, k: u32, matrix: Matrix };

    pub fn view(d: Device, is_stack: bool) error{UnsupportedTables}!View {
        if (is_stack or d.codes.rank != 2 or d.codes.dtype != .f8_e4m3) return error.UnsupportedTables;
        return .{ .n = @intCast(d.codes.dim(0)), .k = @intCast(d.codes.dim(1)), .matrix = .{ .codes = d.codes.ptr, .scales = d.scales.ptr } };
    }
};

const Fake = struct {
    next: u64 = 0x1000,

    fn put(ctx: *anyopaque, t: Tensor) anyerror!types.Buf {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        defer f.next += 0x1000;
        return .{ .ptr = f.next, .len = t.bytes.len, .dtype = t.dtype, .rank = t.rank, .shape = t.shape };
    }
};

test "FP4-Local handles: a baked NVFP4 projection and stack, and an MXFP8 matrix, keep their shapes and planes" {
    var f: Fake = .{};
    const u: types.Uploader = .{ .ctx = &f, .put = Fake.put };
    var words: [2 * 3 * 16]u32 = @splat(0);
    var scales: [2 * 3 * 8]u8 = @splat(0x38);
    var globals = [_]f32{ 1, 2, 3, 4 };
    const s: nvfp4_format.Stack = .{ .count = 2, .rows = 3, .k = 128, .words = &words, .scales = &scales, .globals = &globals };
    const h = nvfp4.stacked(s);
    try std.testing.expectEqual(@as(usize, 3), nvfp4.rows(h));
    try std.testing.expectEqual(@as(usize, words.len * 4 + scales.len + 16), nvfp4.bytes(h));
    const v = try nvfp4.view(try nvfp4.upload(u, h), true);
    try std.testing.expectEqual(@as(u32, 3), v.n);
    try std.testing.expectEqual(@as(u32, 128), v.k);
    try std.testing.expectEqual(@as(u32, 2), v.matrix.per);
    const p: nvfp4_format.Projection = .{ .n = 3, .k = 128, .words = words[0..48], .scales = scales[0..24], .global = 0.5 };
    const single = try nvfp4.view(try nvfp4.upload(u, nvfp4.one(&p)), false);
    try std.testing.expectEqual(@as(u32, 1), single.matrix.per);
    try std.testing.expectError(error.UnsupportedTables, nvfp4.view(try nvfp4.upload(u, nvfp4.one(&p)), true));
    var codes: [3 * 64]u8 = @splat(0x38);
    const e8: [3 * 2]u8 = @splat(127);
    const m = mxfp8.of(3, 64, &codes, &e8);
    const mv = try mxfp8.view(try mxfp8.upload(u, m), false);
    try std.testing.expectEqual(@as(u32, 3), mv.n);
    try std.testing.expectEqual(@as(u32, 64), mv.k);
}
