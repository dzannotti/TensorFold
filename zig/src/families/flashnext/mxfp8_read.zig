//! MXFP8 codes and scales read from the store as stored, and row ranges copied in order; no backend upload.
const std = @import("std");
const core = @import("core");

/// One rank's [from, to) share of a width split evenly over `world`, each share a whole number of 32-wide groups.
pub const Cut = struct {
    from: usize,
    to: usize,

    pub fn init(total_width: usize, world: usize, rank: usize) !Cut {
        if (world == 0 or rank >= world or total_width == 0 or total_width % world != 0) return error.UnevenExpertSplit;
        const part = total_width / world;
        if (part % 32 != 0) return error.UnevenExpertSplit;
        return .{ .from = rank * part, .to = (rank + 1) * part };
    }

    pub fn width(self: Cut) usize {
        return self.to - self.from;
    }
};

pub const Micro = struct {
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,

    pub fn deinit(self: *Micro, a: std.mem.Allocator) void {
        a.free(self.codes);
        a.free(self.scales);
        self.* = undefined;
    }
};

/// Own each plane before the store advances to a possibly different mapped shard.
pub fn readMicro(a: std.mem.Allocator, store: anytype, prefix: []const u8) !Micro {
    var buf: [512]u8 = undefined;
    const w = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    if (w.dtype != .f8_e4m3 or w.rank != 2 or w.dim(0) == 0 or w.dim(1) == 0 or w.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = w.dim(0);
    const k = w.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (w.bytes.len != count) return error.UnexpectedTensor;
    const codes = try a.dupe(u8, w.bytes);
    errdefer a.free(codes);
    const s = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{prefix}));
    if (!s.is(.u8, &.{ n, k / 32 }) or s.bytes.len != count / 32) return error.UnexpectedTensor;
    return .{ .n = n, .k = k, .codes = codes, .scales = try a.dupe(u8, s.bytes) };
}

/// Concatenate output rows of compatible projections without converting or recalibrating either plane.
pub fn readStitched(a: std.mem.Allocator, store: anytype, prefix: []const u8, names: []const []const u8) !Micro {
    if (names.len == 0 or names.len > 6) return error.UnexpectedTensor;
    var parts: [6]Micro = undefined;
    var held: usize = 0;
    defer for (parts[0..held]) |*part| part.deinit(a);
    var n: usize = 0;
    var k: usize = 0;
    var buf: [512]u8 = undefined;
    for (names) |name| {
        parts[held] = try readMicro(a, store, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }));
        held += 1;
        const part = parts[held - 1];
        if (k != 0 and k != part.k) return error.UnexpectedTensor;
        k = part.k;
        n = try std.math.add(usize, n, part.n);
    }
    const count = try std.math.mul(usize, n, k);
    const codes = try a.alloc(u8, count);
    errdefer a.free(codes);
    const scales = try a.alloc(u8, count / 32);
    var row: usize = 0;
    for (parts[0..held]) |part| {
        @memcpy(codes[row * k ..][0..part.codes.len], part.codes);
        @memcpy(scales[row * (k / 32) ..][0..part.scales.len], part.scales);
        row += part.n;
    }
    return .{ .n = n, .k = k, .codes = codes, .scales = scales };
}

/// Up to four [from, to) output-row ranges kept of one stitched part, copied in order.
pub const Ranges = struct {
    cuts: [4]Cut = undefined,
    len: usize = 0,

    pub fn of(cuts: []const Cut) Ranges {
        std.debug.assert(cuts.len <= 4);
        var out: Ranges = .{ .len = cuts.len };
        @memcpy(out.cuts[0..cuts.len], cuts);
        return out;
    }

    pub fn items(self: *const Ranges) []const Cut {
        return self.cuts[0..self.len];
    }

    pub fn rows(self: Ranges) usize {
        var n: usize = 0;
        for (self.items()) |c| n += c.width();
        return n;
    }
};

/// Copy `kept` rows of a row-major `src` into the front of `dst`, in order; returns the bytes written.
pub fn copyRanges(dst: []u8, src: []const u8, row_bytes: usize, kept: Ranges) !usize {
    var at: usize = 0;
    for (kept.items()) |c| {
        if (c.from >= c.to or try std.math.mul(usize, c.to, row_bytes) > src.len) return error.UnexpectedTensor;
        const len = c.width() * row_bytes;
        if (len > dst.len - at) return error.UnexpectedTensor;
        @memcpy(dst[at..][0..len], src[c.from * row_bytes ..][0..len]);
        at += len;
    }
    return at;
}

test "intermediate-channel cuts cover the source once and preserve 32-input kernel groups" {
    const a = try Cut.init(640, 2, 0);
    const b = try Cut.init(640, 2, 1);
    try std.testing.expectEqual(@as(usize, 0), a.from);
    try std.testing.expectEqual(@as(usize, 320), a.to);
    try std.testing.expectEqual(a.to, b.from);
    try std.testing.expectEqual(@as(usize, 640), b.to);
    try std.testing.expectEqual(a.width(), b.width());
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 0, 0));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 2, 2));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 3, 0));
    try std.testing.expectError(error.UnevenExpertSplit, Cut.init(640, 32, 0));
}

const Tensor = core.safetensors.Tensor;

const Fake = struct {
    scratch: [1024]u8 = undefined,
    bad: enum { none, weight, scale, extent, width, missing } = .none,

    pub fn tensor(self: *Fake, name: []const u8) !Tensor {
        const part: usize = if (std.mem.startsWith(u8, name, "root.a.")) 0 else if (std.mem.startsWith(u8, name, "root.b.")) 1 else return error.MissingTensor;
        const n: usize = if (part == 0) 3 else 5;
        const k: usize = if (self.bad == .width and part == 1) 128 else 64;
        const scale = std.mem.endsWith(u8, name, ".weight_scale");
        if (scale and self.bad == .missing) return error.MissingTensor;
        if (!scale and !std.mem.endsWith(u8, name, ".weight")) return error.MissingTensor;
        const cols = if (scale) k / 32 else k;
        const len = n * cols;
        @memset(&self.scratch, 0xee);
        for (self.scratch[0..len], 0..) |*v, i| v.* = @truncate(i * 17 + part * 53 + @as(usize, if (scale) 129 else 7));
        return .{
            .dtype = if (scale) (if (self.bad == .scale) .f8_e4m3 else .u8) else (if (self.bad == .weight) .bf16 else .f8_e4m3),
            .rank = 2,
            .shape = .{ n, cols, 1, 1, 1 },
            .bytes = self.scratch[0 .. len - @as(usize, if (self.bad == .extent) 1 else 0)],
        };
    }
};

fn trial(a: std.mem.Allocator) !void {
    var fake: Fake = .{};
    var value = try readStitched(a, &fake, "root", &.{ ".a", ".b" });
    defer value.deinit(a);
    try std.testing.expectEqual(@as(usize, 8), value.n);
    try std.testing.expectEqual(@as(usize, 64), value.k);
    for (0..2) |part| {
        const start: usize = if (part == 0) 0 else 3;
        const n: usize = if (part == 0) 3 else 5;
        for (0..n * 64) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 7)), value.codes[start * 64 + i]);
        for (0..n * 2) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 129)), value.scales[start * 2 + i]);
    }
}

test "MXFP8 fusion owns each shard plane and preserves exact output row order" {
    try trial(std.testing.allocator);
}

test "MXFP8 fusion cleans every partial allocation on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, trial, .{});
}

test "MXFP8 source refuses incompatible shapes, dtypes, extents and missing scales" {
    const a = std.testing.allocator;
    for ([_]@FieldType(Fake, "bad"){ .weight, .scale, .extent, .width }) |bad| {
        var fake: Fake = .{ .bad = bad };
        try std.testing.expectError(error.UnexpectedTensor, readStitched(a, &fake, "root", &.{ ".a", ".b" }));
    }
    var fake: Fake = .{ .bad = .missing };
    try std.testing.expectError(error.MissingTensor, readMicro(a, &fake, "root.a"));
    try std.testing.expectError(error.UnexpectedTensor, readStitched(a, &fake, "root", &.{}));
    try std.testing.expectError(error.UnexpectedTensor, readStitched(a, &fake, "root", &.{ ".a", ".a", ".a", ".a", ".a", ".a", ".a" }));
}

test "kept row ranges copy in order and a range past the source or the destination is refused" {
    const src = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var dst: [8]u8 = @splat(0xee);
    const kept = Ranges.of(&.{ .{ .from = 4, .to = 6 }, .{ .from = 0, .to = 1 } });
    try std.testing.expectEqual(@as(usize, 3), kept.rows());
    try std.testing.expectEqual(@as(usize, 6), try copyRanges(&dst, &src, 2, kept));
    try std.testing.expectEqualSlices(u8, &.{ 8, 9, 10, 11, 0, 1, 0xee, 0xee }, &dst);
    try std.testing.expectError(error.UnexpectedTensor, copyRanges(&dst, &src, 2, Ranges.of(&.{.{ .from = 5, .to = 7 }})));
    try std.testing.expectError(error.UnexpectedTensor, copyRanges(dst[0..3], &src, 2, Ranges.of(&.{.{ .from = 0, .to = 2 }})));
    try std.testing.expectError(error.UnexpectedTensor, copyRanges(&dst, &src, 2, Ranges.of(&.{.{ .from = 2, .to = 2 }})));
}
