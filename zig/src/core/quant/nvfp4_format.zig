//! NVFP4 routed and MTP experts as gfx1151 FP4-Local kernels read them (the n-gram table stays as stored, on disk).

const std = @import("std");
const nvfp4_slice = @import("../nvfp4_slice.zig");
const decode = @import("fp4local_decode.zig");
const Why = @import("../modelopt.zig").Why;
const refuse = @import("../modelopt.zig").refuse;

/// Inputs one e4m3 scale covers.
pub const group = 16;
/// Nibble i of a baked word holds the stored word's code `order[i]`.
pub const order = [8]u3{ 0, 2, 4, 6, 1, 3, 5, 7 };

/// A stored word (input i of eight in nibble i) baked: nibbles [c0, c2, c4, c6, c1, c3, c5, c7].
pub fn permute(w: u32) u32 {
    var out: u32 = 0;
    for (order, 0..) |from, i| out |= (w >> (@as(u5, from) * 4) & 0xF) << @intCast(4 * i);
    return out;
}

/// A baked word back to the stored order.
pub fn unpermute(w: u32) u32 {
    var out: u32 = 0;
    for (order, 0..) |to, i| out |= (w >> @intCast(4 * i) & 0xF) << (@as(u5, to) * 4);
    return out;
}

/// Codes 2p and 2p + 1 of a baked word, as the kernel takes them: (w >> 4p) & 0x000F000F, low half then high.
pub fn pair(w: u32, p: u2) [2]u4 {
    const v = w >> (@as(u5, p) * 4) & 0x000F000F;
    return .{ @truncate(v), @truncate(v >> 16) };
}

/// One projection baked: words [n][k / 8], scales e4m3 [n][k / 16], and its weight_scale_2; owned.
pub const Projection = struct {
    n: usize,
    k: usize,
    words: []u32,
    scales: []u8,
    global: f32,

    pub fn deinit(p: *Projection, a: std.mem.Allocator) void {
        a.free(p.words);
        a.free(p.scales);
        p.* = undefined;
    }

    pub fn bytes(p: Projection) usize {
        return p.words.len * 4 + p.scales.len + 4;
    }

    /// The code of input `col` in row `row`.
    pub fn code(p: Projection, row: usize, col: usize) u4 {
        const w = p.words[row * (p.k / 8) + col / 8];
        return pair(w, @intCast(col % 8 / 2))[col % 2];
    }

    /// The bf16 weight the kernels dot at (row, col), before the global.
    pub fn weight(p: Projection, row: usize, col: usize) u16 {
        return decode.nvfp4Weight(p.code(row, col), p.scales[row * (p.k / group) + col / group]);
    }
};

/// A projection's device bytes: n x k / 2 of codes, n x k / 16 of scales and its fp32 global.
pub fn bytes(n: usize, k: usize) usize {
    return n * k / 2 + n * (k / group) + 4;
}

/// `name`'s stored tensors baked (CSF decoded, codes permuted); refuses bad shapes, NaN scales, a bad global by name.
pub fn bake(a: std.mem.Allocator, source: nvfp4_slice.Source, name: []const u8, why: ?*Why) !Projection {
    const d = nvfp4_slice.dimensions(source) catch |err| return refuse(why, err, "{s}: not ModelOpt NVFP4 ({t})", .{ name, err });
    if (d.k % decode.fold != 0) return refuse(why, error.UnexpectedTensor, "{s}: K {d} is not whole folds of {d}", .{ name, d.k, decode.fold });
    var linear = try nvfp4_slice.slice(a, source, .{ .from = 0, .to = d.n }, .{ .from = 0, .to = d.k });
    defer a.free(linear.codes);
    errdefer a.free(linear.scales);
    const g = linear.global;
    if (!std.math.isFinite(g) or g <= 0 or !std.math.isFinite(decode.nvfp4Global(g)))
        return refuse(why, error.UnsupportedQuantization, "{s}: weight_scale_2 {e} is not finite and positive", .{ name, g });
    for (linear.scales, 0..) |s, i| if (s & 0x7f == 0x7f)
        return refuse(why, error.UnsupportedQuantization, "{s}: row {d} group {d}: NaN e4m3 scale", .{ name, i / (d.k / group), i % (d.k / group) });
    const words = try a.alloc(u32, d.n * d.k / 8);
    for (words, 0..) |*w, i| w.* = permute(std.mem.readInt(u32, linear.codes[4 * i ..][0..4], .little));
    return .{ .n = d.n, .k = d.k, .words = words, .scales = linear.scales, .global = g };
}

/// The stored code bytes of a baked projection (input 2j in byte j's low nibble): the inverse of `bake`'s permute.
pub fn codes(a: std.mem.Allocator, p: Projection) ![]u8 {
    const out = try a.alloc(u8, p.words.len * 4);
    for (p.words, 0..) |w, i| std.mem.writeInt(u32, out[4 * i ..][0..4], unpermute(w), .little);
    return out;
}

/// Experts stacked [E][member rows in turn][k / 8] (gate | up with `per` 2), scales alike, globals [E][per]; owned.
pub const Stack = struct {
    count: usize,
    /// Rows an expert holds: its members' rows summed.
    rows: usize,
    k: usize,
    words: []u32,
    scales: []u8,
    globals: []f32,

    pub fn deinit(s: *Stack, a: std.mem.Allocator) void {
        a.free(s.words);
        a.free(s.scales);
        a.free(s.globals);
        s.* = undefined;
    }
};

/// `members` in expert order, `per` an expert (gate then up, or down alone); member i of every expert keeps its rows.
pub fn stack(a: std.mem.Allocator, members: []const Projection, per: usize) !Stack {
    if (per == 0 or members.len == 0 or members.len % per != 0) return error.UnexpectedTensor;
    const k = members[0].k;
    var rows: usize = 0;
    for (members[0..per]) |m| rows += m.n;
    for (members, 0..) |m, i| if (m.k != k or m.n != members[i % per].n) return error.UnexpectedTensor;
    const count = members.len / per;
    const words = try a.alloc(u32, count * rows * (k / 8));
    errdefer a.free(words);
    const scales = try a.alloc(u8, count * rows * (k / group));
    errdefer a.free(scales);
    const globals = try a.alloc(f32, members.len);
    var w: usize = 0;
    var s: usize = 0;
    for (members, globals) |m, *g| {
        @memcpy(words[w..][0..m.words.len], m.words);
        @memcpy(scales[s..][0..m.scales.len], m.scales);
        w += m.words.len;
        s += m.scales.len;
        g.* = m.global;
    }
    return .{ .count = count, .rows = rows, .k = k, .words = words, .scales = scales, .globals = globals };
}

test {
    _ = @import("nvfp4_format_test.zig");
}
