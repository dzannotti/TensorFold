//! ciru_q8 (`ciru_h128_w8_g256_bf16_v1`): routed experts as int8 codes of a 128-block Hadamard rotation, read into MLX
//! 8-bit affine stacks the existing tiles run; every other tensor of the checkpoint is plain bf16 (the dense format).
//!
//! The stored math, per expert, of each W [N, K] (gate, up: K = hidden; down: K = moe width):
//!   W' = W H, H = blockdiag over K of 128-wide Sylvester Hadamards scaled by 1/sqrt(128) (H = H^T = H^-1)
//!   W'[n, k] = s[n, k / 256] q[n, k], q int8 in [-127, 127], s fp16 (subnormals allowed), no bias or zero point
//!   codes [E, N/16, K/32, 2, 16, 16]: q[nt, kt, h, r, c] is W'[16 nt + r, 32 kt + 16 h + c]
//!   scales [E, N/16, K/256, 16]:      s[nt, g, r]       is s[16 nt + r, g]
//! so W x = W' (H x): the products read x rotated by 128-blocks, and the down reads silu(gate) * up rotated.
//! As MLX 8-bit group 128: code = q + 128, scale = s, bias = -128 s (both exact in fp16), each 256-group's term twice.
//! The bf16 shared expert joins the stack as its last expert, rotated the same way and rounded to 8-bit group 128
//! (the stack has one format): the one tensor this reader does not keep exactly.

const std = @import("std");
const types = @import("types.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const format = "ciru_h128_w8_g256_bf16_v1";
pub const block = 128;
const group = 256;
/// The MLX group the stacks are read with: the kernels' widest (a 256-group's scale written for both halves).
pub const mlx_group = 128;

/// Whether the config's quantization object declares ciru_q8; a ciru_q8 of other parameters is refused.
pub fn declared(obj: std.json.ObjectMap) error{UnsupportedQuantization}!bool {
    const method = obj.get("quant_method") orelse return false;
    if (method != .string or !std.mem.eql(u8, method.string, "ciru_q8")) return false;
    const want = .{ .{ "group_size", group }, .{ "transform_block", block }, .{ "routed_weight_bits", 8 } };
    inline for (want) |w| {
        const v = obj.get(w[0]) orelse return error.UnsupportedQuantization;
        if (v != .integer or v.integer != w[1]) return error.UnsupportedQuantization;
    }
    const f = obj.get("checkpoint_format") orelse return error.UnsupportedQuantization;
    if (f != .string or !std.mem.eql(u8, f.string, format)) return error.UnsupportedQuantization;
    return true;
}

/// x H in place over 128-blocks of `x`: fp32 Sylvester butterflies at steps 1, 2, .. 64, then times 1 / sqrt(128).
pub fn rotate(x: []f32) void {
    const inv: f32 = @bitCast(@as(u32, 0x3db504f3));
    var at: usize = 0;
    while (at < x.len) : (at += block) {
        const v = x[at..][0..block];
        var h: usize = 1;
        while (h < block) : (h *= 2) {
            var i: usize = 0;
            while (i < block) : (i += 2 * h) for (i..i + h) |j| {
                const a = v[j];
                const b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            };
        }
        for (v) |*e| e.* *= inv;
    }
}

/// One stack under construction: E + 1 experts of `rows` rows of K codes, 16-bit tables a 128-group.
const Stack = struct {
    words: []u8,
    scales: []u16,
    biases: []u16,
    rows: usize,
    k: usize,

    fn init(a: Allocator, count: usize, rows: usize, k: usize) !Stack {
        return .{
            .words = try a.alloc(u8, count * rows * k),
            .scales = try a.alloc(u16, count * rows * (k / mlx_group)),
            .biases = try a.alloc(u16, count * rows * (k / mlx_group)),
            .rows = rows,
            .k = k,
        };
    }

    fn host(s: Stack, count: usize) Tensor3 {
        const g = s.k / mlx_group;
        return .{
            .words = .{ .dtype = .i32, .rank = 3, .shape = .{ count, s.rows, s.k / 4, 1, 1 }, .bytes = s.words },
            .scales = .{ .dtype = .f16, .rank = 3, .shape = .{ count, s.rows, g, 1, 1 }, .bytes = std.mem.sliceAsBytes(s.scales) },
            .biases = .{ .dtype = .f16, .rank = 3, .shape = .{ count, s.rows, g, 1, 1 }, .bytes = std.mem.sliceAsBytes(s.biases) },
        };
    }

    /// Expert `e`'s rows from `row0` of the stored codes and scales of routed expert `e`.
    fn routed(s: Stack, cs: Tensor, scales: Tensor, e: usize, row0: usize) void {
        const nt_n = cs.shape[1];
        const kt_n = cs.shape[2];
        const groups = s.k / group;
        const base = e * s.rows;
        const src = cs.bytes[e * nt_n * kt_n * 512 ..][0 .. nt_n * kt_n * 512];
        var i: usize = 0;
        for (0..nt_n) |nt| for (0..kt_n) |kt| for (0..2) |h| for (0..16) |r| {
            const dst = s.words[(base + row0 + 16 * nt + r) * s.k + 32 * kt + 16 * h ..][0..16];
            for (dst, src[i..][0..16]) |*d, q| d.* = q ^ 0x80;
            i += 16;
        };
        const tab = std.mem.bytesAsSlice(u16, scales.bytes)[e * nt_n * groups * 16 ..][0 .. nt_n * groups * 16];
        for (0..nt_n) |nt| for (0..groups) |g| for (0..16) |r| {
            const bits = std.mem.littleToNative(u16, tab[(nt * groups + g) * 16 + r]);
            const sc: f16 = @bitCast(bits);
            // -128 s is exact in fp16: a power of two times a finite, small scale
            const bias: u16 = @bitCast(@as(f16, @floatCast(-128.0 * @as(f32, sc))));
            const at = (base + row0 + 16 * nt + r) * (s.k / mlx_group) + 2 * g;
            s.scales[at] = bits;
            s.scales[at + 1] = bits;
            s.biases[at] = bias;
            s.biases[at + 1] = bias;
        };
    }

    /// The last expert's rows from `row0`: the bf16 weight [n, k] rotated, rounded to 8-bit codes a 128-group.
    fn shared(s: Stack, a: Allocator, w: Tensor, row0: usize) !void {
        const base = (s.scales.len / ((s.k / mlx_group) * s.rows) - 1) * s.rows;
        const row = try a.alloc(f32, s.k);
        defer a.free(row);
        const src = std.mem.bytesAsSlice(u16, w.bytes);
        for (0..w.shape[0]) |n| {
            for (row, 0..) |*v, j| v.* = @bitCast(@as(u32, std.mem.littleToNative(u16, src[n * s.k + j])) << 16);
            rotate(row);
            const out_row = base + row0 + n;
            for (0..s.k / mlx_group) |g| {
                const v = row[g * mlx_group ..][0..mlx_group];
                var lo: f32 = v[0];
                var hi: f32 = v[0];
                for (v) |e| {
                    lo = @min(lo, e);
                    hi = @max(hi, e);
                }
                const b: f16 = @floatCast(lo);
                var sc: f16 = @floatCast((hi - @as(f32, b)) / 255.0);
                // the scale rounds up past the range so the top code stays at most 255
                while (@as(f32, sc) * 255.0 < hi - @as(f32, b)) sc = std.math.nextAfter(f16, sc, std.math.inf(f16));
                if (sc == 0) sc = 1;
                for (v, 0..) |e, j| {
                    const q = @round((e - @as(f32, b)) / @as(f32, sc));
                    s.words[out_row * s.k + g * mlx_group + j] = @intFromFloat(std.math.clamp(q, 0, 255));
                }
                s.scales[out_row * (s.k / mlx_group) + g] = @bitCast(sc);
                s.biases[out_row * (s.k / mlx_group) + g] = @bitCast(b);
            }
        }
    }
};

const Tensor3 = struct { words: Tensor, scales: Tensor, biases: Tensor };

/// The stacks as the MLX module's host side holds them (quant.Host's .mlx).
pub const Stacks = struct { fused: Tensor3, down: Tensor3, width: usize, dims: usize, count: usize };

fn codes(t: anytype, prefix: []const u8, name: []const u8, n: usize, k: usize, e: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const key = join(&buf, &.{ prefix, "experts.ultra_", name, "_codes" });
    const c = try t.get(key);
    // the loader folds the trailing [16, 16] of [E, N/16, K/32, 2, 16, 16] into one 256 (rank 5 at most)
    if (!c.is(.i8, &.{ e, n / 16, k / 32, 2, 256 })) return refuse(key, "not ciru int8 codes [E, N/16, K/32, 2, 16, 16]");
    return c;
}

fn scalesOf(t: anytype, prefix: []const u8, name: []const u8, n: usize, k: usize, e: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const key = join(&buf, &.{ prefix, "experts.ultra_", name, "_scales" });
    const s = try t.get(key);
    if (!s.is(.f16, &.{ e, n / 16, k / group, 16 })) return refuse(key, "not ciru fp16 scales [E, N/16, K/256, 16]");
    for (std.mem.bytesAsSlice(u16, s.bytes)) |b| {
        const v: f16 = @bitCast(std.mem.littleToNative(u16, b));
        if (!std.math.isFinite(v) or v <= 0) return refuse(key, "a scale is not finite and positive");
    }
    return s;
}

fn sharedOf(t: anytype, prefix: []const u8, name: []const u8, n: usize, k: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const key = join(&buf, &.{ prefix, "shared_expert.", name, ".weight" });
    const w = try t.get(key);
    if (!w.is(.bf16, &.{ n, k })) return refuse(key, "not a bf16 [N, K] shared expert");
    return w;
}

/// Whether `prefix` (a layer's `mlp.`) holds ciru experts.
pub fn holds(t: anytype, prefix: []const u8) bool {
    var buf: [256]u8 = undefined;
    return t.has(join(&buf, &.{ prefix, "experts.ultra_up_codes" }));
}

/// A layer's E + 1 experts as MLX 8-bit group-128 stacks, gate then up fused per expert, the shared one last.
pub fn experts(a: Allocator, t: anytype, prefix: []const u8) !Stacks {
    var buf: [256]u8 = undefined;
    const up_key = join(&buf, &.{ prefix, "experts.ultra_up_codes" });
    const probe = try t.get(up_key);
    if (probe.rank != 5) return refuse(up_key, "not ciru codes");
    const e = probe.shape[0];
    const width = probe.shape[1] * 16;
    const dims = probe.shape[2] * 32;
    if (width % group != 0 or dims % group != 0) return refuse(up_key, "widths are not whole 256-groups");
    const gate_c = try codes(t, prefix, "gate", width, dims, e);
    const up_c = try codes(t, prefix, "up", width, dims, e);
    const down_c = try codes(t, prefix, "down", dims, width, e);
    const gate_s = try scalesOf(t, prefix, "gate", width, dims, e);
    const up_s = try scalesOf(t, prefix, "up", width, dims, e);
    const down_s = try scalesOf(t, prefix, "down", dims, width, e);
    const fused = try Stack.init(a, e + 1, 2 * width, dims);
    const down = try Stack.init(a, e + 1, dims, width);
    for (0..e) |x| {
        fused.routed(gate_c, gate_s, x, 0);
        fused.routed(up_c, up_s, x, width);
        down.routed(down_c, down_s, x, 0);
    }
    try fused.shared(a, try sharedOf(t, prefix, "gate_proj", width, dims), 0);
    try fused.shared(a, try sharedOf(t, prefix, "up_proj", width, dims), width);
    try down.shared(a, try sharedOf(t, prefix, "down_proj", dims, width), 0);
    return .{ .fused = fused.host(e + 1), .down = down.host(e + 1), .width = width, .dims = dims, .count = e + 1 };
}

test "the rotation is its own inverse and matches the Sylvester matrix" {
    var x: [256]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
    const orig = x;
    rotate(&x);
    // H[i, j] = (-1)^popcount(i & j) / sqrt(128) for the first block's element 5
    var want: f64 = 0;
    for (0..128) |j| want += @as(f64, orig[j]) * (if (@popCount(@as(u8, @intCast(5 & j))) % 2 == 0) @as(f64, 1) else -1);
    try std.testing.expectApproxEqAbs(want / @sqrt(128.0), x[5], 1e-4);
    rotate(&x);
    for (x, orig) |v, o| try std.testing.expectApproxEqAbs(o, v, 1e-4);
}

test "codes unswizzle to rows, scales split into two 128-groups with the -128 s bias" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // one expert, N = 16, K = 256: code (r, k) = (r + k) % 127 - 63
    var raw: [16 * 256]i8 = undefined;
    for (0..8) |kt| for (0..2) |h| for (0..16) |r| for (0..16) |c| {
        const k = 32 * kt + 16 * h + c;
        raw[((kt * 2 + h) * 16 + r) * 16 + c] = @intCast(@as(i32, @intCast((r + k) % 127)) - 63);
    };
    var sc: [16]u16 = undefined;
    for (&sc, 0..) |*s, r| s.* = @bitCast(@as(f16, @floatFromInt(r + 1)) * 0.0078125);
    const c6: Tensor = .{ .dtype = .i8, .rank = 5, .shape = .{ 1, 1, 8, 2, 256 }, .bytes = std.mem.sliceAsBytes(&raw) };
    const st: Tensor = .{ .dtype = .f16, .rank = 4, .shape = .{ 1, 1, 1, 16, 1 }, .bytes = std.mem.sliceAsBytes(&sc) };
    const s = try Stack.init(a, 2, 16, 256);
    s.routed(c6, st, 0, 0);
    for (0..16) |r| for (0..256) |k| {
        const q: i32 = @as(i32, s.words[r * 256 + k]) - 128;
        try std.testing.expectEqual(@as(i32, @intCast((r + k) % 127)) - 63, q);
    };
    const s3: f16 = @bitCast(s.scales[3 * 2 + 1]);
    const b3: f16 = @bitCast(s.biases[3 * 2]);
    try std.testing.expectEqual(@as(f16, 4 * 0.0078125), s3);
    try std.testing.expectEqual(@as(f32, -128) * @as(f32, s3), @as(f32, b3));
}
