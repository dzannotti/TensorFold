//! MXFP8 as gfx1151 FP4-Local kernels read it: e4m3 [n][k] row-stitched, e8m0 [n][k / 32], admitted only when exact.

const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");
const fold = @import("fp4local_decode.zig").fold;
const modelopt = @import("../modelopt.zig");

const Tensor = types.Tensor;
const Why = modelopt.Why;
const refuse = modelopt.refuse;

/// Inputs one e8m0 byte scales.
pub const block = 32;
/// The e8m0 range whose every e4m3 product is a normal bf16 (or zero); below it only zero codes are admitted.
pub const min_e: u8 = 10;
/// Above this, 2^(E - 119) is infinite in f32, so no block is admitted, not even a zero one (0 x inf is NaN).
pub const max_e: u8 = 246;

/// A projection's planes as the kernels read them, its parts' rows in turn (the family stitches them); borrowed.
pub const Projection = struct { n: usize, k: usize, codes: []const u8, scales: []const u8 };

/// A projection's device bytes: n x k codes and n x k / 32 scales.
pub fn bytes(n: usize, k: usize) usize {
    return n * k + n * (k / block);
}

/// `{name}.weight` e4m3 [n, k] and `{name}.weight_scale` u8 [n, k / 32], K whole prefill folds, admitted by `name`.
pub fn check(name: []const u8, weight: Tensor, scale: Tensor, why: ?*Why) error{ UnexpectedTensor, UnsupportedQuantization }!void {
    const n = weight.dim(0);
    const k = weight.dim(1);
    if (weight.dtype != .f8_e4m3 or weight.rank != 2 or n == 0 or k == 0 or weight.bytes.len != n * k)
        return refuse(why, error.UnexpectedTensor, "{s}.weight: {t} {any} is not e4m3 [n, k]", .{ name, weight.dtype, weight.shape[0..weight.rank] });
    if (k % fold != 0) return refuse(why, error.UnexpectedTensor, "{s}.weight: K {d} is not whole folds of {d}", .{ name, k, fold });
    if (!scale.is(.u8, &.{ n, k / block }) or scale.bytes.len != n * (k / block))
        return refuse(why, error.UnexpectedTensor, "{s}.weight_scale: {t} {any} is not u8 [{d}, {d}]", .{ name, scale.dtype, scale.shape[0..scale.rank], n, k / block });
    try admit(name, weight.bytes, scale.bytes, n, k, why);
}

/// Refuse, under `name`, a NaN weight byte, an E above `max_e` and a nonzero code under an E below `min_e`.
pub fn admit(name: []const u8, codes: []const u8, scales: []const u8, n: usize, k: usize, why: ?*Why) error{UnsupportedQuantization}!void {
    std.debug.assert(k % block == 0 and codes.len == n * k and scales.len == n * (k / block));
    const fail = error.UnsupportedQuantization;
    for (scales, 0..) |e, at| {
        const row = at / (k / block);
        const col = at % (k / block) * block;
        if (e > max_e) return refuse(why, fail, "{s}: row {d} block {d}: e8m0 {d} above {d}", .{ name, row, col / block, e, max_e });
        for (codes[row * k + col ..][0..block], 0..) |b, i| {
            if (b & 0x7f == 0x7f) return refuse(why, fail, "{s}: row {d} column {d}: e4m3 NaN byte 0x{x:0>2}", .{ name, row, col + i, b });
            if (b & 0x7f != 0 and e < min_e)
                return refuse(why, fail, "{s}: row {d} block {d}: e8m0 {d} below {d} holds a nonzero code", .{ name, row, col / block, e, min_e });
        }
    }
}

/// fp64 dequantization of every weight, e4m3 x 2^(E - 127), into `out` [n][k].
pub fn reference(p: Projection, out: []f64) void {
    std.debug.assert(out.len == p.n * p.k);
    for (out, p.codes, 0..) |*o, b, i| {
        const e = p.scales[i / p.k * (p.k / block) + i % p.k / block];
        o.* = fp8.e4m3(b) * std.math.ldexp(@as(f64, 1), @as(i32, e) - 127);
    }
}

test {
    _ = @import("mxfp8_format_test.zig");
}
