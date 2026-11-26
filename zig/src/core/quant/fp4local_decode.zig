//! gfx1151 FP4-Local decode and prefill sum, bit for bit; needs fp16 denormals kept (v_pk_mul_f16, v_cvt_f32_f16).

const std = @import("std");

/// Inputs a prefill group folds into the accumulator (scale 1, no bias): fixed, as another value changes every bit.
pub const fold = 128;

/// The fp16 bits of e2m1 code `c` times 2^-14: its sign to bit 15, exponent and mantissa to bits 9 to 11.
pub fn e2m1Half(c: u4) u16 {
    return @as(u16, c & 7) << 9 | @as(u16, c & 8) << 12;
}

/// The fp16 bits of e4m3 byte `b` times 2^-8: its sign to bit 15, exponent and mantissa to bits 7 to 13.
pub fn e4m3Half(b: u8) u16 {
    return @as(u16, b & 0x7f) << 7 | @as(u16, b & 0x80) << 8;
}

/// An NVFP4 block scale in fp16: e4m3 times 2^-8, then times 256 in fp16 (v_pk_mul_f16).
pub fn nvfp4Scale(b: u8) f16 {
    const h: f16 = @bitCast(e4m3Half(b));
    return h * @as(f16, 256);
}

/// Weight times scale in fp16 (v_pk_mul_f16), widened to f32 (v_cvt_f32_f16): e2m1 x e4m3 x 2^-14.
pub fn nvfp4Product(c: u4, scale: u8) f32 {
    const w: f16 = @bitCast(e2m1Half(c));
    return w * nvfp4Scale(scale);
}

/// The baked NVFP4 weight: `nvfp4Product`'s high half (v_perm), the bf16 the kernels dot.
pub fn nvfp4Weight(c: u4, scale: u8) u16 {
    return high(nvfp4Product(c, scale));
}

/// The f32 epilogue's factor for a projection's weight_scale_2: global x 2^14, undoing the fp16 weight's 2^-14.
pub fn nvfp4Global(global: f32) f32 {
    return global * 0x1p14;
}

/// The f32 multiplier of an MXFP8 block's e8m0 byte `e`: 2^(e - 119), the e4m3 fp16 trick's 2^-8 undone.
pub fn mxfp8Multiplier(e: u8) f32 {
    return std.math.ldexp(@as(f32, 1), @as(i32, e) - 119);
}

/// e4m3 byte `b` in fp16 (`e4m3Half`), widened to f32 (v_cvt_f32_f16), times its block's multiplier (v_mul_f32).
pub fn mxfp8Product(b: u8, e: u8) f32 {
    const h: f16 = @bitCast(e4m3Half(b));
    return @as(f32, h) * mxfp8Multiplier(e);
}

/// The baked MXFP8 weight: `mxfp8Product`'s high half, the bf16 the kernels dot.
pub fn mxfp8Weight(b: u8, e: u8) u16 {
    return high(mxfp8Product(b, e));
}

/// The high half of an f32: its bf16 by truncation, exact when the low half is zero.
pub fn high(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

pub fn widen(bf: u16) f32 {
    return @bitCast(@as(u32, bf) << 16);
}

/// One v_dot2_f32_bf16, modelled as the ISA's pseudocode orders it: acc plus the low product, then the high, in f32.
pub fn dot2(acc: f32, x: [2]u16, w: [2]u16) f32 {
    const lo = acc + widen(x[0]) * widen(w[0]);
    return lo + widen(x[1]) * widen(w[1]);
}

/// A prefill output as tiles/gemm.hpp sums it: a dot2 chain per `fold` inputs, each added to acc, then the global once.
pub fn prefill(x: []const u16, w: []const u16, global: f32) error{UnfoldedK}!f32 {
    if (x.len != w.len or x.len % fold != 0) return error.UnfoldedK;
    var acc: f32 = 0;
    var g: usize = 0;
    while (g < x.len) : (g += fold) {
        var d: f32 = 0;
        var t = g;
        while (t < g + fold) : (t += 2) d = dot2(d, .{ x[t], x[t + 1] }, .{ w[t], w[t + 1] });
        acc = acc + d;
    }
    return acc * global;
}

test {
    _ = @import("fp4local_decode_test.zig");
}
