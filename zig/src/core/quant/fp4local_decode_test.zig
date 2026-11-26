//! The decode's exactness claims checked over every code and scale against the fp64 references, not assumed.

const std = @import("std");
const d = @import("fp4local_decode.zig");
const nvfp4 = @import("nvfp4.zig");
const fp8 = @import("fp8.zig");
const mxfp8_format = @import("mxfp8_format.zig");

fn same(want: f64, got: f64) bool {
    return @as(u64, @bitCast(want)) == @as(u64, @bitCast(got));
}

test "fp4local: the e2m1 fp16 trick is e2m1 x 2^-14 for all 16 codes, -0 and the subnormal 0.5 included" {
    for (0..16) |i| {
        const c: u4 = @intCast(i);
        const h: f16 = @bitCast(d.e2m1Half(c));
        try std.testing.expect(same(nvfp4.e2m1(c) * 0x1p-14, h));
    }
    try std.testing.expectEqual(@as(u16, 0x8000), d.e2m1Half(8));
    try std.testing.expectEqual(@as(u16, 0x0200), d.e2m1Half(1));
}

test "fp4local: the e4m3 fp16 trick is e4m3 x 2^-8 for all 254 finite bytes, zero, -0 and subnormals included" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) continue;
        const h: f16 = @bitCast(d.e4m3Half(b));
        try std.testing.expect(same(fp8.e4m3(b) * 0x1p-8, h));
    }
}

test "fp4local: NVFP4: all 16 codes x all 254 finite e4m3 scales decode to e2m1 x e4m3 exactly, -0 kept" {
    var finite: usize = 0;
    for (0..256) |j| {
        const s: u8 = @intCast(j);
        if (s & 0x7f == 0x7f) continue;
        finite += 1;
        const scale = d.nvfp4Scale(s);
        try std.testing.expect(same(fp8.e4m3(s), scale));
        const unscaled: f16 = @bitCast(d.e4m3Half(s));
        try std.testing.expectEqual(@as(f32, unscaled) * 256, @as(f32, scale));
        for (0..16) |i| {
            const c: u4 = @intCast(i);
            const w: f16 = @bitCast(d.e2m1Half(c));
            const p = d.nvfp4Product(c, s);
            // the fp16 product rounds nothing: it equals the f32 product of its operands
            try std.testing.expect(same(@as(f32, w) * @as(f32, scale), p));
            try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(p)) & 0xffff);
            const want = nvfp4.e2m1(c) * fp8.e4m3(s);
            try std.testing.expect(same(want * 0x1p-14, d.widen(d.nvfp4Weight(c, s))));
            try std.testing.expect(same(want, @as(f64, d.widen(d.nvfp4Weight(c, s))) * 0x1p14));
        }
    }
    try std.testing.expectEqual(@as(usize, 254), finite);
    try std.testing.expectEqual(@as(u16, 0x8000), d.nvfp4Weight(8, 0x38));
    try std.testing.expectEqual(@as(u16, 0x8000), d.nvfp4Weight(1, 0x80));
    try std.testing.expectEqual(@as(u16, 0x0000), d.nvfp4Weight(9, 0x80));
}

test "fp4local: NVFP4: the smallest product is fp16's smallest subnormal and the largest stays finite" {
    try std.testing.expectEqual(@as(f32, 0x1p-24), d.nvfp4Product(1, 0x01));
    try std.testing.expectEqual(@as(f32, 2688 * 0x1p-14), d.nvfp4Product(7, 0x7e));
    try std.testing.expectEqual(@as(f32, 0x1p14), d.nvfp4Global(1));
}

test "fp4local: MXFP8: all 256 bytes x E 0 to 254 decode exactly where admitted, and the rest are refused" {
    var admitted: usize = 0;
    for (0..255) |je| {
        const e: u8 = @intCast(je);
        for (0..256) |i| {
            const b: u8 = @intCast(i);
            var block: [32]u8 = @splat(0);
            block[5] = b;
            var why: @import("../modelopt.zig").Why = .{};
            const scale = [_]u8{e};
            const ok = if (mxfp8_format.admit("m.weight", &block, &scale, 1, 32, &why)) true else |err| blk: {
                try std.testing.expectEqual(error.UnsupportedQuantization, err);
                try std.testing.expect(std.mem.startsWith(u8, why.text(), "m.weight: "));
                break :blk false;
            };
            const nan = b & 0x7f == 0x7f;
            const zero = b & 0x7f == 0;
            const want_ok = !nan and e <= mxfp8_format.max_e and (zero or e >= mxfp8_format.min_e);
            try std.testing.expectEqual(want_ok, ok);
            if (!ok) continue;
            admitted += 1;
            const p = d.mxfp8Product(b, e);
            try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(p)) & 0xffff);
            const want = fp8.e4m3(b) * std.math.ldexp(@as(f64, 1), @as(i32, e) - 127);
            try std.testing.expect(same(want, d.widen(d.mxfp8Weight(b, e))));
            try std.testing.expect(zero or std.math.isNormal(p));
        }
    }
    // 237 in-range E x 254 finite bytes, and the 10 low E x 2 zeros
    try std.testing.expectEqual(@as(usize, 237 * 254 + 10 * 2), admitted);
}

test "fp4local: MXFP8: E 246 is the last finite multiplier; from 247 even a zero block decodes to NaN" {
    for (247..255) |je| {
        const e: u8 = @intCast(je);
        try std.testing.expect(std.math.isInf(d.mxfp8Multiplier(e)));
        try std.testing.expect(std.math.isNan(d.mxfp8Product(0, e)));
    }
    try std.testing.expect(same(448 * 0x1p119, d.widen(d.mxfp8Weight(0x7e, 246))));
    try std.testing.expect(same(0, d.widen(d.mxfp8Weight(0, 246))));
    var block: [32]u8 = @splat(0);
    try mxfp8_format.admit("m", &block, &.{246}, 1, 32, null);
    var why: @import("../modelopt.zig").Why = .{};
    try std.testing.expectError(error.UnsupportedQuantization, mxfp8_format.admit("m", &block, &.{247}, 1, 32, &why));
    try std.testing.expectEqualStrings("m: row 0 block 0: e8m0 247 above 246", why.text());
    block[0] = 0x7e;
    try mxfp8_format.admit("m", &block, &.{246}, 1, 32, null);
    try std.testing.expect(!std.math.isNormal(d.mxfp8Product(0x01, 9)));
    try std.testing.expect(std.math.isNormal(d.mxfp8Product(0x01, 10)));
}

test "fp4local: the prefill model refuses K that is not whole folds" {
    const x: [192]u16 = @splat(0);
    try std.testing.expectError(error.UnfoldedK, d.prefill(&x, &x, 1));
    try std.testing.expectError(error.UnfoldedK, d.prefill(x[0..128], x[0..64], 1));
}

test "fp4local: the fold's order shows: one accumulator over 256 inputs gives other bits than a fold at 128" {
    const x: [256]u16 = @splat(d.high(1));
    var w: [256]u16 = @splat(d.high(0x1p-24));
    w[0] = d.high(0x1p24);
    w[128] = d.high(-0x1p24);
    var flat: f32 = 0;
    for (0..256) |i| flat = flat + d.widen(x[i]) * d.widen(w[i]);
    try std.testing.expectEqual(@as(f32, 127 * 0x1p-24), flat);
    try std.testing.expectEqual(@as(f32, 0), try d.prefill(&x, &w, 1));
}
