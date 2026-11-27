//! MXFP8 checks and admission: every refusal named, admitted bytes decoded exactly against fp64.

const std = @import("std");
const m = @import("mxfp8_format.zig");
const decode = @import("fp4local_decode.zig");
const tensor = @import("types.zig").view;
const Why = @import("../modelopt.zig").Why;

test "fp4local: every admitted weight of a random projection decodes as its fp64 reference" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const n = 16;
    const k = 256;
    var codes: [n * k]u8 = undefined;
    var scales: [n * k / 32]u8 = undefined;
    rng.bytes(&codes);
    for (&codes) |*b| if (b.* & 0x7f == 0x7f) {
        b.* ^= 1;
    };
    for (&scales) |*e| e.* = m.min_e + rng.uintAtMost(u8, m.max_e - m.min_e);
    try m.check("a.q_proj", tensor(.f8_e4m3, &.{ n, k }, &codes), tensor(.u8, &.{ n, k / 32 }, &scales), null);
    const ref = try a.alloc(f64, n * k);
    defer a.free(ref);
    m.reference(.{ .n = n, .k = k, .codes = &codes, .scales = &scales }, ref);
    for (ref, codes, 0..) |want, b, i| {
        const e = scales[i / k * (k / 32) + i % k / 32];
        try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(@as(f64, decode.widen(decode.mxfp8Weight(b, e))))));
    }
}

test "fp4local: the check refuses each bad tensor by its own name" {
    var codes: [2 * 128]u8 = @splat(0x38);
    var scales: [8]u8 = @splat(127);
    const w = tensor(.f8_e4m3, &.{ 2, 128 }, &codes);
    const s = tensor(.u8, &.{ 2, 4 }, &scales);
    var why: Why = .{};
    codes[134] = 0xff;
    try std.testing.expectError(error.UnsupportedQuantization, m.check("l.in_proj_qkv", w, s, &why));
    try std.testing.expectEqualStrings("l.in_proj_qkv: row 1 column 6: e4m3 NaN byte 0xff", why.text());
    codes[134] = 0x38;
    scales[5] = 255;
    try std.testing.expectError(error.UnsupportedQuantization, m.check("l.in_proj_z", w, s, &why));
    try std.testing.expectEqualStrings("l.in_proj_z: row 1 block 1: e8m0 255 above 246", why.text());
    scales[5] = 9;
    try std.testing.expectError(error.UnsupportedQuantization, m.check("l.in_proj_z", w, s, &why));
    try std.testing.expectEqualStrings("l.in_proj_z: row 1 block 1: e8m0 9 below 10 holds a nonzero code", why.text());
    @memset(codes[160..192], 0x80);
    try m.check("l.in_proj_z", w, s, &why);
    const short = tensor(.f8_e4m3, &.{ 4, 64 }, &codes);
    try std.testing.expectError(error.UnexpectedTensor, m.check("l.in_proj_b", short, tensor(.u8, &.{ 4, 2 }, &scales), &why));
    try std.testing.expectEqualStrings("l.in_proj_b.weight: K 64 is not whole folds of 128", why.text());
    try std.testing.expectError(error.UnexpectedTensor, m.check("l.in_proj_a", w, tensor(.f8_e8m0, &.{ 2, 4 }, &scales), &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.in_proj_a.weight_scale: "));
    try std.testing.expectError(error.UnexpectedTensor, m.check("l.out_proj", tensor(.bf16, &.{ 1, 128 }, &codes), s, &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.out_proj.weight: "));
}
