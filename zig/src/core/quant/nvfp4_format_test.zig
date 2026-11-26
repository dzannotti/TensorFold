//! The baked NVFP4 layout against the stored tensors: permute, CSF scales, round trips, stacks, refusals and bytes.

const std = @import("std");
const f = @import("nvfp4_format.zig");
const decode = @import("fp4local_decode.zig");
const nvfp4 = @import("nvfp4.zig");
const nvfp4_slice = @import("../nvfp4_slice.zig");
const Why = @import("../modelopt.zig").Why;

const tensor = @import("types.zig").view;

/// Random stored planes of one projection [n, k]: codes, finite scales, and a CSF form of the same scales.
const Planes = struct {
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,
    fixed: []u8,
    exceptions: []u8,
    global: [4]u8,

    fn init(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize, global: f32) !Planes {
        const cols = k / 16;
        var p: Planes = .{
            .n = n,
            .k = k,
            .codes = try a.alloc(u8, n * k / 2),
            .scales = try a.alloc(u8, n * cols),
            .fixed = try a.alloc(u8, n / 16 * (16 + 8 * cols)),
            .exceptions = undefined,
            .global = @bitCast(global),
        };
        rng.bytes(p.codes);
        var list: std.ArrayList(u8) = .empty;
        defer list.deinit(a);
        for (0..n / 16) |slab| {
            const at = slab * (16 + 8 * cols);
            for (0..16) |r| {
                const row = slab * 16 + r;
                const base = rng.uintAtMost(u8, 0x6e);
                p.fixed[at + r] = base;
                for (0..cols) |c| {
                    const off = rng.uintLessThan(u8, 16);
                    const byte = &p.fixed[at + 16 + r * (cols / 2) + c / 2];
                    if (c % 2 == 0) byte.* = off else byte.* |= off << 4;
                    p.scales[row * cols + c] = base + off;
                    if (rng.uintLessThan(u8, 5) == 0) {
                        var v = rng.int(u8);
                        if (v & 0x7f == 0x7f) v ^= 1;
                        p.scales[row * cols + c] = v;
                        var word: [4]u8 = undefined;
                        std.mem.writeInt(u32, &word, @as(u32, v) << 24 | @as(u32, @intCast(row * cols + c)), .little);
                        try list.appendSlice(a, &word);
                    }
                }
            }
        }
        p.exceptions = try list.toOwnedSlice(a);
        return p;
    }

    fn deinit(p: *Planes, a: std.mem.Allocator) void {
        for ([_][]u8{ p.codes, p.scales, p.fixed, p.exceptions }) |b| a.free(b);
    }

    fn source(p: *const Planes, csf: bool) nvfp4_slice.Source {
        return .{
            .weight = tensor(.u8, &.{ p.n, p.k / 2 }, p.codes),
            .scales = if (csf) .{ .csf = .{
                .fixed = tensor(.u8, &.{ p.n / 16, 16 + 8 * (p.k / 16) }, p.fixed),
                .exceptions = tensor(.u32, &.{p.exceptions.len / 4}, p.exceptions),
            } } else .{ .plain = tensor(.f8_e4m3, &.{ p.n, p.k / 16 }, p.scales) },
            .global = tensor(.f32, &.{}, &p.global),
        };
    }
};

test "fp4local: the nibble permute puts pair p at bits 4p and 16 + 4p, and round-trips every nibble at every place" {
    for (0..8) |i| for (0..16) |v| {
        const w = @as(u32, @intCast(v)) << @intCast(4 * i);
        const b = f.permute(w);
        try std.testing.expectEqual(w, f.unpermute(b));
        const p: u2 = @intCast(i / 2);
        try std.testing.expectEqual(@as(u4, @intCast(v)), f.pair(b, p)[i % 2]);
        try std.testing.expectEqual(b, (b >> (@as(u5, p) * 4) & 0x000F000F) << (@as(u5, p) * 4));
    };
    try std.testing.expectEqual(@as(u32, 0x7531_6420), f.permute(0x7654_3210));
    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (0..100_000) |_| {
        const w = prng.random().int(u32);
        try std.testing.expectEqual(w, f.unpermute(f.permute(w)));
        for (0..4) |p| {
            const pq = f.pair(f.permute(w), @intCast(p));
            try std.testing.expectEqual(@as(u4, @truncate(w >> @intCast(8 * p))), pq[0]);
            try std.testing.expectEqual(@as(u4, @truncate(w >> @intCast(8 * p + 4))), pq[1]);
        }
    }
}

test "fp4local: a baked projection round-trips its codes and scales and decodes each weight as fp64 does" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for ([_][2]usize{ .{ 16, 128 }, .{ 32, 256 }, .{ 48, 640 }, .{ 16, 2560 } }) |shape| {
        var planes = try Planes.init(a, rng, shape[0], shape[1], 0.375);
        defer planes.deinit(a);
        for ([_]bool{ false, true }) |csf| {
            var p = try f.bake(a, planes.source(csf), "m.experts.0.gate_proj", null);
            defer p.deinit(a);
            try std.testing.expectEqual(f.bytes(shape[0], shape[1]), p.bytes());
            const back = try f.codes(a, p);
            defer a.free(back);
            try std.testing.expectEqualSlices(u8, planes.codes, back);
            try std.testing.expectEqualSlices(u8, planes.scales, p.scales);
            const ref = try a.alloc(f64, shape[0] * shape[1]);
            defer a.free(ref);
            try nvfp4.reference(.{
                .weight = tensor(.u8, &.{ shape[0], shape[1] / 2 }, planes.codes),
                .scale = tensor(.f8_e4m3, &.{ shape[0], shape[1] / 16 }, planes.scales),
                .global = &.{1},
            }, ref);
            for (0..shape[0]) |r| for (0..shape[1]) |c| {
                const got: f64 = decode.widen(p.weight(r, c));
                try std.testing.expectEqual(@as(u64, @bitCast(ref[r * shape[1] + c])), @as(u64, @bitCast(got * 0x1p14)));
            };
            try std.testing.expectEqual(@as(f32, 0.375), p.global);
        }
    }
}

test "fp4local: CSF scales decode to the plain bytes nvfp4_slice gives, exceptions included" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    for (0..20) |_| {
        var planes = try Planes.init(a, prng.random(), 32, 128, 1);
        defer planes.deinit(a);
        try std.testing.expect(planes.exceptions.len > 0);
        var p = try f.bake(a, planes.source(true), "m", null);
        defer p.deinit(a);
        var s = try nvfp4_slice.slice(a, planes.source(true), .{ .from = 0, .to = 32 }, .{ .from = 0, .to = 128 });
        defer s.deinit(a);
        try std.testing.expectEqualSlices(u8, s.scales, p.scales);
        try std.testing.expectEqualSlices(u8, planes.scales, p.scales);
    }
}

test "fp4local: experts stack gate | up rows an expert, down alone, with their globals" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    var members: [6]f.Projection = undefined;
    var planes: [6]Planes = undefined;
    for (&members, &planes, 0..) |*m, *pl, i| {
        pl.* = try Planes.init(a, rng, 16, 128, @floatFromInt(i + 1));
        m.* = try f.bake(a, pl.source(i % 2 == 1), "m", null);
    }
    defer for (&members, &planes) |*m, *pl| {
        m.deinit(a);
        pl.deinit(a);
    };
    var fused = try f.stack(a, &members, 2);
    defer fused.deinit(a);
    try std.testing.expectEqual(@as(usize, 3), fused.count);
    try std.testing.expectEqual(@as(usize, 32), fused.rows);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, fused.globals);
    for (0..3) |e| for (0..2) |h| {
        const m = members[2 * e + h];
        const row0 = e * 32 + h * 16;
        try std.testing.expectEqualSlices(u32, m.words, fused.words[row0 * 16 ..][0 .. 16 * 16]);
        try std.testing.expectEqualSlices(u8, m.scales, fused.scales[row0 * 8 ..][0 .. 16 * 8]);
    };
    var down = try f.stack(a, members[0..3], 1);
    defer down.deinit(a);
    try std.testing.expectEqual(@as(usize, 16), down.rows);
    try std.testing.expectEqualSlices(u32, members[2].words, down.words[2 * 16 * 16 ..][0 .. 16 * 16]);
    try std.testing.expectError(error.UnexpectedTensor, f.stack(a, members[0..3], 2));
}

test "fp4local: baking refuses, by name, a NaN scale, a bad global, K not whole chunks and malformed tensors" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    var planes = try Planes.init(a, prng.random(), 16, 128, 1);
    defer planes.deinit(a);
    var why: Why = .{};
    planes.scales[13] = 0x7f;
    try std.testing.expectError(error.UnsupportedQuantization, f.bake(a, planes.source(false), "l.up_proj", &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.up_proj: row 1 group 5: NaN"));
    planes.scales[13] = 0x38;
    // row 2's CSF base 0x70 plus every offset 15 decodes to 0x7f: refused once decoded, as a plain NaN scale is
    var saved: [5]u8 = undefined;
    @memcpy(saved[0..1], planes.fixed[2..3]);
    @memcpy(saved[1..5], planes.fixed[24..28]);
    planes.fixed[2] = 0x70;
    @memset(planes.fixed[24..28], 0xff);
    try std.testing.expectError(error.UnsupportedQuantization, f.bake(a, planes.source(true), "l.up_proj", &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.up_proj: row 2 group "));
    try std.testing.expect(std.mem.endsWith(u8, why.text(), ": NaN e4m3 scale"));
    @memcpy(planes.fixed[2..3], saved[0..1]);
    @memcpy(planes.fixed[24..28], saved[1..5]);
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32), 0x1p120 }) |g| {
        planes.global = @bitCast(g);
        try std.testing.expectError(error.UnsupportedQuantization, f.bake(a, planes.source(false), "l.up_proj", &why));
        try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.up_proj: weight_scale_2"));
    }
    planes.global = @bitCast(@as(f32, 1));
    var narrow = planes.source(false);
    narrow.weight = tensor(.u8, &.{ 32, 8 }, planes.codes[0..256]);
    narrow.scales.plain = tensor(.f8_e4m3, &.{ 32, 1 }, planes.scales[0..32]);
    try std.testing.expectError(error.UnexpectedTensor, f.bake(a, narrow, "l.down_proj", &why));
    try std.testing.expectEqualStrings("l.down_proj: K 16 is not whole folds of 128", why.text());
    var bad = planes.source(false);
    bad.scales.plain.dtype = .u8;
    try std.testing.expectError(error.BadNvfp4, f.bake(a, bad, "l.gate_proj", &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "l.gate_proj: not ModelOpt NVFP4"));
}
