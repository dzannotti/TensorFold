//! EXL3 projections on the GPU: the format's fp64 decoder as reference, and every row's bits alone equal to its bits
//! among 1 to 128 rows (lane rounds and prompt tiles alike).

const std = @import("std");
const base = @import("rig.zig");
const exl3 = @import("core").quant.exl3;
const launches = @import("../launches.zig");
const fmt = exl3.format;
const gpa = base.gpa;
const Buffer = base.Buffer;

const Case = struct { name: []const u8, k: usize, n: usize, halves: u16, codebook: fmt.Codebook };

const cases = [_]Case{
    .{ .name = "3-bit mul1", .k = 1024, .n = 512, .halves = 6, .codebook = .mul1 },
    .{ .name = "4-bit mul1, K split", .k = 2048, .n = 256, .halves = 8, .codebook = .mul1 },
    .{ .name = "6-bit mcg, one split", .k = 512, .n = 4096, .halves = 12, .codebook = .mcg },
    .{ .name = "3.5-bit mul1", .k = 1024, .n = 384, .halves = 7, .codebook = .mul1 },
    .{ .name = "2-bit 3inst", .k = 512, .n = 512, .halves = 4, .codebook = .inst3 },
    .{ .name = "8-bit mcg", .k = 512, .n = 256, .halves = 16, .codebook = .mcg },
};

const most_rows = 128;
const together = [_]usize{ 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 128 };

/// One case: the 128 rows against the fp64 reference, then each row count's rows against the same rows alone.
fn run(t: *base.Rig, rng: *base.Rng, c: Case) !void {
    const words = (c.k / 16) * (c.n / 16) * @as(usize, c.halves) * 8;
    const trellis = try gpa.alloc(i16, words);
    defer gpa.free(trellis);
    for (trellis) |*w| w.* = @bitCast(@as(u16, @truncate(rng.next())));
    const suh = try gpa.alloc(f16, c.k);
    defer gpa.free(suh);
    for (suh) |*s| s.* = @floatCast((1 + 0.1 * rng.unit()) * @as(f32, if (rng.next() & 1 == 0) 1 else -1));
    const svh = try gpa.alloc(f16, c.n);
    defer gpa.free(svh);
    for (svh) |*s| s.* = @floatCast(0.02 * (1 + 0.1 * rng.unit()));
    const x = try gpa.alloc(u16, most_rows * c.k);
    defer gpa.free(x);
    for (x) |*v| v.* = base.bf16Bits(rng.unit());

    var dt = try Buffer.fromHost(t, trellis);
    defer dt.free();
    var ds = try Buffer.fromHost(t, suh);
    defer ds.free();
    var dv = try Buffer.fromHost(t, svh);
    defer dv.free();
    var dx = try Buffer.fromHost(t, x);
    defer dx.free();
    var out = try Buffer.alloc(t, most_rows * c.n * 4);
    defer out.free();
    var scratch = try Buffer.alloc(t, launches.Launcher.exl3Scratch(most_rows, @intCast(c.k), @intCast(c.n)));
    defer scratch.free();
    const m: exl3.Matrix = .{ .trellis = dt.ptr, .suh = ds.ptr, .svh = dv.ptr, .bias = 0, .halves = @intCast(c.halves), .codebook = c.codebook };

    const all = try gpa.alloc(f32, most_rows * c.n);
    defer gpa.free(all);
    const got = try gpa.alloc(f32, most_rows * c.n);
    defer gpa.free(got);
    try t.on.exl3Project(dx.ptr, 1, m, @intCast(c.n), @intCast(c.k), most_rows, out.ptr, 0, scratch.ptr, t.stream.handle);
    try t.stream.synchronize();
    try out.download(0, std.mem.sliceAsBytes(all));

    // the reference: W [K, N] in fp64, against the first rows
    const w = try gpa.alloc(f64, 3 * c.k * c.n);
    defer gpa.free(w);
    fmt.dequantize(.{ .halves = c.halves }, c.codebook, trellis, c.k / 16, c.n / 16, suh, svh, w[0 .. c.k * c.n], w[c.k * c.n ..]);
    var err: f64 = 0;
    var mag: f64 = 0;
    for (0..4) |r| for (0..c.n) |j| {
        var want: f64 = 0;
        for (0..c.k) |i| want += base.bf16Value(x[r * c.k + i]) * w[i * c.n + j];
        err = @max(err, @abs(want - all[r * c.n + j]));
        mag = @max(mag, @abs(want));
    };
    // the input rotation rounds to fp16 (as ExLlamaV3's): about 2^-11 of each term
    try base.expect(err <= 2e-3 * mag, "{s}: max error {e} of {e}", .{ c.name, err, mag });

    for (together) |rows| {
        const first = (rows * 5) % (most_rows - rows + 1);
        try t.on.exl3Project(dx.ptr + first * c.k * 2, 1, m, @intCast(c.n), @intCast(c.k), rows, out.ptr, 0, scratch.ptr, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. rows * c.n]));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got[0 .. rows * c.n]), std.mem.sliceAsBytes(all[first * c.n ..][0 .. rows * c.n]));
        try base.expect(same, "{s}: {d} rows from row {d} differ from the same rows among 128", .{ c.name, rows, first });
    }
}

test "EXL3 products match the format's fp64 decoder and keep every row's bits at any row count" {
    const t = try base.Rig.open(1 << 20);
    defer t.close();
    var rng: base.Rng = .{ .state = 0x45584c33 };
    for (cases) |c| try run(t, &rng, c);
}

test "EXL3 packs' dense rows over fp16 weights (in_proj_a and b) match fp64 and keep each row's bits" {
    const t = try base.Rig.open(1 << 20);
    defer t.close();
    var rng: base.Rng = .{ .state = 0x64656e73 };
    const k = 5120;
    const n = 48;
    const rows = 16;
    const w = try gpa.alloc(f16, n * k);
    defer gpa.free(w);
    for (w) |*v| v.* = @floatCast(0.05 * rng.unit());
    const x = try gpa.alloc(u16, rows * k);
    defer gpa.free(x);
    for (x) |*v| v.* = base.bf16Bits(rng.unit());
    var dw = try Buffer.fromHost(t, w);
    defer dw.free();
    var dx = try Buffer.fromHost(t, x);
    defer dx.free();
    var out = try Buffer.alloc(t, rows * n * 2);
    defer out.free();
    const all = try gpa.alloc(u16, rows * n);
    defer gpa.free(all);
    const one = try gpa.alloc(u16, n);
    defer gpa.free(one);
    // act.hpp kinds: bf16 x (2), fp16 weights (1)
    try t.on.tf_dense_rows(@ptrFromInt(dx.ptr), 2, @ptrFromInt(dw.ptr), 1, @ptrFromInt(out.ptr), rows, n, k, t.stream.handle);
    try t.stream.synchronize();
    try out.download(0, std.mem.sliceAsBytes(all));
    for (0..rows) |r| for (0..n) |c| {
        var want: f64 = 0;
        for (0..k) |i| want += base.bf16Value(x[r * k + i]) * @as(f64, w[c * k + i]);
        const got = base.bf16Value(all[r * n + c]);
        try base.expect(@abs(got - want) <= 1e-2 * (1 + @abs(want)), "dense rows ({d}, {d}): {d} vs {d}", .{ r, c, got, want });
    };
    for (0..rows) |r| {
        try t.on.tf_dense_rows(@ptrFromInt(dx.ptr + r * k * 2), 2, @ptrFromInt(dw.ptr), 1, @ptrFromInt(out.ptr), 1, n, k, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(one));
        try base.expect(std.mem.eql(u16, one, all[r * n ..][0..n]), "dense rows: row {d} alone differs", .{r});
    }
}
