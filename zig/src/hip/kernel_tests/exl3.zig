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
    .{ .name = "6-bit mul1, two column blocks a wave", .k = 256, .n = 81920, .halves = 12, .codebook = .mul1 },
    .{ .name = "6-bit mul1", .k = 1024, .n = 1024, .halves = 12, .codebook = .mul1 },
    .{ .name = "5-bit mul1", .k = 1024, .n = 640, .halves = 10, .codebook = .mul1 },
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
    const m: exl3.Matrix = .{ .trellis = dt.ptr, .suh = ds.ptr, .svh = dv.ptr, .bias = 0, .halves = @intCast(c.halves), .codebook = c.codebook, .columns = @intCast(c.n) };

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

    // the first columns of the stored matrix alone (a draft head's narrowed vocabulary)
    if (c.n % 256 == 0) {
        const half: u32 = @intCast(c.n / 2);
        try t.on.exl3Project(dx.ptr, 1, m, half, @intCast(c.k), 4, out.ptr, 0, scratch.ptr, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. 4 * half]));
        var narrow_err: f64 = 0;
        for (0..4) |r| for (0..half) |j| {
            var want: f64 = 0;
            for (0..c.k) |i| want += base.bf16Value(x[r * c.k + i]) * w[i * c.n + j];
            narrow_err = @max(narrow_err, @abs(want - got[r * half + j]));
        };
        try base.expect(narrow_err <= 2e-3 * mag, "{s}: first {d} columns, max error {e} of {e}", .{ c.name, half, narrow_err, mag });
    }

    for (together) |rows| {
        const first = (rows * 5) % (most_rows - rows + 1);
        try t.on.exl3Project(dx.ptr + first * c.k * 2, 1, m, @intCast(c.n), @intCast(c.k), rows, out.ptr, 0, scratch.ptr, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. rows * c.n]));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got[0 .. rows * c.n]), std.mem.sliceAsBytes(all[first * c.n ..][0 .. rows * c.n]));
        try base.expect(same, "{s}: {d} rows from row {d} differ from the same rows among 128", .{ c.name, rows, first });
    }

    // the prompt product: 300 rows (past one 256-row block, on the tile) against fp64, then each row count's rows against
    // the same rows among the 300, up to 128 rows on exl3_chain (and the first 128 against the decode product's to its
    // rounding)
    const pm = 300;
    const px = try gpa.alloc(u16, pm * c.k);
    defer gpa.free(px);
    for (px, 0..) |*v, i| v.* = if (i < x.len) x[i] else base.bf16Bits(rng.unit());
    var dpx = try Buffer.fromHost(t, px);
    defer dpx.free();
    var pout = try Buffer.alloc(t, pm * c.n * 4);
    defer pout.free();
    var pscratch = try Buffer.alloc(t, launches.Launcher.exl3Scratch(pm, @intCast(c.k), @intCast(c.n)));
    defer pscratch.free();
    if (!try t.on.exl3Prompt(dpx.ptr, 1, m, @intCast(c.n), @intCast(c.k), pm, pout.ptr, 0, pscratch.ptr, t.stream.handle)) return;
    try t.stream.synchronize();
    const pall = try gpa.alloc(f32, pm * c.n);
    defer gpa.free(pall);
    const pgot = try gpa.alloc(f32, pm * c.n);
    defer gpa.free(pgot);
    try pout.download(0, std.mem.sliceAsBytes(pall));
    var perr: f64 = 0;
    var pmag: f64 = 0;
    for ([_]usize{ 0, 1, 2, 3, 130, 257, 299 }) |r| for (0..c.n) |j| {
        var want: f64 = 0;
        for (0..c.k) |i| want += base.bf16Value(px[r * c.k + i]) * w[i * c.n + j];
        perr = @max(perr, @abs(want - pall[r * c.n + j]));
        pmag = @max(pmag, @abs(want));
    };
    try base.expect(perr <= 2e-3 * pmag, "{s} prompt: max error {e} of {e}", .{ c.name, perr, pmag });
    var drift: f64 = 0;
    for (0..most_rows * c.n) |i| drift = @max(drift, @abs(@as(f64, pall[i]) - all[i]));
    try base.expect(drift <= 1e-4 * pmag, "{s} prompt: {e} from the decode product's sums", .{ c.name, drift });
    for (together ++ [_]usize{ 200, 256, 257, 300 }) |rows| {
        const first = (rows * 5) % (pm - rows + 1);
        try base.expect(try t.on.exl3Prompt(dpx.ptr + first * c.k * 2, 1, m, @intCast(c.n), @intCast(c.k), rows, pout.ptr, 0, pscratch.ptr, t.stream.handle), "{s}: prompt refused", .{c.name});
        try t.stream.synchronize();
        try pout.download(0, std.mem.sliceAsBytes(pgot[0 .. rows * c.n]));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(pgot[0 .. rows * c.n]), std.mem.sliceAsBytes(pall[first * c.n ..][0 .. rows * c.n]));
        try base.expect(same, "{s} prompt: {d} rows from row {d} differ from the same rows among {d}", .{ c.name, rows, first, pm });
    }
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// The prompt tile against the decode product on the 27B's prompt shapes (TF_EXL3_BENCH=1; TF_EXL3_BENCH_M rows,
// default a prefill span's 2048): microseconds and TFLOPS, bf16 out.
test "EXL3 prompt timings on the 27B shapes (TF_EXL3_BENCH=1)" {
    _ = std.c.getenv("TF_EXL3_BENCH") orelse return;
    const t = try base.Rig.open(1 << 20);
    defer t.close();
    const rows: usize = if (std.c.getenv("TF_EXL3_BENCH_M")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
    const Shape = struct { name: []const u8, k: usize, n: usize, halves: u16 };
    const shapes = [_]Shape{
        .{ .name = "qkv 4b", .k = 5120, .n = 10240, .halves = 8 },
        .{ .name = "z 4b", .k = 5120, .n = 6144, .halves = 8 },
        .{ .name = "out 4b", .k = 6144, .n = 5120, .halves = 8 },
        .{ .name = "gate 3b", .k = 5120, .n = 17408, .halves = 6 },
        .{ .name = "gate 4b", .k = 5120, .n = 17408, .halves = 8 },
        .{ .name = "down 3b", .k = 17408, .n = 5120, .halves = 6 },
        .{ .name = "down 4b", .k = 17408, .n = 5120, .halves = 8 },
        .{ .name = "q 4b", .k = 5120, .n = 12288, .halves = 8 },
        .{ .name = "k/v 4b", .k = 5120, .n = 1024, .halves = 8 },
        .{ .name = "o 5b", .k = 6144, .n = 5120, .halves = 10 },
        .{ .name = "head 6b", .k = 5120, .n = 248320, .halves = 12 },
    };
    std.debug.print("\nshape        M   kernel      us     TFLOPS\n", .{});
    var rng: base.Rng = .{ .state = 0x62656e63 };
    for (shapes) |sh| {
        const m_rows = if (sh.n > 100000) @min(rows, 256) else rows;
        const words = (sh.k / 16) * (sh.n / 16) * @as(usize, sh.halves) * 8;
        const trellis = try gpa.alloc(i16, words);
        defer gpa.free(trellis);
        for (trellis) |*v| v.* = @bitCast(@as(u16, @truncate(rng.next())));
        const ones = try gpa.alloc(f16, @max(sh.k, sh.n));
        defer gpa.free(ones);
        for (ones) |*v| v.* = 0.01;
        var dt = try Buffer.fromHost(t, trellis);
        defer dt.free();
        var ds = try Buffer.fromHost(t, ones);
        defer ds.free();
        var dx = try Buffer.alloc(t, m_rows * sh.k * 2);
        defer dx.free();
        try dx.fill8(0x3c, null);
        var out = try Buffer.alloc(t, m_rows * sh.n * 2);
        defer out.free();
        var scratch = try Buffer.alloc(t, launches.Launcher.exl3Scratch(m_rows, @intCast(sh.k), @intCast(sh.n)));
        defer scratch.free();
        const mat: exl3.Matrix = .{ .trellis = dt.ptr, .suh = ds.ptr, .svh = ds.ptr, .bias = 0, .halves = @intCast(sh.halves), .codebook = .mul1, .columns = @intCast(sh.n) };
        for (0..2) |which| {
            var times: [5]f64 = undefined;
            for (0..times.len + 1) |r| {
                const t0 = nowNs();
                for (0..4) |_| {
                    if (which == 0) {
                        try t.on.exl3Project(dx.ptr, 1, mat, @intCast(sh.n), @intCast(sh.k), m_rows, out.ptr, 1, scratch.ptr, t.stream.handle);
                    } else if (!try t.on.exl3Prompt(dx.ptr, 1, mat, @intCast(sh.n), @intCast(sh.k), m_rows, out.ptr, 1, scratch.ptr, t.stream.handle)) return error.NoPromptTile;
                }
                try t.stream.synchronize();
                if (r > 0) times[r - 1] = @as(f64, @floatFromInt(nowNs() - t0)) / 4e3;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            const us = times[2];
            const tf = 2.0 * @as(f64, @floatFromInt(m_rows * sh.n * sh.k)) / us / 1e6;
            std.debug.print("{s:<10} {d:>5}   {s:<8} {d:>9.1} {d:>8.2} ({d:.0}% of 57)\n", .{ sh.name, m_rows, if (which == 0) "decode" else "prompt", us, tf, tf / 57 * 100 });
        }
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
