//! GGUF projections on the GPU against ggml's dequantize_row_* (core/quant/gguf.zig's fp32 reference, widened):
//! the row gather value for value (a zero may carry the other sign: s u + mn adds to +0 where ggml's product is -0), the product against fp64 with every row's bits alone equal to its bits among 1 to 128
//! rows, and the prompt product against fp64 of its BF16 weights with its bits independent of the row count.

const std = @import("std");
const base = @import("rig.zig");
const gguf = @import("core").quant.gguf;
const rne = @import("core").quant.convert.bf16;
const gpa = base.gpa;
const Buffer = base.Buffer;

const together = [_]usize{ 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 34, 40, 48, 63, 64, 65, 66, 80, 96, 97, 100, 112, 120, 127, 128 };
const most_rows = 128;

/// The byte offsets of a superblock's fp16 scales (d, dmin), kept finite and small; the other bytes stay random.
fn halves(t: gguf.Type) []const usize {
    return switch (t) {
        .q4_k, .q5_k => &.{ 0, 2 },
        .q6_k => &.{208},
        .q3_k => &.{108},
        .iq4_xs, .iq3_s => &.{0},
        .q8_0 => &.{ 0, 34, 68, 102, 136, 170, 204, 238 },
        .iq4_nl => &.{ 0, 18, 36, 54, 72, 90, 108, 126 },
        else => unreachable,
    };
}

/// Random blocks of type `t` for `n` rows of `k`.
fn blocks(rng: *base.Rng, t: gguf.Type, n: usize, k: usize) ![]u8 {
    const blk = t.block().?;
    const row = k / blk.elems * blk.bytes;
    const out = try gpa.alloc(u8, n * row);
    for (out) |*b| b.* = @truncate(rng.next());
    const super: usize = 256 / blk.elems * blk.bytes;
    var at: usize = 0;
    while (at < out.len) : (at += super) for (halves(t)) |o| {
        const v: f16 = @floatCast(0.001 + 0.004 * rng.unit());
        std.mem.writeInt(u16, out[at + o ..][0..2], @bitCast(v), .little);
    };
    return out;
}

/// prompt_pack_rn's BF16: round to nearest, ties away from zero.
fn bf16Rna(v: f32) f64 {
    const u: u32 = @bitCast(v);
    return @as(f32, @bitCast((u +% 0x8000) & 0xffff0000));
}

fn run(t: *base.Rig, rng: *base.Rng, kind: gguf.Type, k: usize, n: usize) !void {
    const raw = try blocks(rng, kind, n, k);
    defer gpa.free(raw);
    const row_bytes = raw.len / n;
    const host: gguf.Host = .{ .blocks = .{ .dtype = .u8, .rank = 2, .shape = .{ n, row_bytes, 1, 1, 1 }, .bytes = raw }, .type = kind, .k = k };
    const w = try gpa.alloc(f64, n * k);
    defer gpa.free(w);
    try gguf.reference(host, w);
    var dw = try Buffer.fromHost(t, raw);
    defer dw.free();
    const mat: gguf.Matrix = .{ .blocks = dw.ptr, .type = kind, .row_bytes = @intCast(row_bytes) };
    const name = @tagName(kind);

    // the gather: every row as ggml's fp32 values, bit for bit
    {
        var out = try Buffer.alloc(t, n * k * 4);
        defer out.free();
        try t.on.ggufRows(mat, 0, n, @intCast(k), out.ptr, 0, t.stream.handle);
        try t.stream.synchronize();
        const got = try gpa.alloc(f32, n * k);
        defer gpa.free(got);
        try out.download(0, std.mem.sliceAsBytes(got));
        for (got, w, 0..) |g, want, i| try base.expect(@as(f64, g) == want, "{s} rows: element {d} is {e}, ggml's {e}", .{ name, i, g, want });
    }

    const pm = 300;
    const x = try gpa.alloc(u16, pm * k);
    defer gpa.free(x);
    for (x) |*v| v.* = base.bf16Bits(rng.unit());
    var dx = try Buffer.fromHost(t, x);
    defer dx.free();
    var out = try Buffer.alloc(t, pm * n * 4);
    defer out.free();
    const all = try gpa.alloc(f32, pm * n);
    defer gpa.free(all);
    const got = try gpa.alloc(f32, pm * n);
    defer gpa.free(got);

    // the product: 128 rows against fp64, then each row count's rows against the same rows among the 128
    try t.on.ggufProject(dx.ptr, mat, @intCast(n), @intCast(k), most_rows, out.ptr, 0, t.stream.handle);
    try t.stream.synchronize();
    try out.download(0, std.mem.sliceAsBytes(all[0 .. most_rows * n]));
    var err: f64 = 0;
    var mag: f64 = 0;
    for ([_]usize{ 0, 1, 17, 64, 127 }) |r| for (0..n) |j| {
        var want: f64 = 0;
        for (0..k) |i| want += base.bf16Value(x[r * k + i]) * w[j * k + i];
        err = @max(err, @abs(want - all[r * n + j]));
        mag = @max(mag, @abs(want));
    };
    try base.expect(err <= 1e-5 * mag, "{s} product: max error {e} of {e}", .{ name, err, mag });
    for (together) |rows| {
        const first = (rows * 5) % (most_rows - rows + 1);
        try t.on.ggufProject(dx.ptr + first * k * 2, mat, @intCast(n), @intCast(k), rows, out.ptr, 0, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. rows * n]));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got[0 .. rows * n]), std.mem.sliceAsBytes(all[first * n ..][0 .. rows * n]));
        try base.expect(same, "{s} product: {d} rows from row {d} differ from the same rows among 128", .{ name, rows, first });
    }

    // the prompt product: 300 rows (past one 256-row block) against fp64 of its BF16 weights, then each row count's
    // rows (the chain up to 128, the tile past it) against the same rows among the 300
    try base.expect(try t.on.ggufPrompt(dx.ptr, mat, @intCast(n), @intCast(k), pm, out.ptr, 0, t.stream.handle), "{s}: prompt refused", .{name});
    try t.stream.synchronize();
    try out.download(0, std.mem.sliceAsBytes(all));
    var perr: f64 = 0;
    var pmag: f64 = 0;
    for ([_]usize{ 0, 1, 130, 257, 299 }) |r| for (0..n) |j| {
        var want: f64 = 0;
        for (0..k) |i| want += base.bf16Value(x[r * k + i]) * bf16Rna(@floatCast(w[j * k + i]));
        perr = @max(perr, @abs(want - all[r * n + j]));
        pmag = @max(pmag, @abs(want));
    };
    try base.expect(perr <= 1e-5 * pmag, "{s} prompt: max error {e} of {e}", .{ name, perr, pmag });
    for (together ++ [_]usize{ 129, 200, 256, 257, 300 }) |rows| {
        const first = (rows * 5) % (pm - rows + 1);
        try base.expect(try t.on.ggufPrompt(dx.ptr + first * k * 2, mat, @intCast(n), @intCast(k), rows, out.ptr, 0, t.stream.handle), "{s}: prompt refused", .{name});
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. rows * n]));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got[0 .. rows * n]), std.mem.sliceAsBytes(all[first * n ..][0 .. rows * n]));
        try base.expect(same, "{s} prompt: {d} rows from row {d} differ from the same rows among {d}", .{ name, rows, first, pm });
    }

    // bf16 out: the fp32 sums rounded to nearest even (decode), and the gather of a few ids
    {
        var o16 = try Buffer.alloc(t, @max(16 * n, 3 * k) * 2);
        defer o16.free();
        try t.on.ggufProject(dx.ptr, mat, @intCast(n), @intCast(k), 16, out.ptr, 0, t.stream.handle);
        try t.on.ggufProject(dx.ptr, mat, @intCast(n), @intCast(k), 16, o16.ptr, 1, t.stream.handle);
        try t.stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. 16 * n]));
        const h = try gpa.alloc(u16, @max(16 * n, 3 * k));
        defer gpa.free(h);
        try o16.download(0, std.mem.sliceAsBytes(h[0 .. 16 * n]));
        for (h[0 .. 16 * n], got[0 .. 16 * n]) |b, f| try base.expect(b == rne(f), "{s}: bf16 out {x} of {e}", .{ name, b, f });
        const ids = [_]i32{ @intCast(n - 1), 0, 3 };
        var dids = try Buffer.fromHost(t, &ids);
        defer dids.free();
        try t.on.ggufRows(mat, dids.ptr, ids.len, @intCast(k), o16.ptr, 1, t.stream.handle);
        try t.stream.synchronize();
        try o16.download(0, std.mem.sliceAsBytes(h[0 .. ids.len * k]));
        for (ids, 0..) |id, r| for (0..k) |i| {
            const want = rne(@floatCast(w[@as(usize, @intCast(id)) * k + i]));
            try base.expect(h[r * k + i] == want or (h[r * k + i] | want) == 0x8000, "{s}: gathered row {d} element {d}: {x} want {x} of {e}", .{ name, id, i, h[r * k + i], want, w[@as(usize, @intCast(id)) * k + i] });
        };
    }
}

test "GGUF products match ggml's dequantization in fp64 and keep every row's bits at any row count" {
    const t = try base.Rig.open(1 << 20);
    defer t.close();
    var rng: base.Rng = .{ .state = 0x67677566 };
    inline for (gguf.kinds) |kind| {
        // six superblocks (the four waves take two, two, one and one) on 80 columns (a short column block), and four
        // on 48 (the 27B's in_proj_a and b)
        try run(t, &rng, kind, 1536, 80);
        try run(t, &rng, kind, 1024, 48);
    }
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// The 27B's shapes in the UD-Q4_K_XL file's main types (TF_GGUF_BENCH=1; TF_GGUF_BENCH_M rows for the prompt, default
// a prefill span's 2048): the product at 1, 16, 64 and 128 rows in GB/s of weights, the prompt tile in TFLOPS.
test "GGUF timings on the 27B shapes (TF_GGUF_BENCH=1)" {
    _ = std.c.getenv("TF_GGUF_BENCH") orelse return;
    const t = try base.Rig.open(1 << 20);
    defer t.close();
    const prompt_m: usize = if (std.c.getenv("TF_GGUF_BENCH_M")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
    const Shape = struct { name: []const u8, k: usize, n: usize, type: gguf.Type };
    const shapes = [_]Shape{
        .{ .name = "qkv q5_K", .k = 5120, .n = 10240, .type = .q5_k },
        .{ .name = "qkv q4_K", .k = 5120, .n = 10240, .type = .q4_k },
        .{ .name = "z q5_K", .k = 5120, .n = 6144, .type = .q5_k },
        .{ .name = "out q5_K", .k = 6144, .n = 5120, .type = .q5_k },
        .{ .name = "out q6_K", .k = 6144, .n = 5120, .type = .q6_k },
        .{ .name = "gate iq4_xs", .k = 5120, .n = 17408, .type = .iq4_xs },
        .{ .name = "gate q5_K", .k = 5120, .n = 17408, .type = .q5_k },
        .{ .name = "down q5_K", .k = 17408, .n = 5120, .type = .q5_k },
        .{ .name = "down iq4_xs", .k = 17408, .n = 5120, .type = .iq4_xs },
        .{ .name = "down iq4_nl", .k = 17408, .n = 5120, .type = .iq4_nl },
        .{ .name = "k q6_K", .k = 5120, .n = 1024, .type = .q6_k },
        .{ .name = "v q8_0", .k = 5120, .n = 1024, .type = .q8_0 },
        .{ .name = "head q6_K", .k = 5120, .n = 248320, .type = .q6_k },
    };
    std.debug.print("\nshape          kernel   M       us     GB/s   TFLOPS\n", .{});
    var rng: base.Rng = .{ .state = 0x62656e63 };
    for (shapes) |sh| {
        const raw = try blocks(&rng, sh.type, sh.n, sh.k);
        defer gpa.free(raw);
        var dw = try Buffer.fromHost(t, raw);
        defer dw.free();
        const mat: gguf.Matrix = .{ .blocks = dw.ptr, .type = sh.type, .row_bytes = @intCast(raw.len / sh.n) };
        const pm = if (sh.n > 100000) @min(prompt_m, 256) else prompt_m;
        var dx = try Buffer.alloc(t, pm * sh.k * 2);
        defer dx.free();
        try dx.fill8(0x3c, null);
        var out = try Buffer.alloc(t, pm * sh.n * 2);
        defer out.free();
        for ([_]usize{ 1, 16, 64, 128, 0 }) |m0| {
            const m = if (m0 == 0) pm else m0;
            var times: [5]f64 = undefined;
            for (0..times.len + 1) |r| {
                const t0 = nowNs();
                for (0..4) |_| {
                    if (m0 == 0) {
                        _ = try t.on.ggufPrompt(dx.ptr, mat, @intCast(sh.n), @intCast(sh.k), m, out.ptr, 1, t.stream.handle);
                    } else try t.on.ggufProject(dx.ptr, mat, @intCast(sh.n), @intCast(sh.k), m, out.ptr, 1, t.stream.handle);
                }
                try t.stream.synchronize();
                if (r > 0) times[r - 1] = @as(f64, @floatFromInt(nowNs() - t0)) / 4e3;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            const us = times[2];
            const gbs = @as(f64, @floatFromInt(raw.len)) / us / 1e3;
            const tf = 2.0 * @as(f64, @floatFromInt(m * sh.n * sh.k)) / us / 1e6;
            std.debug.print("{s:<14} {s:<7} {d:>5} {d:>9.1} {d:>8.1} {d:>8.2}\n", .{ sh.name, if (m0 == 0) "prompt" else "product", m, us, gbs, tf });
        }
    }
}
