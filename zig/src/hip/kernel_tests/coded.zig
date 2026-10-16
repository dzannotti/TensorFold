//! The coded decoders' tiles (launches/coded.zig): block FP8 and BF16 products against float64 on the served shapes,
//! and every row writing the same bytes alone and in a window of 2 to 64 rows (decode), or in a prompt of any length.

const std = @import("std");
const base = @import("rig.zig");
const affine = @import("../launches/affine.zig");
const coded = @import("../launches/coded.zig");
const fp8 = @import("core").quant.fp8;
const Buffer = base.Buffer;
const gpa = base.gpa;
const expect = base.expect;

const Case = struct { name: []const u8, code: coded.Code, n: usize, k: usize };

const cases = [_]Case{
    // Flash Next INT4-AutoRound: DeltaNet out, the healed shared expert's gate | up and down, attention q
    .{ .name = "fp8 out", .code = .fp8, .n = 640, .k = 6144 },
    .{ .name = "fp8 shared gate_up", .code = .fp8, .n = 1280, .k = 2560 },
    .{ .name = "fp8 shared down", .code = .fp8, .n = 2560, .k = 1280 },
    .{ .name = "fp8 many columns", .code = .fp8, .n = 33024, .k = 256 },
    .{ .name = "fp8 ragged", .code = .fp8, .n = 1000, .k = 384 },
    // the hyper-connections' down (few columns, long K) and up (K 320: the stream tile), the DeltaNet a | b, the indexer
    .{ .name = "bf16 hc down", .code = .bf16, .n = 324, .k = 10240 },
    .{ .name = "bf16 hc up", .code = .bf16, .n = 4096, .k = 320 },
    .{ .name = "bf16 a", .code = .bf16, .n = 48, .k = 2560 },
    .{ .name = "bf16 index", .code = .bf16, .n = 640, .k = 2560 },
};

const window = [_]usize{ 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 24, 32, 48, 64 };
const prompt = [_]usize{ 1, 7, 64, 129, 136 };
const most_rows = 136;

/// One case's weights on the host and the device, x, and the float64 reference of every row.
const Product = struct {
    c: Case,
    x: Buffer,
    words: Buffer,
    lo: Buffer,
    hi: Buffer,
    out: Buffer,
    /// [rows, n]: the exact sums, and the sums of |x w| (an fp32 sum's error scale)
    ref: []f64,
    mag: []f64,
    /// FP8 weights rounded to bf16 after their scale, as a prompt's tile reads them: [rows, n]
    ref_dq: []f64,

    fn init(t: *base.Rig, c: Case, rng: *base.Rng) !Product {
        const hx = try gpa.alloc(u16, most_rows * c.k);
        defer gpa.free(hx);
        for (hx) |*v| v.* = base.bf16Bits(@as(f32, @floatCast(rng.normal())) * 0.5);
        const xv = try gpa.alloc(f64, hx.len);
        defer gpa.free(xv);
        for (xv, hx) |*v, b| v.* = base.bf16Value(b);
        const w = try gpa.alloc(f64, c.n * c.k);
        defer gpa.free(w);
        const wq = try gpa.alloc(f64, c.n * c.k);
        defer gpa.free(wq);
        const kb = c.k / 128;
        var words: []u8 = undefined;
        const lo = try gpa.alloc(u16, @max(c.n * kb, 1));
        defer gpa.free(lo);
        const hi = try gpa.alloc(u16, @max(c.n * kb, 1));
        defer gpa.free(hi);
        @memset(lo, 0);
        @memset(hi, 0);
        if (c.code == .fp8) {
            words = try gpa.alloc(u8, c.n * c.k);
            const blocks = try gpa.alloc(f32, ((c.n + 127) / 128) * kb);
            defer gpa.free(blocks);
            for (blocks) |*s| s.* = @floatCast(0.001 + 0.05 * rng.uniform());
            for (words, 0..) |*b, i| {
                var v: u8 = @truncate(rng.next() >> 32);
                if (v & 0x7f == 0x7f) v ^= 1; // no NaN codes
                b.* = v;
                const r = i / c.k;
                const s = blocks[(r / 128) * kb + (i % c.k) / 128];
                w[i] = fp8.e4m3(v) * s;
                wq[i] = base.bf16Value(@import("core").quant.convert.bf16(@floatCast(fp8.e4m3(v) * @as(f64, s))));
            }
            for (0..c.n) |r| for (0..kb) |g| {
                const bits: u32 = @bitCast(blocks[(r / 128) * kb + g]);
                lo[r * kb + g] = @truncate(bits);
                hi[r * kb + g] = @truncate(bits >> 16);
            };
        } else {
            words = try gpa.alloc(u8, 2 * c.n * c.k);
            for (w, wq, 0..) |*v, *q, i| {
                const b = base.bf16Bits(@as(f32, @floatCast(rng.normal())) * 0.05);
                std.mem.writeInt(u16, words[2 * i ..][0..2], b, .little);
                v.* = base.bf16Value(b);
                q.* = v.*;
            }
        }
        defer gpa.free(words);
        const ref = try gpa.alloc(f64, most_rows * c.n);
        const mag = try gpa.alloc(f64, most_rows * c.n);
        const ref_dq = try gpa.alloc(f64, most_rows * c.n);
        for (0..most_rows) |r| for (0..c.n) |col| {
            var s: f64 = 0;
            var m: f64 = 0;
            var q: f64 = 0;
            for (0..c.k) |i| {
                const xv_ = xv[r * c.k + i];
                s += xv_ * w[col * c.k + i];
                m += @abs(xv_ * w[col * c.k + i]);
                q += xv_ * wq[col * c.k + i];
            }
            ref[r * c.n + col] = s;
            mag[r * c.n + col] = m;
            ref_dq[r * c.n + col] = q;
        };
        return .{
            .c = c,
            .x = try Buffer.fromHost(t, hx),
            .words = try Buffer.fromHost(t, words),
            .lo = try Buffer.fromHost(t, lo),
            .hi = try Buffer.fromHost(t, hi),
            .out = try Buffer.alloc(t, most_rows * c.n * 4),
            .ref = ref,
            .mag = mag,
            .ref_dq = ref_dq,
        };
    }

    fn deinit(p: *Product) void {
        p.x.free();
        p.words.free();
        p.lo.free();
        p.hi.free();
        p.out.free();
        gpa.free(p.ref);
        gpa.free(p.mag);
        gpa.free(p.ref_dq);
    }

    /// Rows [row0, row0 + m) through the tile into fp32 rows, downloaded.
    fn run(p: *Product, t: *base.Rig, row0: usize, m: usize, prefill: bool, out: []f32) !void {
        const c = p.c;
        const tables = if (c.code == .fp8) [2]u64{ p.lo.ptr, p.hi.ptr } else [2]u64{ p.words.ptr, p.words.ptr };
        const arg: affine.Arg = .{
            .x = p.x.ptr + row0 * c.k * 2,
            .words = p.words.ptr,
            .scale = .{ .p = tables[0], .kind = 1 },
            .bias = .{ .p = tables[1], .kind = 1 },
            .out = p.out.ptr,
            .m = @intCast(m),
            .n = @intCast(c.n),
            .k = @intCast(c.k),
            .bits = if (c.code == .fp8) 8 else 16,
            .group = 128,
            .fp16 = 0,
        };
        try p.out.fill8(0xA5, null);
        try t.on.coded.?.run(&t.r, c.code, arg, prefill, t.stream.handle);
        try t.stream.synchronize();
        try p.out.download(0, std.mem.sliceAsBytes(out[0 .. m * c.n]));
    }
};

fn check(t: *base.Rig, c: Case, rng: *base.Rng) !void {
    var p = try Product.init(t, c, rng);
    defer p.deinit();
    const alone = try gpa.alloc(f32, most_rows * c.n);
    defer gpa.free(alone);
    const got = try gpa.alloc(f32, most_rows * c.n);
    defer gpa.free(got);
    // a window: each of 64 rows alone within an fp32 sum of float64, then the same bytes in every window size
    const rows = 64;
    var worst: f64 = 0;
    for (0..rows) |r| {
        try p.run(t, r, 1, false, alone[r * c.n ..]);
        for (0..c.n) |col| {
            const e = @abs(alone[r * c.n + col] - p.ref[r * c.n + col]) / (p.mag[r * c.n + col] + 1e-30);
            worst = @max(worst, e);
        }
    }
    try expect(worst < 2e-5, "{s}: a window row is {e} of its |x w| sum from float64", .{ c.name, worst });
    for (window) |m| {
        try p.run(t, 0, m, false, got);
        try expect(std.mem.eql(f32, got[0 .. m * c.n], alone[0 .. m * c.n]), "{s}: a window of {d} rows writes other bytes than its rows alone", .{ c.name, m });
    }
    // a prompt: its rows against float64 of the weights as the tile rounds them, the same bytes at every length
    try p.run(t, 0, most_rows, true, alone);
    worst = 0;
    for (0..most_rows * c.n) |i| worst = @max(worst, @abs(alone[i] - p.ref_dq[i]) / (p.mag[i] + 1e-30));
    try expect(worst < 2e-5, "{s}: a prompt row is {e} of its |x w| sum from float64", .{ c.name, worst });    for (prompt) |m| {
        try p.run(t, most_rows - m, m, true, got);
        try expect(std.mem.eql(f32, got[0 .. m * c.n], alone[(most_rows - m) * c.n ..][0 .. m * c.n]), "{s}: a prompt of {d} rows writes other bytes", .{ c.name, m });
    }
}

test "block FP8 and BF16 products sit within an fp32 sum of float64, and every row keeps its bytes in any window or prompt" {
    const t = try base.Rig.open(64 << 20);
    defer t.close();
    if (t.on.coded == null) return error.SkipZigTest;
    var rng: base.Rng = .{ .state = 0x51ED2701 };
    for (cases) |c| try check(t, c, &rng);
}
