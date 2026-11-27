//! The Loom hyper-connection read-out against the hipcc chain it replaces: every output byte-equal, rows and cuts.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const g = core.contract.gates;
const fp4 = hip.launches.fp4local;
const hcr = hip.launches.hc_readout;

const S = hcr.streams;
const D = hcr.dims;
const W = S * D;
const low = 320;
const eps: f32 = 1e-6;

const Shape = struct { rows: usize, n: usize, add: bool };
/// Rows across both down launches (the wide one from 1,024), the down with and without the inject rows.
const shapes = [_]Shape{
    .{ .rows = 1, .n = 324, .add = true },    .{ .rows = 7, .n = 320, .add = false },
    .{ .rows = 129, .n = 324, .add = true },  .{ .rows = 300, .n = 320, .add = true },
    .{ .rows = 1100, .n = 324, .add = true }, .{ .rows = 1030, .n = 320, .add = false },
};

/// Every shape, then a 1,100-row read-out cut at 1, 129 and 1,024 rows byte-equal to one launch; N/A without Loom.
pub fn check(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, t: *g.Tally) !void {
    if (l.hcr.f == null) {
        std.debug.print("N/A  loom hc_readout: built without -Dloom-compile, the hipcc chain serves prompts\n", .{});
        return;
    }
    var rig: rig_.Rig = .{ .a = a, .r = r, .s = s };
    defer rig.deinit();
    for (shapes) |sh| try tally(t, try equal(a, &rig, l, sh));
    try tally(t, try cuts(a, &rig, l));
}

fn tally(t: *g.Tally, ok: bool) !void {
    t.kernels += 1;
    if (ok) t.passed += 1 else t.failed += 1;
}

const Bufs = struct { h0: []u16, branch: []u16, inject: []u16, scale: []f32, down: []u16, up: []u16 };

/// Random inputs: streams over decades with NaN, inf and signed zeros sprinkled in, a block's branch and gates.
fn inputs(a: std.mem.Allocator, rows: usize, n: usize, seed: u64) !Bufs {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const b: Bufs = .{
        .h0 = try a.alloc(u16, rows * W),
        .branch = try a.alloc(u16, rows * D),
        .inject = try a.alloc(u16, rows * S),
        .scale = try a.alloc(f32, W),
        .down = try a.alloc(u16, n * W),
        .up = try a.alloc(u16, W * low),
    };
    for (b.h0, 0..) |*v, i| {
        const mag = std.math.pow(f32, 10, @floatFromInt(@as(i32, @intCast((i / W) % 5)) - 2));
        v.* = bf16(rnd.floatNorm(f32) * mag);
        if (i % 9973 == 0) v.* = 0x8000;
        if (i % 104729 == 17) v.* = if (i % 2 == 0) 0x7f80 else 0x7fc1;
    }
    for (b.branch) |*v| v.* = bf16(rnd.floatNorm(f32));
    for (b.inject) |*v| v.* = bf16(2 * rnd.float(f32));
    for (b.scale, 0..) |*v, i| v.* = 0.5 + rnd.float(f32) - (if (i % 37 == 0) @as(f32, 1.2) else 0);
    for (b.down) |*v| v.* = bf16(rnd.floatNorm(f32) * 0.02);
    for (b.up) |*v| v.* = bf16(rnd.floatNorm(f32) * 0.05);
    return b;
}

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn product(l: *const hip.Launcher, r: *const hip.Runtime, x: u64, w: u64, out: u64, m: usize, n: usize, k: usize, s: hip.Stream) !void {
    const arg: fp4.Arg = .{
        .x = x,
        .words = w,
        .scale = .{ .p = 0, .kind = 0 },
        .bias = .{ .p = 0, .kind = 0 },
        .out = out,
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .bits = fp4.Format.chunkWords(.bf16),
        .group = fp4.bf16Fold(@intCast(k)),
        .fp16 = 0,
    };
    try l.fp4.tilePromptGo(r, .bf16, arg, 1, 0, s.handle);
}

const Outs = struct { h: u64, x: u64, act: u64, inject: u64, mixed: u64 };

/// The chain main runs on rows `m` of the inputs at `in`.
fn chain(rig: *rig_.Rig, l: *const hip.Launcher, in: Ins, m: usize, n: usize, add: bool) !Outs {
    const s = rig.s.handle;
    const o: Outs = .{ .h = try rig.scratch(m * W * 2), .x = try rig.output(m * W * 2), .act = try rig.output(m * low * 2), .inject = try rig.output(m * S * 2), .mixed = try rig.output(m * D * 2) };
    try rig.put(o.h, try rig.bytes(in.h, m * W * 2));
    const P = ?*anyopaque;
    const ptr = struct {
        fn p(v: u64) P {
            return @ptrFromInt(v);
        }
    }.p;
    try l.fn_hc_norm(ptr(o.h), if (add) ptr(in.branch) else null, if (add) ptr(in.inject) else null, if (add) ptr(o.h) else null, ptr(o.x), @ptrFromInt(in.scale), 2, @intCast(m), S, D, eps, s);
    const z = try rig.scratch(m * n * 4);
    try product(l, rig.r, o.x, in.down, z, m, n, W, rig.s);
    try l.fn_hc_gates(@ptrFromInt(z), ptr(o.act), if (n > low) ptr(o.inject) else null, 2, @intCast(m), low, S, s);
    const up = try rig.scratch(m * W * 4);
    try product(l, rig.r, o.act, in.up, up, m, W, low, rig.s);
    try l.fn_hc_mix(@ptrFromInt(up), ptr(o.x), ptr(o.mixed), 2, @intCast(m), S, D, s);
    return o;
}

/// The Loom read-out on rows [r0, r0 + m) of the inputs into `o` (row r0 of each output at its offset).
fn fused(rig: *rig_.Rig, l: *const hip.Launcher, in: Ins, o: Outs, r0: usize, m: usize, n: usize, add: bool) !void {
    const s = rig.s.handle;
    const h = o.h + r0 * W * 2;
    const x = o.x + r0 * W * 2;
    const act = o.act + r0 * low * 2;
    const inj = o.inject + r0 * S * 2;
    try l.hcr.norm(rig.r, h, if (add) in.branch + r0 * D * 2 else null, if (add) in.inject + r0 * S * 2 else null, in.scale, x, m, eps, s);
    try l.hcr.down(rig.r, x, in.down, act, if (n > low) inj else null, m, n, low, s);
    try l.hcr.upMix(rig.r, act, in.up, x, o.mixed + r0 * D * 2, m, low, s);
}

const Ins = struct { h: u64, branch: u64, inject: u64, scale: u64, down: u64, up: u64 };

fn upload(rig: *rig_.Rig, b: Bufs) !Ins {
    return .{
        .h = try rig.input(std.mem.sliceAsBytes(b.h0)),
        .branch = try rig.input(std.mem.sliceAsBytes(b.branch)),
        .inject = try rig.input(std.mem.sliceAsBytes(b.inject)),
        .scale = try rig.input(std.mem.sliceAsBytes(b.scale)),
        .down = try rig.input(std.mem.sliceAsBytes(b.down)),
        .up = try rig.input(std.mem.sliceAsBytes(b.up)),
    };
}

fn fusedOuts(rig: *rig_.Rig, in: Ins, m: usize) !Outs {
    const o: Outs = .{ .h = try rig.scratch(m * W * 2), .x = try rig.output(m * W * 2), .act = try rig.output(m * low * 2), .inject = try rig.output(m * S * 2), .mixed = try rig.output(m * D * 2) };
    try rig.put(o.h, try rig.bytes(in.h, m * W * 2));
    return o;
}

/// The names of the outputs that differ between `want` and `got` (rows m), or null when all are byte-equal.
fn differs(a: std.mem.Allocator, rig: *rig_.Rig, want: Outs, got: Outs, m: usize, n: usize) !?[]const u8 {
    const names = [_][]const u8{ "h", "x", "act", "inject", "mixed" };
    const sizes = [_]usize{ m * W * 2, m * W * 2, m * low * 2, if (n > low) m * S * 2 else 0, m * D * 2 };
    const ws = [_]u64{ want.h, want.x, want.act, want.inject, want.mixed };
    const gs = [_]u64{ got.h, got.x, got.act, got.inject, got.mixed };
    for (names, sizes, ws, gs) |name, len, wp, gp| {
        if (len == 0) continue;
        const w = try a.dupe(u8, try rig.bytes(wp, len));
        defer a.free(w);
        if (!std.mem.eql(u8, w, try rig.bytes(gp, len))) return name;
    }
    return null;
}

fn equal(a: std.mem.Allocator, rig: *rig_.Rig, l: *const hip.Launcher, sh: Shape) !bool {
    rig.reset();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const in = try upload(rig, try inputs(arena.allocator(), sh.rows, sh.n, 0x6863 + sh.rows));
    const want = try chain(rig, l, in, sh.rows, sh.n, sh.add);
    const got = try fusedOuts(rig, in, sh.rows);
    try fused(rig, l, in, got, 0, sh.rows, sh.n, sh.add);
    const bad = try differs(a, rig, want, got, sh.rows, sh.n);
    const say = "{s} loom hc_readout: {d} rows, down {d} x {d}{s}: h, x, act, inject, mixed byte-equal to the hipcc chain{s}{s}\n";
    std.debug.print(say, .{ if (bad == null) "PASS" else "FAIL", sh.rows, sh.n, W, if (sh.add) " with a branch" else "", if (bad != null) "; first differing: " else "", bad orelse "" });
    return bad == null;
}

fn cuts(a: std.mem.Allocator, rig: *rig_.Rig, l: *const hip.Launcher) !bool {
    rig.reset();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const m = 1100;
    const n = 324;
    const in = try upload(rig, try inputs(arena.allocator(), m, n, 0x6375));
    const want = try fusedOuts(rig, in, m);
    try fused(rig, l, in, want, 0, m, n, true);
    var bad: ?usize = null;
    for ([_]usize{ 1, 129, 1024 }) |cut| {
        const got = try fusedOuts(rig, in, m);
        try fused(rig, l, in, got, 0, cut, n, true);
        try fused(rig, l, in, got, cut, m - cut, n, true);
        if (try differs(a, rig, want, got, m, n) != null) bad = bad orelse cut;
    }
    if (bad) |c| {
        std.debug.print("FAIL loom hc_readout cuts: 1,100 rows cut at {d} differ from one launch\n", .{c});
        return false;
    }
    std.debug.print("PASS loom hc_readout cuts: 1,100 rows cut at 1, 129 and 1,024 byte-equal to one launch\n", .{});
    return true;
}
