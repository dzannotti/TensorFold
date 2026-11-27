//! FP4-Local's contract entry: gates 1-4, bit identity across decode rows and prompt rows and cuts, then the Loom GEMV.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Profile = @import("contract_entry.zig").Profile;
const k_ = @import("contract_fp4local.zig");
const gemv_ = @import("contract_mxfp8_gemv.zig");

const g = core.contract.gates;

pub fn run(a: std.mem.Allocator, _: std.Io, _: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var t: g.Tally = .{};
    if (!l.fp4.matrix) {
        std.debug.print("N/A  hip fp4l_*: FP4-Local runs on gfx11's matrix cores only\n", .{});
        return t;
    }
    var cases: [k_.registrations.len]k_.Case = undefined;
    for (&cases, k_.registrations) |*x, reg| x.* = .init(a, &r, s, &l, reg);
    defer for (&cases) |*x| x.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    for (&cases) |*x| try kernels.append(a, x.kernel());
    try g.runAll(a, kernels.items, 0x6b63, &t);
    for (&cases) |*x| {
        const ok = if (x.reg.routed) try shuffled(a, x) else try identity(a, x);
        t.kernels += 1;
        if (ok) t.passed += 1 else t.failed += 1;
    }
    try loom(a, &r, s, &l, &t);
    try experts(a, &r, s, &l, &t);
    return t;
}

/// The Loom grouped-expert kernels against fp4l_wmma's and fp4l_pair's bytes, when the build has them.
fn experts(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, t: *g.Tally) !void {
    if (l.fp4.experts.f == null) {
        std.debug.print("N/A  loom nvfp4_expert_*: built without -Dloom-compile, the matrix tiles serve routed prompts\n", .{});
        return;
    }
    var cases: [k_.loom_registrations.len]k_.Case = undefined;
    for (&cases, k_.loom_registrations) |*x, reg| x.* = .init(a, r, s, l, reg);
    defer for (&cases) |*x| x.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    for (&cases) |*x| try kernels.append(a, x.kernel());
    try g.runAll(a, kernels.items, 0x6b34, t);
    for (&cases) |*x| {
        const ok = try shuffled(a, x);
        t.kernels += 1;
        if (ok) t.passed += 1 else t.failed += 1;
    }
}

/// The Loom MXFP8 GEMV against fp4l_stream's bytes, when the build has it (-Dloom-compile).
fn loom(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, t: *g.Tally) !void {
    if (l.fp4.gemv.f == null) {
        std.debug.print("N/A  loom mxfp8_gemv: built without -Dloom-compile, fp4l_stream serves MXFP8 decode\n", .{});
        return;
    }
    var cases: [gemv_.registrations.len]gemv_.Case = undefined;
    for (&cases, gemv_.registrations) |*x, reg| x.* = .init(a, r, s, l, reg);
    defer for (&cases) |*x| x.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    for (&cases) |*x| try kernels.append(a, x.kernel());
    try g.runAll(a, kernels.items, 0x6c6d, t);
    for (&cases) |*x| {
        const ok = try gemv_.identity(a, x);
        t.kernels += 1;
        if (ok) t.passed += 1 else t.failed += 1;
    }
}

/// A routed plan's pairs come out the same bytes under another plan of them: items in another order, one item split.
fn shuffled(a: std.mem.Allocator, x: *k_.Case) !bool {
    const served = x.reg.served[0];
    if (!try x.prepare(served, .random, 0x5f)) return false;
    try x.run();
    const total = served.rows * x.rowBytes();
    const want = try a.dupe(u8, try x.rig.bytes(x.outOf(x.arg), total));
    defer a.free(want);
    const m: i32 = @intCast(served.rows);
    const half = @divTrunc(m + 1, 2);
    const cut = @divTrunc(half, 3);
    const items = [_]i32{ 0, half, m - half, 1, cut, half - cut, 1, 0, cut };
    var arg = x.arg;
    arg.route.items = try x.rig.input(std.mem.sliceAsBytes(&items));
    retarget(x, &arg, 0);
    x.items = 3;
    try x.launchWith(arg);
    x.items = 2;
    const same = std.mem.eql(u8, try x.rig.bytes(scratchOf(x), total), want);
    const say = "{s} shuffled {s}: items reordered and split, {d} pairs at {d}x{d}\n";
    std.debug.print(say, .{ if (same) "PASS" else "FAIL", x.reg.name, served.rows, served.n, served.k });
    return same;
}

/// Where a check's launch writes: the case's scratch, `at` bytes on.
fn retarget(x: *k_.Case, arg: *k_.Arg, at: usize) void {
    if (x.reg.tile.act()) arg.out16 = x.base16 + at else arg.out = x.base + at;
}

fn scratchOf(x: *const k_.Case) u64 {
    return if (x.reg.tile.act()) x.base16 else x.base;
}

/// The rows of a launch of fewer rows (and of later cuts, for the prompt tile) equal the full launch's bytes.
fn identity(a: std.mem.Allocator, x: *k_.Case) !bool {
    const prompt = x.reg.tile != .stream;
    const served = x.reg.served[0];
    const full: usize = if (prompt) 256 else 8;
    if (!try x.prepare(.{ .rows = full, .n = served.n, .k = served.k }, .random, 0x1d)) return false;
    try x.run();
    const n = served.n;
    const row = x.rowBytes();
    const want = try a.dupe(u8, try x.rig.bytes(x.outOf(x.arg), full * row));
    defer a.free(want);
    const counts: []const usize = if (prompt) &.{ 1, 7, 16, 64, 127, 128, 129, 200 } else &.{ 1, 2, 3, 4, 5, 6, 7 };
    var bad: ?usize = null;
    for (counts) |m| {
        var arg = x.arg;
        arg.m = @intCast(m);
        retarget(x, &arg, 0);
        try x.launchWith(arg);
        if (!std.mem.eql(u8, try x.rig.bytes(scratchOf(x), m * row), want[0 .. m * row])) bad = bad orelse m;
    }
    // a prompt cut at 1, 7, 64, 128 and 129 rows: each piece's rows from its own launch
    if (prompt) for ([_]usize{ 1, 7, 64, 128, 129 }) |cut| {
        var at: usize = 0;
        while (at < full) : (at += cut) {
            const m = @min(cut, full - at);
            var arg = x.arg;
            arg.m = @intCast(m);
            arg.x += at * served.k * 2;
            retarget(x, &arg, at * row);
            try x.launchWith(arg);
        }
        if (!std.mem.eql(u8, try x.rig.bytes(scratchOf(x), full * row), want)) bad = bad orelse 1000 + cut;
    };
    const what = if (prompt) "row counts 1-200 and cuts 1, 7, 64, 128, 129 of 256 rows" else "row counts 1-7 of 8";
    if (bad) |m| {
        std.debug.print("FAIL identity {s}: {s}: first differing case {d}\n", .{ x.reg.name, what, m });
        return false;
    }
    std.debug.print("PASS identity {s}: {s} byte-equal at {d}x{d}\n", .{ x.reg.name, what, n, served.k });
    return true;
}
