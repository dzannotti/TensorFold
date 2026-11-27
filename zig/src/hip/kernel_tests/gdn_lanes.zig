//! The CUDA family's DeltaNet lanes (front, chain with a pending-row log, roll) against today's per-stream launches.
const std = @import("std");
const platform = @import("sys");
const rig = @import("rig.zig");
const plan = @import("../launches/plan.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const key_heads = 16;
const value_heads = 48;
const dk = 128;
const dv = 128;
const conv = 4;
const taps = conv - 1;
const kw = key_heads * dk;
const vw = value_heads * dv;
const channels = 2 * kw + vw;
const state_len = value_heads * dk * dv;
const log_rows = 48;
const log_len = log_rows * (kw + vw + 2 * value_heads);
const eps = 1e-6;

/// Weights and gate inputs one layer's front takes.
const Layer = struct {
    w: DeviceBuffer,
    qw: DeviceBuffer,
    kw_w: DeviceBuffer,
    a_log: DeviceBuffer,
    dt: DeviceBuffer,

    fn make(t: *Rig, rng: *rig.Rng) !Layer {
        const hw = try gpa.alloc(f32, channels * conv);
        defer gpa.free(hw);
        for (hw) |*e| e.* = rng.unit() / 4;
        const hn = try gpa.alloc(f32, kw);
        defer gpa.free(hn);
        for (hn) |*e| e.* = rng.unit() + 1;
        const ha = try gpa.alloc(f32, value_heads);
        defer gpa.free(ha);
        for (ha) |*e| e.* = rng.unit() - 1;
        const hd = try gpa.alloc(f32, value_heads);
        defer gpa.free(hd);
        for (hd) |*e| e.* = rng.unit();
        return .{ .w = try t.upload(hw), .qw = try t.upload(hn), .kw_w = try t.upload(hn), .a_log = try t.upload(ha), .dt = try t.upload(hd) };
    }

    fn free(l: *Layer) void {
        for ([_]*DeviceBuffer{ &l.w, &l.qw, &l.kw_w, &l.a_log, &l.dt }) |b| b.free();
    }
};

fn randomBits(t: *Rig, n: usize, rng: *rig.Rng, scale: f32) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*e| e.* = t.bits(rng.unit() * scale);
    return out;
}

/// One lane's buffers over `layers` layers.
const Lane = struct {
    window: DeviceBuffer,
    state: DeviceBuffer,
    spare: DeviceBuffer,
    log: DeviceBuffer,
    samples: DeviceBuffer,
    table: DeviceBuffer,

    fn make(t: *Rig, rng: *rig.Rng, layers: usize) !Lane {
        const hs = try gpa.alloc(f32, state_len);
        defer gpa.free(hs);
        const hwin = try gpa.alloc(f32, taps * channels);
        defer gpa.free(hwin);
        for (hwin) |*e| e.* = rng.unit();
        var l: Lane = .{
            .window = try t.upload(hwin),
            .state = try t.alloc(layers * state_len * 4),
            .spare = try t.alloc(layers * state_len * 4),
            .log = try t.alloc(layers * log_len * 4),
            .samples = try t.alloc(8 * channels * 4),
            .table = try t.alloc(layers * 5 * 8),
        };
        for (0..layers) |i| {
            for (hs) |*e| e.* = 0.1 * rng.unit();
            try l.state.upload(i * state_len * 4, std.mem.sliceAsBytes(hs));
        }
        const words = try gpa.alloc(u64, layers * 5);
        defer gpa.free(words);
        for (0..layers) |i| {
            words[5 * i] = at(l.window);
            words[5 * i + 1] = at(l.state) + i * state_len * 4;
            words[5 * i + 2] = at(l.spare) + i * state_len * 4;
            words[5 * i + 3] = at(l.log) + i * log_len * 4;
            words[5 * i + 4] = at(l.samples);
        }
        try l.table.upload(0, std.mem.sliceAsBytes(words));
        return l;
    }

    fn free(l: *Lane) void {
        for ([_]*DeviceBuffer{ &l.window, &l.state, &l.spare, &l.log, &l.samples, &l.table }) |b| b.free();
    }
};

/// A round's words: first and rows a slot, [pending, log first, kept, 0] a slot, the tables.
const Words = struct {
    first: DeviceBuffer,
    count: DeviceBuffer,
    ctl: DeviceBuffer,
    tab: DeviceBuffer,

    fn make(t: *Rig, lanes: []const Lane, counts: []const u32, ctl: []const i32) !Words {
        const firsts = try gpa.alloc(u32, lanes.len);
        defer gpa.free(firsts);
        var sum: u32 = 0;
        for (counts, firsts) |c, *f| {
            f.* = sum;
            sum += c;
        }
        const tabs = try gpa.alloc(u64, lanes.len);
        defer gpa.free(tabs);
        for (lanes, tabs) |l, *p| p.* = at(l.table);
        return .{ .first = try t.upload(firsts), .count = try t.upload(counts), .ctl = try t.upload(ctl), .tab = try t.upload(tabs) };
    }

    fn args(w: *const Words) plan.LaneArgs {
        return .{ .first = at(w.first), .count = at(w.count), .ctl = at(w.ctl), .tab = at(w.tab) };
    }

    fn free(w: *Words) void {
        for ([_]*DeviceBuffer{ &w.first, &w.count, &w.ctl, &w.tab }) |b| b.free();
    }
};

fn front(t: *Rig, w: *const Words, l: Layer, x: DeviceBuffer, ga: DeviceBuffer, gb: DeviceBuffer, qn: DeviceBuffer, layer: usize, slots: usize, widest: usize) !void {
    try t.on.laneFront(.{
        .x = at(x),
        .kind = t.kind(),
        .weight = at(l.w),
        .qw = at(l.qw),
        .kw_w = at(l.kw_w),
        .eps = eps,
        .qn = at(qn),
        .ga = at(ga),
        .gb = at(gb),
        .a_log = at(l.a_log),
        .dt_bias = at(l.dt),
        .channels = channels,
        .kernel = conv,
        .kw = kw,
        .vw = vw,
        .heads = value_heads,
        .layer = @intCast(layer),
        .log_rows = log_rows,
        .lanes = w.args(),
    }, slots, widest, t.stream.handle);
}

fn near(want: []const f32, got: []const f32, what: []const u8) !void {
    var worst: f32 = 0;
    for (want, got) |a, b| worst = @max(worst, @abs(a - b) / (1e-2 + @abs(a)));
    if (worst > 2e-3) std.debug.print("{s}: worst relative difference {d}\n", .{ what, worst });
    try std.testing.expect(worst <= 2e-3);
}

test "a partly kept window stays in the log: the next window's chain meets today's serial run" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x1A7E };
    var layer = try Layer.make(t, &rng);
    defer layer.free();
    const ra = 4;
    const rb = 3;
    const kept = 2;
    const total = kept + rb;
    // rows of the serial run: the two kept rows, then the second window's
    const hx = try randomBits(t, (ra + rb) * channels, &rng, 2);
    defer gpa.free(hx);
    const hga = try randomBits(t, (ra + rb) * value_heads, &rng, 1);
    defer gpa.free(hga);
    const hgb = try randomBits(t, (ra + rb) * value_heads, &rng, 1);
    defer gpa.free(hgb);
    var x = try t.upload(hx);
    defer x.free();
    var ga = try t.upload(hga);
    defer ga.free();
    var gb = try t.upload(hgb);
    defer gb.free();
    var lanes = [_]Lane{try Lane.make(t, &rng, 1)};
    defer lanes[0].free();
    const hs = try gpa.alloc(f32, state_len);
    defer gpa.free(hs);
    try lanes[0].state.download(0, std.mem.sliceAsBytes(hs));
    var hwin: [taps * channels]f32 = undefined;
    try lanes[0].window.download(0, std.mem.sliceAsBytes(&hwin));

    // today's: the serial rows (A0, A1, B0, B1, B2) through the conv split and the recurrence
    var sx = try t.alloc(total * channels * 2);
    defer sx.free();
    try sx.upload(0, std.mem.sliceAsBytes(hx[0 .. kept * channels]));
    try sx.upload(kept * channels * 2, std.mem.sliceAsBytes(hx[ra * channels ..][0 .. rb * channels]));
    var sga = try t.alloc(total * value_heads * 2);
    defer sga.free();
    var sgb = try t.alloc(total * value_heads * 2);
    defer sgb.free();
    inline for (.{ .{ &sga, hga }, .{ &sgb, hgb } }) |p| {
        try p[0].upload(0, std.mem.sliceAsBytes(p[1][0 .. kept * value_heads]));
        try p[0].upload(kept * value_heads * 2, std.mem.sliceAsBytes(p[1][ra * value_heads ..][0 .. rb * value_heads]));
    }
    var swin = try t.upload(&hwin);
    defer swin.free();
    var sq = try t.alloc(total * kw * 4);
    defer sq.free();
    var sk = try t.alloc(total * kw * 4);
    defer sk.free();
    var sv = try t.alloc(total * vw * 4);
    defer sv.free();
    var sg = try t.alloc(total * value_heads * 4);
    defer sg.free();
    var sb = try t.alloc(total * value_heads * 4);
    defer sb.free();
    try t.on.tf_conv_split(.{
        .x = at(sx),
        .kind = t.kind(),
        .weight = at(layer.w),
        .state = at(swin),
        .qn = at(sq),
        .kn = at(sk),
        .v = at(sv),
        .channels = channels,
        .kernel = conv,
        .rows = total,
        .kw = kw,
        .vw = vw,
        .qw = at(layer.qw),
        .kw_w = at(layer.kw_w),
        .eps = eps,
        .norm = 128,
        .ga = at(sga),
        .gb = at(sgb),
        .a_log = at(layer.a_log),
        .dt_bias = at(layer.dt),
        .gate = at(sg),
        .beta = at(sb),
        .gcount = total * value_heads,
        .heads = value_heads,
    }, t.stream.handle);
    var sstate = try t.upload(hs);
    defer sstate.free();
    var sy = try t.alloc(total * vw * 4);
    defer sy.free();
    try o.gatedDelta(at(sq), at(sk), at(sv), at(sg), at(sb), at(sstate), at(sy), total, key_heads, value_heads, dk, dv, null);

    // the lanes': window A of four rows (two kept), then window B of three on the pending two
    var qn = try t.alloc((ra + rb) * kw * 4);
    defer qn.free();
    var y = try t.alloc((ra + rb) * vw * 4);
    defer y.free();
    var wa = try Words.make(t, &lanes, &.{ra}, &.{ 0, 0, -1, 0 });
    defer wa.free();
    try front(t, &wa, layer, x, ga, gb, qn, 0, 1, ra);
    try t.on.laneChain(at(qn), at(y), wa.args(), 1, 0, log_rows, 12, key_heads, value_heads, t.stream.handle);
    try t.on.laneRoll(wa.args(), 1, 1, channels, taps, t.stream.handle);
    // the keep: a roll by `kept` rows, the two rows pending
    var wk = try Words.make(t, &lanes, &.{ra}, &.{ 0, 0, kept, 0 });
    defer wk.free();
    try t.on.laneRoll(wk.args(), 1, 1, channels, taps, t.stream.handle);
    var xb = try t.alloc(rb * channels * 2);
    defer xb.free();
    try xb.upload(0, std.mem.sliceAsBytes(hx[ra * channels ..][0 .. rb * channels]));
    var gab = try t.alloc(rb * value_heads * 2);
    defer gab.free();
    var gbb = try t.alloc(rb * value_heads * 2);
    defer gbb.free();
    try gab.upload(0, std.mem.sliceAsBytes(hga[ra * value_heads ..][0 .. rb * value_heads]));
    try gbb.upload(0, std.mem.sliceAsBytes(hgb[ra * value_heads ..][0 .. rb * value_heads]));
    var wb = try Words.make(t, &lanes, &.{rb}, &.{ kept, 0, -1, 0 });
    defer wb.free();
    try front(t, &wb, layer, xb, gab, gbb, qn, 0, 1, rb);
    try t.on.laneChain(at(qn), at(y), wb.args(), 1, 0, log_rows, 12, key_heads, value_heads, t.stream.handle);
    try t.stream.synchronize();

    const want_y = try rig.download(f32, sy);
    defer gpa.free(want_y);
    const got_y = try rig.download(f32, y);
    defer gpa.free(got_y);
    try near(want_y[kept * vw ..], got_y[0 .. rb * vw], "y of the second window");
    const want_s = try rig.download(f32, sstate);
    defer gpa.free(want_s);
    const got_s = try rig.download(f32, lanes[0].spare);
    defer gpa.free(got_s);
    try near(want_s, got_s, "the end state");
}

fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "A/B: lanes against today's per-stream launches, 12 layers, 1 and 8 streams at 1, 4 and 8 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    const layers = 12;
    var rng: rig.Rng = .{ .state = 0xAB };
    var lyr = try Layer.make(t, &rng);
    defer lyr.free();
    inline for (.{ 1, 8 }) |n| {
        inline for (.{ 1, 4, 8 }) |r| {
            var lanes: [n]Lane = undefined;
            var made: usize = 0;
            defer for (lanes[0..made]) |*l| l.free();
            for (&lanes) |*l| {
                l.* = try Lane.make(t, &rng, layers);
                made += 1;
            }
            const rows = n * r;
            const hx = try randomBits(t, rows * channels, &rng, 2);
            defer gpa.free(hx);
            const hga = try randomBits(t, rows * value_heads, &rng, 1);
            defer gpa.free(hga);
            var x = try t.upload(hx);
            defer x.free();
            var ga = try t.upload(hga);
            defer ga.free();
            var qn = try t.alloc(rows * kw * 4);
            defer qn.free();
            var y = try t.alloc(rows * vw * 4);
            defer y.free();
            var kn = try t.alloc(rows * kw * 4);
            defer kn.free();
            var vn = try t.alloc(rows * vw * 4);
            defer vn.free();
            var gt = try t.alloc(rows * value_heads * 4);
            defer gt.free();
            var bt = try t.alloc(rows * value_heads * 4);
            defer bt.free();
            var counts: [n]u32 = @splat(r);
            var ctl: [4 * n]i32 = undefined;
            for (0..n) |i| ctl[4 * i ..][0..4].* = .{ 0, 0, 0, 0 };
            var w = try Words.make(t, &lanes, &counts, &ctl);
            defer w.free();
            const reps = 20;
            var ours_ms: f64 = 0;
            var lanes_ms: f64 = 0;
            for (0..2) |warm| {
                const began = nowNs();
                for (0..reps) |_| {
                    for (0..layers) |li| {
                        for (0..n) |s| {
                            const sx = at(x) + s * r * channels * 2;
                            try t.on.tf_conv_split(.{
                                .x = sx,
                                .kind = t.kind(),
                                .weight = at(lyr.w),
                                .state = at(lanes[s].window),
                                .qn = at(qn) + s * r * kw * 4,
                                .kn = at(kn) + s * r * kw * 4,
                                .v = at(vn) + s * r * vw * 4,
                                .channels = channels,
                                .kernel = conv,
                                .rows = r,
                                .kw = kw,
                                .vw = vw,
                                .qw = at(lyr.qw),
                                .kw_w = at(lyr.kw_w),
                                .eps = eps,
                                .norm = 128,
                                .ga = at(ga) + s * r * value_heads * 2,
                                .gb = at(ga) + s * r * value_heads * 2,
                                .a_log = at(lyr.a_log),
                                .dt_bias = at(lyr.dt),
                                .gate = at(gt) + s * r * value_heads * 4,
                                .beta = at(bt) + s * r * value_heads * 4,
                                .gcount = r * value_heads,
                                .heads = value_heads,
                            }, t.stream.handle);
                            try o.gatedDelta(at(qn) + s * r * kw * 4, at(kn) + s * r * kw * 4, at(vn) + s * r * vw * 4, at(gt) + s * r * value_heads * 4, at(bt) + s * r * value_heads * 4, at(lanes[s].state) + li * state_len * 4, at(y) + s * r * vw * 4, r, key_heads, value_heads, dk, dv, null);
                        }
                    }
                }
                try t.stream.synchronize();
                const mid = nowNs();
                for (0..reps) |_| {
                    for (0..layers) |li| {
                        try front(t, &w, lyr, x, ga, ga, qn, li, n, r);
                        try t.on.laneChain(at(qn), at(y), w.args(), n, li, log_rows, 12, key_heads, value_heads, t.stream.handle);
                    }
                    try t.on.laneRoll(w.args(), n, layers, channels, taps, t.stream.handle);
                }
                try t.stream.synchronize();
                const end = nowNs();
                if (warm == 1) {
                    ours_ms = @as(f64, @floatFromInt(mid - began)) / 1e6 / reps;
                    lanes_ms = @as(f64, @floatFromInt(end - mid)) / 1e6 / reps;
                }
            }
            std.debug.print("gdn A/B {d} streams x {d} rows, {d} layers: ours (conv+plan chain, no replay) {d:.3} ms, lanes (front+chain+roll) {d:.3} ms\n", .{ n, r, layers, ours_ms, lanes_ms });
        }
    }
}
