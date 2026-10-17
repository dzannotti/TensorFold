//! A lane round on the engine: its device plan, its forward (replayed, captured or eager), its draws and its keep.

const std = @import("std");
const hip = @import("hip");
const state = @import("../forward/state.zig");
const win = @import("../forward/window.zig");
const fwd = @import("../forward/forward.zig");
const plan = @import("../forward/plan.zig");
const view = @import("../model/view.zig");
const draw = @import("hip_serve").draw;
const round_graphs = @import("round_graphs.zig");
const Engine = @import("engine.zig").Engine;

/// What a round does with its shape's graph: run eagerly, replay it, or capture it (every rank does the same).
pub const Pick = enum(u32) { eager, replay, capture };

/// What a verify hands back: the round's final rows, padding included.
pub const Verified = struct { hidden: hip.ops.Tensor };

/// One stream's rows of a round: tokens (the pending one, then drafts) from slot `pos` over its caches; for a tree,
/// each row's parent row (row 0: -1), a row then keyed at its depth and seeing only its ancestors.
pub const Rows = struct { caches: *state.Caches, pos: usize, tokens: []const u32, parents: ?[]const i32 = null };

const Chosen = struct { pick: Pick, entry: ?*round_graphs.Entry, shape: plan.Shape };

/// The slots a keep names and the plan they index.
const Last = struct { plan: hip.plan_ops.Plan, keep_at: usize, hidden: u64 };

/// Keep lists in flight: a flush's pinned words stay until its copy has run, which a later round's sync guarantees.
const keep_ring = 4;

/// TF_ROUND_PROFILE: GPU timestamps of each round's verify, keep and drafter hold (events against one base), summed.
pub const Prof = struct {
    pub const v0 = 0;
    pub const v1 = 1;
    pub const k0 = 2;
    pub const k1 = 3;
    pub const h0 = 4;
    pub const h1 = 5;
    base: hip.Event,
    ev: [6]hip.Event,
    /// Whether the keep and hold events were recorded since the last read.
    kept: bool = false,
    held: bool = false,
    pv0: f64 = -1,
    pv1: f64 = -1,
    /// verify, keep, hold, verify end to hold start, hold end to next verify start, round (verify start to the next's)
    sum: [6]f64 = @splat(0),
    n: [6]u64 = @splat(0),

    fn at(p: *Prof, i: usize) f64 {
        return hip.Event.elapsedMs(p.base, p.ev[i]) catch -1;
    }

    fn add(p: *Prof, i: usize, ms: f64) void {
        if (ms < 0 or ms > 1000) return;
        p.sum[i] += ms;
        p.n[i] += 1;
    }

    /// After the round's draw (its stream drained, the hold before it synced): the previous round's pieces and this verify.
    fn read(p: *Prof) void {
        const v0t = p.at(v0);
        const v1t = p.at(v1);
        p.add(0, v1t - v0t);
        if (p.kept) p.add(1, p.at(k1) - p.at(k0));
        if (p.held) {
            const a = p.at(h0);
            const b = p.at(h1);
            p.add(2, b - a);
            if (p.pv1 >= 0) p.add(3, a - p.pv1);
            p.add(4, v0t - b);
        }
        if (p.pv0 >= 0) p.add(5, v0t - p.pv0);
        p.pv0 = v0t;
        p.pv1 = v1t;
        p.kept = false;
        p.held = false;
    }

    fn print(p: *const Prof) void {
        const names = [_][]const u8{ "verify", "keep", "hold", "verify->hold", "hold->verify", "round" };
        for (names, p.sum, p.n) |name, x, n| if (n > 0) std.debug.print("round profile: {s} {d:.2} ms ({d})\n", .{ name, x / @as(f64, @floatFromInt(n)), n });
    }
};

pub const State = struct {
    prof: ?*Prof = null,
    buffer: plan.Buffer,
    /// The caches the padding rows run in.
    scratch: state.Caches,
    /// Each linear layer's snapshots of the round in flight.
    snaps: [][2]u64,
    wins: []plan.Window,
    /// The rows each slot keeps at the next flush (negative: not listed), and the pinned words they go up in.
    staging: []i32,
    keeps: hip.HostBuffer,
    ring: usize = 0,
    /// For checks: a layer-by-layer trace of the forward (graphs off), and a least bucket of rows.
    trace: ?fwd.Trace = null,
    pad_to: usize = 0,
    chosen: ?Chosen = null,
    last: ?Last = null,
    /// The layout of the plan in flight, for the copy a round starts with.
    layout: plan.Layout = undefined,
    /// Recorded before each keep's launches (ordering only).
    before_keep: hip.Event,
    votes: ?hip.Votes = null,

    pub fn init(e: *Engine) !State {
        const m = e.model();
        if (!win.supported(e.ops(&e.rounds), m)) return error.RoundsUnsupported;
        const rows = e.o.batch_rows;
        var buffer = try plan.Buffer.init(&e.driver, rows, m.spec.n_layers, e.pool.count);
        errdefer buffer.deinit();
        var scratch = try state.Caches.initFull(e.gpa, &e.pool, m, rows);
        errdefer scratch.deinit(e.gpa);
        const snaps = try e.gpa.alloc([2]u64, m.spec.n_layers);
        errdefer e.gpa.free(snaps);
        const wins = try e.gpa.alloc(plan.Window, rows);
        errdefer e.gpa.free(wins);
        const staging = try e.gpa.alloc(i32, rows + 1);
        errdefer e.gpa.free(staging);
        @memset(staging, -1);
        var votes: ?hip.Votes = if (e.o.world > 1) try hip.Votes.init(&e.driver, e.o.world) else null;
        errdefer if (votes) |*v| v.deinit();
        var keeps = try hip.HostBuffer.alloc(&e.driver, keep_ring * 4 * (rows + 1));
        errdefer keeps.free();
        return .{
            .buffer = buffer,
            .scratch = scratch,
            .snaps = snaps,
            .wins = wins,
            .staging = staging,
            .keeps = keeps,
            .votes = votes,
            .before_keep = try hip.Event.init(&e.driver, false),
            .prof = if (std.c.getenv("TF_ROUND_PROFILE") != null) blk: {
                const p = try e.gpa.create(Prof);
                p.* = .{ .base = try hip.Event.init(&e.driver, true), .ev = undefined };
                for (&p.ev) |*x| x.* = try hip.Event.init(&e.driver, true);
                try p.base.record(e.stream);
                break :blk p;
            } else null,
        };
    }

    pub fn deinit(s: *State, e: *Engine) void {
        if (s.prof) |p| {
            p.print();
            e.gpa.destroy(p);
        }
        s.before_keep.deinit();
        if (s.votes) |*v| v.deinit();
        s.keeps.free();
        e.gpa.free(s.staging);
        e.gpa.free(s.wins);
        e.gpa.free(s.snaps);
        s.scratch.deinit(e.gpa);
        s.buffer.deinit();
    }
};

/// The shape of a round of `rows`: its bucket of rows, its slots and the keys its walk covers.
pub fn shapeOf(e: *Engine, rows: []const Rows) !plan.Shape {
    var total: usize = 0;
    var visible: usize = 0;
    for (rows) |r| {
        total += r.tokens.len;
        visible = @max(visible, r.pos + r.tokens.len);
    }
    if (rows.len == 0 or total == 0 or total > e.o.batch_rows) return error.WindowTooWide;
    const cap = std.mem.alignForward(usize, e.o.capacity, plan.min_span);
    const bucket = plan.bucketOf(@max(total, e.round.pad_to), e.o.batch_rows);
    const slots = plan.bucketOf(rows.len + 1, e.o.batch_rows + 1);
    return .{ .rows = @intCast(bucket), .slots = @intCast(slots), .span = @intCast(@min(plan.spanOf(visible), cap)) };
}

/// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys).
pub fn choose(e: *Engine, rows: []const Rows, forced: ?Pick) !Pick {
    const st = &e.round;
    const shape = try shapeOf(e, rows);
    st.chosen = .{ .pick = .eager, .entry = null, .shape = shape };
    if (!e.o.graphs) {
        if ((forced orelse .eager) != .eager) return error.GraphsDisagree;
        return .eager;
    }
    const entry = try e.graphs.find(shape);
    const mine: Pick = switch (entry.state) {
        .failed => .eager,
        .ready => .replay,
        .seen => .capture,
    };
    const pick = forced orelse mine;
    if (pick == .replay and entry.state != .ready) return error.GraphsDisagree;
    st.chosen = .{ .pick = pick, .entry = entry, .shape = shape };
    return pick;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Every window in one forward: `out` the token of every row per `reqs`; snapshots live until the next round's `keep`.
pub fn verify(e: *Engine, rows: []const Rows, reqs: []const draw.Request, out: []u32) !Verified {
    const st = &e.round;
    const m = e.model();
    const began = nowNs();
    if (st.chosen == null) _ = try choose(e, rows, null);
    const c = st.chosen.?;
    defer st.chosen = null;
    var total: usize = 0;
    for (rows, 0..) |r, i| {
        if (r.pos + r.tokens.len > r.caches.covered()) return error.ContextFull;
        st.wins[i] = .{ .desc = r.caches.desc.base(), .pos = r.pos, .tokens = r.tokens, .parents = r.parents, .ring = r.caches.tap };
        total += r.tokens.len;
    }
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, c.shape.rows, st.snaps);
    const l = plan.Layout.of(c.shape.rows, c.shape.slots, m.spec.n_layers);
    try st.buffer.fill(l, c.shape, st.wins[0..rows.len], st.scratch.desc.base(), st.snaps);
    st.layout = l;
    const p: hip.plan_ops.Plan = .{ .args = st.buffer.args(l), .rows = c.shape.rows, .slots = c.shape.slots };
    const r: win.Round = .{ .plan = p, .tokens = st.buffer.dev.base() + 4 * l.tokens, .span = c.shape.span, .inputs = st.snaps, .taps = e.taps[0..e.n_taps], .rings = st.buffer.dev.base() + 4 * l.rings, .ring_rows = @import("engine.zig").tap_rows };
    if (st.prof) |pf| try pf.ev[Prof.v0].record(e.stream);
    const done = try run(e, c, r);
    if (st.prof) |pf| try pf.ev[Prof.v1].record(e.stream);
    e.graphs.rounds += 1;
    e.graphs.submit_ns += nowNs() - began;
    try e.drawRows(o, done.y, total, reqs, out, m.tp == null);
    if (st.prof) |pf| {
        try e.stream.synchronize();
        pf.read();
    }
    st.last = .{ .plan = p, .keep_at = 4 * l.keep, .hidden = done.hidden.ptr };
    return .{ .hidden = done.hidden };
}

/// The forward and its logits projection, and the greedy draw of every row when this rank holds the whole head.
fn body(e: *Engine, r: win.Round) !round_graphs.Out {
    // the plan goes up first, inside the graph
    try e.round.buffer.send(e.stream.handle, e.round.layout);
    return forwardOn(e.ops(&e.rounds), e.model(), r, e.round.trace, e.drawer.argmaxAt());
}

/// `body` after the plan's copy, on `o`: on the counting stream it measures what a round takes.
pub fn forwardOn(ops: hip.ops.Ops, m: *const view.Model, r: win.Round, trace: ?fwd.Trace, argmax_at: u64) !round_graphs.Out {
    var o = ops;
    // a round's rows keep their decode kernels however many share it
    o.window = true;
    const hidden = try win.forward(o, m, r, trace);
    const y = try o.project(hidden, m.head, r.plan.rows, false);
    if (m.tp == null) try o.argmaxRows(y, r.plan.rows, m.head.n, argmax_at);
    return .{ .hidden = hidden, .y = y, .used = o.arena.used };
}

/// One round's forward as `choose` picked: replayed from the shape's graph, captured, or eager.
fn run(e: *Engine, c: Chosen, r: win.Round) !round_graphs.Out {
    const entry = c.entry orelse return body(e, r);
    switch (c.pick) {
        .eager => return body(e, r),
        .replay => {
            e.rounds.used = entry.out.used;
            try entry.exec.?.launchOn(e.stream);
            e.graphs.replayed += 1;
            return entry.out;
        },
        .capture => return capture(e, entry, r),
    }
}

/// Records the round, then every rank votes: all kept replay it, one decline runs it eagerly, one failure stops all.
fn capture(e: *Engine, entry: *round_graphs.Entry, r: win.Round) !round_graphs.Out {
    const at = e.rounds.mark();
    const began = nowNs();
    const m = &e.bridge.model;
    const got = hip.vote.record(round_graphs.Out, e.gpa, e.stream, &m.tp, e.o.policy.failsCapture(e.o.rank), Body{
        .e = e,
        .r = r,
    }, Body.run);
    e.graphs.capture_ns += nowNs() - began;
    if (got.split) |split| e.graphs.hold(entry, split, got.out);
    const all = if (e.round.votes) |*v| try v.agree(e.comm, got.vote, e.stream) else got.vote;
    if (all == .kept) {
        try entry.exec.?.launchOn(e.stream);
        return entry.out;
    }
    e.graphs.revoke(entry);
    e.rounds.release(at);
    if (all == .declined) return body(e, r);
    if (e.o.world == 1) return error.OutOfDeviceMemory;
    // no rank runs another collective once one could not take a round
    e.stopped = true;
    return error.GroupStopped;
}

/// A round's forward as the recorder runs it.
const Body = struct {
    e: *Engine,
    r: win.Round,

    fn run(b: Body) anyerror!round_graphs.Out {
        return body(b.e, b.r);
    }
};

/// Marks slot `slot` of the last verify to keep, at the next `flush`, the path to its row `last` - 1 (a chain's first
/// `last` rows, a tree's ancestors of that row).
pub fn mark(e: *Engine, slot: usize, last: usize) void {
    e.round.staging[slot] = @intCast(last);
}

/// Keeps every marked slot of the last verify in one launch: linear states after the kept rows, last kept final rows.
pub fn flush(e: *Engine) !void {
    const st = &e.round;
    const last = st.last orelse return;
    const slots = last.plan.slots;
    var any = false;
    for (st.staging[0..slots]) |k| any = any or k >= 0;
    if (!any) return;
    const per = e.o.batch_rows + 1;
    const from = 4 * st.ring * per;
    const words = st.keeps.slice(i32)[st.ring * per ..][0..slots];
    st.ring = (st.ring + 1) % keep_ring;
    @memcpy(words, st.staging[0..slots]);
    @memset(st.staging[0..slots], -1);
    try st.before_keep.record(e.stream);
    if (st.prof) |pf| try pf.ev[Prof.k0].record(e.stream);
    try st.buffer.dev.uploadAsync(last.keep_at, st.keeps.view(from, from + 4 * slots), e.stream);
    try win.keep(e.ops(&e.rounds), e.model(), last.plan, st.buffer.dev.base() + last.keep_at, last.hidden);
    if (st.prof) |pf| {
        try pf.ev[Prof.k1].record(e.stream);
        pf.kept = true;
    }
    e.keep_last = true;
}
