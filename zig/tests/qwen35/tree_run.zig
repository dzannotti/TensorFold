//! `tree <model dir> [cases] [seed]`: tree windows and kept tree paths against each path decoded one row at a time.
//!
//! Every case verifies a few streams' random trees (and a chain) in one padded round, then checks that each row's
//! residuals after every layer, its final row and its tapped residuals equal, byte for byte, the serial steps along
//! its ancestor path; then it keeps a random root-to-node path of each tree and checks the final row it leaves and
//! the next two serial steps against a stream that decoded that path serially. A last pass replays captured graphs.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;
const Caches = qwen35.state.Caches;

/// The layers whose residuals the case taps, as a drafter would.
const taps = [_]u32{ 1, 30, 62 };

/// Each forward layer's residual rows and the final rows of one round: (layers + 1, rows, width) bytes.
const Capture = struct {
    e: *Engine,
    bytes: []u8,
    rows: usize,

    fn width(c: *const Capture) usize {
        return c.e.model().spec.hidden * c.e.model().act.size();
    }

    fn row(c: *const Capture, slot: usize, r: usize) []const u8 {
        const w = c.width();
        return c.bytes[(slot * c.rows + r) * w ..][0..w];
    }

    fn put(c: *Capture, slot: usize, x: u64) !void {
        try c.e.stream.synchronize();
        const w = c.width();
        try hip.runtime.check(c.e.driver.api.hipMemcpyDtoH(c.bytes[slot * c.rows * w ..].ptr, @ptrFromInt(x), c.rows * w));
    }

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, _: usize) anyerror!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        try c.put(index, x.ptr);
    }
};

/// A stream's caches with a final-row buffer and a tap ring, its prompt prefilled.
const Lane = struct {
    caches: Caches,
    hidden: hip.DeviceBuffer,
    ring: hip.DeviceBuffer,
    len: usize,

    fn open(e: *Engine, prompt: []const u32) !Lane {
        var caches = try e.newCaches(prompt.len + 64);
        errdefer caches.deinit(e.gpa);
        var hidden = try hip.DeviceBuffer.alloc(&e.driver, e.model().spec.hidden * e.model().act.size());
        errdefer hidden.free();
        var ring = try hip.DeviceBuffer.alloc(&e.driver, e.tapBytes());
        errdefer ring.free();
        try caches.setHidden(hidden.base());
        caches.tap = ring.base();
        _ = try e.prefill(&caches, prompt, 0, hidden, .{ .sampling = null, .position = prompt.len }, null);
        return .{ .caches = caches, .hidden = hidden, .ring = ring, .len = prompt.len };
    }

    fn close(l: *Lane, e: *Engine) void {
        e.drain();
        l.caches.deinit(e.gpa);
        l.hidden.free();
        l.ring.free();
    }

    /// `bytes` of the ring row at position `pos`.
    fn tapRow(l: *const Lane, e: *Engine, pos: usize, out: []u8) !void {
        try e.stream.synchronize();
        const at = (pos % qwen35.engine.tap_rows) * out.len;
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(out.ptr, @ptrFromInt(l.ring.base() + at), out.len));
    }

    fn final(l: *const Lane, e: *Engine, out: []u8) !void {
        try e.stream.synchronize();
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(out.ptr, @ptrFromInt(l.hidden.base()), out.len));
    }
};

/// One round over `lanes`' windows (greedy draws at each row's depth); `cap` traces it; `keeps` (last row + 1 a lane,
/// 0: none) are flushed after it.
fn round(e: *Engine, lanes: []const *Lane, tokens: []const []const u32, parents: []const ?[]const i32, cap: ?*Capture, keeps: []const usize, pad: usize) !void {
    var rows: [8]Engine.Rows = undefined;
    var reqs: [128]qwen35.draw.Request = undefined;
    var out: [128]u32 = undefined;
    var total: usize = 0;
    for (lanes, tokens, parents, 0..) |l, t, p, j| {
        rows[j] = .{ .caches = &l.caches, .pos = l.len, .tokens = t, .parents = p };
        var anc: [16]u32 = undefined;
        if (p) |ps| try qwen35.plan.ancestors(ps, &anc);
        for (0..t.len) |i| reqs[total + i] = .{ .sampling = null, .position = l.len + 1 + if (p != null) @popCount(anc[i]) - 1 else i };
        total += t.len;
    }
    e.round.pad_to = pad;
    defer e.round.pad_to = 0;
    if (cap) |c| {
        c.rows = total;
        e.round.trace = .{ .ctx = c, .layer = Capture.layer };
    }
    defer e.round.trace = null;
    const done = try e.verify(rows[0..lanes.len], reqs[0..total], out[0..total]);
    if (cap) |c| try c.put(e.model().spec.n_layers, done.hidden.ptr);
    for (keeps, 0..) |k, j| if (k > 0) e.keep(j, k);
    try e.flush();
    try e.stream.synchronize();
}

/// One serial step of `token` on `lane`, traced into `cap` (one row).
fn step(e: *Engine, lane: *Lane, token: u32, cap: ?*Capture) !void {
    try round(e, &.{lane}, &.{&.{token}}, &.{null}, cap, &.{1}, 0);
    lane.len += 1;
}

/// The path from the root to `node`, ascending.
fn pathTo(parents: []const i32, node: usize, out: *[16]u32) []const u32 {
    var n: usize = 0;
    var at: i32 = @intCast(node);
    while (at >= 0) : (at = parents[@intCast(at)]) {
        out[n] = @intCast(at);
        n += 1;
    }
    std.mem.reverse(u32, out[0..n]);
    return out[0..n];
}

const Fail = error{Mismatch};

fn same(what: []const u8, a: []const u8, b: []const u8, case: usize, stream: usize, row: usize) Fail!void {
    const at = std.mem.indexOfDiff(u8, a, b) orelse return;
    std.debug.print("FAIL case {d} stream {d} row {d}: {s} differs from serial at byte {d}\n", .{ case, stream, row, what, at });
    return error.Mismatch;
}

/// A stream of a case: its prompt, tokens and parents (null: a chain).
const Window = struct { prompt: []const u32, tokens: []u32, parents: ?[]i32 };

/// Random trees (and a chain) over 1 to 3 streams; their rows, keeps and next steps against serial.
fn runCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, rand: std.Random, case: usize) !void {
    const m = e.model();
    const layers = m.spec.n_layers;
    const streams = rand.intRangeAtMost(usize, 1, 3);
    var wins: [3]Window = undefined;
    var token_buf: [3][16]u32 = undefined;
    var parent_buf: [3][16]i32 = undefined;
    var total: usize = 0;
    for (wins[0..streams], 0..) |*w, j| {
        const n = rand.intRangeAtMost(usize, 2, 16);
        const plen = rand.intRangeAtMost(usize, 20, 90);
        const from = rand.uintLessThan(usize, ids.len - plen);
        w.prompt = ids[from..][0..plen];
        w.tokens = token_buf[j][0..n];
        for (w.tokens) |*t| t.* = rand.intRangeLessThan(u32, 1000, 60000);
        // the last stream of a case of several is a chain, the others trees of random shape
        if (j > 0 and j + 1 == streams) {
            w.parents = null;
        } else {
            w.parents = parent_buf[j][0..n];
            w.parents.?[0] = -1;
            for (1..n) |r| w.parents.?[r] = @intCast(rand.uintLessThan(usize, r));
        }
        total += n;
    }
    const width = m.spec.hidden * m.act.size();
    const tap_width = taps.len * width;
    var tree_cap: Capture = .{ .e = e, .bytes = try gpa.alloc(u8, (layers + 1) * 64 * width), .rows = 0 };
    defer gpa.free(tree_cap.bytes);
    var one_cap: Capture = .{ .e = e, .bytes = try gpa.alloc(u8, (layers + 1) * width), .rows = 1 };
    defer gpa.free(one_cap.bytes);
    const a = try gpa.alloc(u8, tap_width);
    defer gpa.free(a);
    const b = try gpa.alloc(u8, tap_width);
    defer gpa.free(b);

    var tree_lanes: [3]Lane = undefined;
    var opened: usize = 0;
    defer for (tree_lanes[0..opened]) |*l| l.close(e);
    for (wins[0..streams], 0..) |w, j| {
        tree_lanes[j] = try Lane.open(e, w.prompt);
        opened += 1;
    }
    var lane_ptrs: [3]*Lane = undefined;
    var toks: [3][]const u32 = undefined;
    var pars: [3]?[]const i32 = undefined;
    for (0..streams) |j| {
        lane_ptrs[j] = &tree_lanes[j];
        toks[j] = wins[j].tokens;
        pars[j] = wins[j].parents;
    }
    // a padded round of every stream's window, kept later
    // each window keeps a random root-to-node path in the keep after its round (a keep follows its own verify)
    var keeps: [3]usize = undefined;
    var kept: [3][16]u32 = undefined;
    var kept_paths: [3][]const u32 = undefined;
    for (wins[0..streams], 0..) |w, j| {
        const node = rand.uintLessThan(usize, w.tokens.len);
        const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        kept_paths[j] = pathTo(w.parents orelse chain[0..w.tokens.len], node, &kept[j]);
        keeps[j] = node + 1;
    }
    const pad = if (rand.boolean()) @as(usize, 32) else 0;
    try round(e, lane_ptrs[0..streams], toks[0..streams], pars[0..streams], &tree_cap, keeps[0..streams], pad);
    for (0..streams) |j| tree_lanes[j].len += kept_paths[j].len;
    var row0: usize = 0;
    for (wins[0..streams], 0..) |w, j| {
        const n = w.tokens.len;
        const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        const ps: []const i32 = w.parents orelse chain[0..n];
        // the leaves' paths cover every row
        var parent_of: [16]bool = @splat(false);
        for (ps[1..]) |p| parent_of[@intCast(p)] = true;
        for (0..n) |leaf| {
            if (parent_of[leaf]) continue;
            var buf: [16]u32 = undefined;
            const path = pathTo(ps, leaf, &buf);
            var serial = try Lane.open(e, w.prompt);
            defer serial.close(e);
            for (path, 0..) |r, d| {
                try step(e, &serial, w.tokens[r], &one_cap);
                for (0..layers + 1) |slot| try same(if (slot == layers) "final row" else "layer residual", tree_cap.row(slot, row0 + r), one_cap.row(slot, 0), case, j, r);
                try tree_lanes[j].tapRow(e, w.prompt.len + r, a);
                try serial.tapRow(e, w.prompt.len + d, b);
            // the tree lane's ring is as its round left it: the keep moves keys and values, not taps
                try same("tapped residuals", a, b, case, j, r);
            }
        }
        row0 += n;
    }
    // the kept path's final row, then two more steps, against the path decoded serially
    for (wins[0..streams], 0..) |w, j| {
        var serial = try Lane.open(e, w.prompt);
        defer serial.close(e);
        for (kept_paths[j]) |r| try step(e, &serial, w.tokens[r], null);
        try tree_lanes[j].final(e, a[0..width]);
        try serial.final(e, b[0..width]);
        try same("kept final row", a[0..width], b[0..width], case, j, kept_paths[j][kept_paths[j].len - 1]);
        var next_cap: Capture = .{ .e = e, .bytes = try gpa.alloc(u8, (layers + 1) * width), .rows = 1 };
        defer gpa.free(next_cap.bytes);
        for (0..2) |k| {
            const t: u32 = @intCast(2000 + 31 * k + j);
            try step(e, &serial, t, &one_cap);
            try step(e, &tree_lanes[j], t, &next_cap);
            for (0..layers + 1) |slot| try same(if (slot == layers) "next step's final row" else "next step's residual", next_cap.row(slot, 0), one_cap.row(slot, 0), case, j, k);
        }
    }
}

/// Trees through captured graphs: two rounds of one shape (a capture, then a replay) against serial final rows.
fn graphCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, rand: std.Random, case: usize) !void {
    const m = e.model();
    const width = m.spec.hidden * m.act.size();
    const n = 8;
    const prompt = ids[case..][0..40];
    const got = try gpa.alloc(u8, n * width);
    defer gpa.free(got);
    const want = try gpa.alloc(u8, width);
    defer gpa.free(want);
    for (0..3) |pass| {
        var tokens: [n]u32 = undefined;
        var parents: [n]i32 = undefined;
        for (&tokens) |*t| t.* = rand.intRangeLessThan(u32, 1000, 60000);
        parents[0] = -1;
        for (1..n) |r| parents[r] = @intCast(rand.uintLessThan(usize, r));
        var lane = try Lane.open(e, prompt);
        defer lane.close(e);
        var rows = [_]Engine.Rows{.{ .caches = &lane.caches, .pos = lane.len, .tokens = &tokens, .parents = &parents }};
        var anc: [16]u32 = undefined;
        try qwen35.plan.ancestors(&parents, &anc);
        var reqs: [n]qwen35.draw.Request = undefined;
        var out: [n]u32 = undefined;
        for (&reqs, 0..) |*q, i| q.* = .{ .sampling = null, .position = lane.len + @popCount(anc[i]) };
        const done = try e.verify(&rows, &reqs, &out);
        try e.stream.synchronize();
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(got.ptr, @ptrFromInt(done.hidden.ptr), n * width));
        for (0..n) |leaf| {
            var buf: [16]u32 = undefined;
            const path = pathTo(&parents, leaf, &buf);
            var serial = try Lane.open(e, prompt);
            defer serial.close(e);
            for (path) |r| try step(e, &serial, tokens[r], null);
            try serial.final(e, want);
            try same(if (pass < 2) "first rounds' final row" else "replayed round's final row", got[leaf * width ..][0..width], want, pass, 0, leaf);
        }
    }
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 1) return error.MissingArgument;
    const cases = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 8;
    const seed = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 1;
    const ids = try @import("ids_file.zig").synthetic(gpa, 512);
    defer gpa.free(ids);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = 256, .prompt_rows = 256, .streams = 8, .batch_rows = 64, .graphs = true, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    try e.setTaps(&taps);
    var prng = std.Random.DefaultPrng.init(seed);
    // the trace waits on the stream layer by layer, which a capture cannot
    const graphs = e.o.graphs;
    e.o.graphs = false;
    for (0..cases) |c| try runCase(gpa, e, ids, prng.random(), c);
    std.debug.print("PASS {d} cases: every tree row, tap and kept path equals its path one row at a time\n", .{cases});
    e.o.graphs = graphs;
    if (!graphs) return std.debug.print("SKIP graphs: off by the policy\n", .{});
    try graphCase(gpa, e, ids, prng.random(), 0);
    std.debug.print("PASS graphs: a captured and replayed tree round's rows equal their paths one row at a time\n", .{});
}
