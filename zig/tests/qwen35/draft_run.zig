//! `draft <model dir> <drafter dir> [--bits 0|4] [--prompt TEXT]`: the DFlash2 drafter on the target's real taps.
//! The target writes a greedy reply; the drafter, fed the target's tapped rows as the round loop would, drafts after
//! each reply token, and the drafts are scored against the reply. Then batched, tree and graph holds against singles,
//! and a hold's cost at 1, 2, 4 and 8 streams.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const tokenizer = @import("tokenizer");

const Draft = qwen35.draft.Draft;
const fwd = qwen35.forward;

const default_prompt =
    \\<|im_start|>user
    \\Write a Python function that returns the n-th Fibonacci number iteratively, then a short test.<|im_end|>
    \\<|im_start|>assistant
    \\<think>
    \\
    \\</think>
    \\
    \\
;

/// The tapped layers' residual rows into `feats` (rows, taps, hidden), as a target's `features` would hand them.
const Taps = struct {
    e: *qwen35.engine.Engine,
    taps: []const u32,
    feats: hip.DeviceBuffer,
    hidden: usize,

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, rows: usize) anyerror!void {
        const t: *Taps = @ptrCast(@alignCast(ctx));
        for (t.taps, 0..) |tap, j| if (tap == index) {
            const width = t.hidden * 2;
            for (0..rows) |r| try hip.raw.copy(t.feats, (r * t.taps.len + j) * width, x.ptr + r * width, width, t.e.stream.handle);
        };
    }
};

fn arg(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |a, i| if (std.mem.eql(u8, a, name) and i + 1 < args.len) return args[i + 1];
    return null;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    const bits: u8 = if (arg(args, "--bits")) |b| try std.fmt.parseInt(u8, b, 10) else 4;
    const reply = 96;
    const cap = 8;
    const e = try qwen35.engine.Engine.open(gpa, io, args[0], .{ .capacity = 1024, .prompt_rows = 1024, .streams = 2, .batch_rows = 64, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    const d = try Draft.init(gpa, io, e, args[1], bits, cap);
    defer d.deinit();
    std.debug.print("RESULT drafter loaded in {d:.1} s ({d}-bit), hold {d:.2} ms wall / {d:.2} ms GPU, 16-lane tree {d:.2} ms, step_ms {d:.3}\n", .{ @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9, bits, d.step_ms * @as(f64, @floatFromInt(d.c.block - 1)), d.gpu_ms, d.tree_ms, d.step_ms });
    const dr = d.drafter();
    if (arg(args, "--bench") != null) return bench(e, d);

    // the target's greedy reply, a token a pass
    const tk_path = try std.fs.path.join(gpa, &.{ args[0], "tokenizer.json" });
    defer gpa.free(tk_path);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var tok = try tokenizer.loadTokenizer(io, arena.allocator(), tk_path);
    defer tok.deinit();
    const prompt = try tok.encode(gpa, arg(args, "--prompt") orelse default_prompt);
    defer gpa.free(prompt);
    var seq: std.ArrayList(u32) = .empty;
    defer seq.deinit(gpa);
    try seq.appendSlice(gpa, prompt);
    {
        var caches = try e.newCaches(prompt.len + reply + 8);
        defer {
            e.drain();
            caches.deinit(gpa);
        }
        var from: usize = 0;
        for (0..reply) |_| {
            const next = try e.prefill(&caches, seq.items, from, null, .{ .sampling = null, .position = seq.items.len }, null);
            from = seq.items.len;
            try seq.append(gpa, next);
        }
    }
    const text = try tok.decode(gpa, seq.items[prompt.len..], false);
    defer gpa.free(text);
    std.debug.print("reply: {s}\n", .{text});

    // every row's taps, from one pass over the prompt and reply
    const c = d.c;
    const len = seq.items.len;
    const row_bytes = c.n_taps * c.hidden * 2;
    var taps: Taps = .{ .e = e, .taps = d.c.taps[0..c.n_taps], .feats = try hip.DeviceBuffer.alloc(&e.driver, len * row_bytes), .hidden = c.hidden };
    defer taps.feats.free();
    {
        var caches = try e.newCaches(len + 8);
        defer {
            e.drain();
            caches.deinit(gpa);
        }
        e.prompts.reset();
        const ids = e.ids.slice(u32)[0..len];
        @memcpy(ids, seq.items);
        try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * len), e.stream);
        _ = try fwd.span(e.ops(&e.prompts), e.model(), &caches, e.ids_dev.base(), len, 0, .{ .ctx = &taps, .layer = Taps.layer });
        try e.stream.synchronize();
    }
    const features: lanes.backend.Features = .{ .buffer = taps.feats.base(), .rows = @intCast(len), .row_bytes = @intCast(row_bytes), .space = .device, .dtype = .bf16 };

    // the round loop's order: the prompt's rows but the last, then after each token its row, a hold, the drafts scored
    var s: lanes.Stream = undefined;
    try dr.open(&s);
    defer dr.release(&s);
    const P = prompt.len;
    var head = features;
    head.rows = @intCast(P - 1);
    try dr.absorb(&.{.{ .stream = &s, .features = head, .start = 0, .follow = &.{}, .rows = null }});
    var accepted: usize = 0;
    var rounds: usize = 0;
    var by_depth: [16]usize = @splat(0);
    var t = P - 1; // the pending token's position: rows before it absorbed
    var singles: [cap][16]u32 = undefined;
    var anchors: [cap]usize = undefined;
    while (t + c.block <= len) : (t += 1) {
        if (t > P - 1) { // the row before the pending token, as a verify that kept it hands it over
            var kept: [1]u32 = .{@intCast(t - 1)};
            try dr.absorb(&.{.{ .stream = &s, .features = features, .start = t - 1, .follow = &.{}, .rows = &kept }});
        }
        try dr.hold(&.{.{ .stream = &s, .pending = seq.items[t], .position = t + 1, .depth = @intCast(c.block - 1) }});
        var drafts: [16]u32 = undefined;
        try dr.held(&.{&s}, &.{drafts[0 .. c.block - 1]});
        var n: usize = 0;
        while (n < c.block - 1 and drafts[n] == seq.items[t + 1 + n]) n += 1;
        for (0..n) |k| by_depth[k] += 1;
        accepted += n;
        rounds += 1;
        if (rounds <= cap) {
            anchors[rounds - 1] = t;
            singles[rounds - 1] = drafts;
        }
    }
    std.debug.print("RESULT acceptance: {d} rounds, {d:.2} drafts accepted a round (of {d}); by depth", .{ rounds, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(rounds)), c.block - 1 });
    for (by_depth[0 .. c.block - 1]) |k| std.debug.print(" {d:.2}", .{@as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(rounds))});
    std.debug.print("\n", .{});
    if (rounds < cap) return error.ReplyTooShort;

    // the first `cap` anchors held together, each its own stream at its own position: the singles' drafts again
    var ss: [cap]lanes.Stream = undefined;
    var ptrs: [cap]*lanes.Stream = undefined;
    var holds: [cap]lanes.drafter.Hold = undefined;
    var outs: [cap][16]u32 = undefined;
    var slices: [cap][]u32 = undefined;
    for (0..cap) |i| {
        ptrs[i] = &ss[i];
        try dr.open(&ss[i]);
        var f = features; // rows [0, anchor)
        f.rows = @intCast(anchors[i]);
        try dr.absorb(&.{.{ .stream = &ss[i], .features = f, .start = 0, .follow = &.{}, .rows = null }});
        holds[i] = .{ .stream = &ss[i], .pending = seq.items[anchors[i]], .position = anchors[i] + 1, .depth = @intCast(c.block - 1) };
        slices[i] = outs[i][0 .. c.block - 1];
    }
    defer for (ptrs) |p| dr.release(p);
    var same: usize = 0;
    for (0..3) |_| { // the first captures the batch's graph, the next replay it
        try dr.hold(&holds);
        try dr.held(&ptrs, &slices);
        for (0..cap) |i| {
            if (std.mem.eql(u32, slices[i], singles[i][0 .. c.block - 1])) same += 1;
        }
    }
    std.debug.print("RESULT batched holds: {d} of {d} streams draft as they do alone\n", .{ same, 3 * cap });

    // a tree whose rank-0 path is the chain: that path's lanes hold the chain's drafts
    var parents: [16]i32 = undefined;
    var ranks: [16]u8 = undefined;
    var depths: [16]u8 = undefined;
    var n_lanes: usize = 0;
    for (0..c.block - 1) |k| {
        const par: i32 = if (k == 0) -1 else @intCast(2 * (k - 1));
        for (0..2) |r| {
            parents[n_lanes] = par;
            ranks[n_lanes] = @intCast(r);
            depths[n_lanes] = @intCast(k);
            n_lanes += 1;
        }
    }
    const shape: lanes.shape.Shape = .{ .parents = parents[0..n_lanes], .ranks = ranks[0..n_lanes], .depths = depths[0..n_lanes], .expected = 0 };
    try dr.hold(&.{.{ .stream = &ss[0], .pending = seq.items[anchors[0]], .position = anchors[0] + 1, .depth = 0, .lanes = &shape }});
    var tree: [16]u32 = undefined;
    try dr.held(&.{&ss[0]}, &.{tree[0..n_lanes]});
    var on_path: usize = 0;
    var k: usize = 0;
    while (k * 2 < n_lanes) : (k += 1) {
        if (tree[k * 2] == singles[0][k]) on_path += 1;
    }
    std.debug.print("RESULT tree: rank-0 lanes match the chain at {d} of {d} depths; rank-1 lanes differ from them: {}\n", .{ on_path, k, tree[1] != tree[0] });
    if (same != 3 * cap or on_path != k) return error.DraftsDiffer;

    // a hold's cost by streams (wall: hold and held; GPU: events)
    for ([_]usize{ 1, 2, 4, 8 }) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..8) |_| {
            const a = std.Io.Clock.awake.now(io);
            try dr.hold(holds[0..n]);
            try dr.held(ptrs[0..n], slices[0..n]);
            best = @min(best, @as(f64, @floatFromInt(a.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6);
        }
        std.debug.print("RESULT hold of {d} stream{s}: {d:.2} ms ({d:.3} ms a draft a stream)\n", .{ n, if (n == 1) "" else "s", best, best / @as(f64, @floatFromInt((c.block - 1) * n)) });
    }
}

/// The drafter's widest products (gate, the draft head) through its row kernel at each row count: GB/s of weights read.
fn bench(e: *qwen35.engine.Engine, d: *Draft) !void {
    var arena = try hip.Arena.init(&e.driver, 256 << 20);
    defer arena.deinit();
    var x = try hip.DeviceBuffer.alloc(&e.driver, 64 * 17408 * 2);
    defer x.free();
    try x.fill8(0x3c);
    var start = try hip.Event.init(&e.driver, true);
    defer start.deinit();
    var stop = try hip.Event.init(&e.driver, true);
    defer stop.deinit();
    const ps = [_]struct { name: []const u8, p: qwen35.draft.Linear }{ .{ .name = "gate", .p = d.w.layers[0].gate }, .{ .name = "qkv", .p = d.w.layers[0].qkv }, .{ .name = "down", .p = d.w.layers[0].down }, .{ .name = "head", .p = .{ .q4 = d.head } } };
    for (ps) |pp| for ([_]usize{ 1, 8, 16, 32, 64 }) |m| {
        const o: hip.ops.Ops = .{ .l = &e.lib, .bf16 = true, .stream = e.stream.handle, .arena = &arena };
        var best: f32 = std.math.inf(f32);
        for (0..6) |_| {
            arena.reset();
            try start.record(e.stream);
            _ = try qwen35.draft.proj(o, .{ .ptr = x.base(), .kind = .bf16 }, pp.p, m);
            try stop.record(e.stream);
            try e.stream.synchronize();
            best = @min(best, try hip.Event.elapsedMs(start, stop));
        }
        const n = pp.p.rows();
        const k: usize = switch (pp.p) {
            .q4 => |q| q.k,
            .bf16 => |b| b.k,
        };
        const per: f64 = if (pp.p == .q4) 0.5625 else 2;
        const bytes = @as(f64, @floatFromInt(n * k)) * per;
        std.debug.print("RESULT {s} ({d}x{d}) m {d}: {d:.3} ms, {d:.0} GB/s\n", .{ pp.name, n, k, m, best, bytes / @as(f64, best) / 1e6 });
    };
}
