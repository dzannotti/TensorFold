//! A verify window's DeltaNet without a snapshot a row: the round runs from a plan, the keep replays it.
const std = @import("std");
const hip = @import("hip");

const pops = hip.plan_ops;
const cache = @import("cache.zig");
const config = @import("../config.zig");
const forward = @import("forward.zig");
const model = @import("model.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;

/// Rows one window holds at most: the pending token and its drafts.
pub const window_rows = cache.max_window;

/// Whether this model's shapes are the plan kernels': key dim 128, eight value rows a wave, gated heads.
pub fn supported(d: model.Dims, layers: usize) bool {
    if (d.key_dim != 128) return false;
    if (d.value_dim % 8 != 0 or d.key_heads < 1 or d.value_heads % d.key_heads != 0) return false;
    if (d.conv < 1 or d.conv > 8) return false;
    return layers > 0;
}

/// The plan's words in four-byte units over `slots` slots (a round's streams and an empty last one): each slot's first
/// row and rows, a kept-row array a slot (each stream's keep its own), the descriptor pointers and a layer's snapshots.
pub const Layout = struct {
    slots: usize,
    first: usize,
    count: usize,
    keep: usize,
    ptr: usize,
    snaps: usize,
    words: usize,

    pub fn of(layers: usize, slots: usize) Layout {
        var l: Layout = undefined;
        l.slots = slots;
        l.first = 0;
        l.count = slots;
        l.keep = 2 * slots;
        l.ptr = std.mem.alignForward(usize, l.keep + slots * slots, 4); // eight-byte pointers start on an even word
        l.snaps = l.ptr + 2 * slots;
        l.words = l.snaps + 4 * layers;
        return l;
    }

    /// Words a slot's descriptor takes: cached positions, the last kept final row, two addresses a layer.
    pub fn desc(layers: usize) usize {
        return 2 * (2 + 2 * layers);
    }

    /// Where the descriptors start (eight-byte aligned).
    fn start(layers: usize, slots: usize) usize {
        return std.mem.alignForward(usize, of(layers, slots).words, 2);
    }

    /// Every word of the plan: its control words, then a descriptor a slot.
    pub fn total(layers: usize, slots: usize) usize {
        return start(layers, slots) + slots * desc(layers);
    }
};

fn logOn() bool {
    return std.c.getenv("TF_GDN_LOG") != null;
}

/// A round's window plan over every stream in it: the words, a descriptor a stream, and the rows each keep reads again.
pub const Plan = struct {
    r: *const hip.Runtime,
    layers: usize,
    /// Streams a round holds at most, and the slots the words lay out: one more, the empty slot.
    streams: usize,
    /// Rows the buffers hold at most: a window a stream.
    room: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    channels: usize,
    kept_stride: usize,
    samples_stride: usize,
    /// The words and descriptors, and the device copy a round and a keep read.
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// A layer's k, v, gate and beta over the round's rows, section-major: what a keep replays from.
    kept: hip.DeviceBuffer,
    /// A layer's conv input rows over the round, section-major: where a keep takes its conv tail.
    samples: hip.DeviceBuffer,
    /// The last round's rows, streams, and each stream's cache slot and rows.
    rows: usize = 0,
    n: usize = 0,
    ids: [forward.max_segs]u32 = undefined,
    counts: [forward.max_segs]usize = undefined,

    pub fn init(d: *const hip.Runtime, dims: model.Dims, streams: usize) !Plan {
        if (!supported(dims, dims.layers)) return error.UnsupportedPlan;
        const n = @min(@max(streams, 1), forward.max_segs);
        var host = try hip.HostBuffer.alloc(d, 4 * Layout.total(dims.layers, n + 1));
        errdefer host.free();
        var dev = try hip.DeviceBuffer.alloc(d, host.bytes.len);
        errdefer dev.free();
        const room = n * window_rows;
        var kept = try hip.DeviceBuffer.alloc(d, 4 * room * width(dims) * dims.layers);
        errdefer kept.free();
        var samples = try hip.DeviceBuffer.alloc(d, 4 * room * dims.convWidth() * dims.layers);
        errdefer samples.free();
        @memset(host.bytes, 0);
        return .{
            .r = d,
            .layers = dims.layers,
            .streams = n,
            .room = room,
            .key_heads = dims.key_heads,
            .value_heads = dims.value_heads,
            .key_dim = dims.key_dim,
            .value_dim = dims.value_dim,
            .conv = dims.conv,
            .channels = dims.convWidth(),
            .kept_stride = 4 * room * width(dims),
            .samples_stride = 4 * room * dims.convWidth(),
            .host = host,
            .dev = dev,
            .kept = kept,
            .samples = samples,
        };
    }

    pub fn deinit(p: *Plan) void {
        p.samples.free();
        p.kept.free();
        p.dev.free();
        p.host.free();
    }

    /// A row's kept values: k, v, gate and beta.
    fn width(dims: model.Dims) usize {
        return dims.key_heads * dims.key_dim + dims.value_heads * dims.value_dim + 2 * dims.value_heads;
    }

    /// Device bytes the plan for `streams` streams takes, whether or not one is made: the admission counts it either way.
    pub fn bytes(dims: model.Dims, streams: usize) usize {
        const n = @min(@max(streams, 1), forward.max_segs);
        const room = n * window_rows;
        return 4 * Layout.total(dims.layers, n + 1) + 4 * room * width(dims) * dims.layers +
            4 * room * dims.convWidth() * dims.layers;
    }

    pub fn layout(p: *const Plan) Layout {
        return Layout.of(p.layers, p.streams + 1);
    }

    /// A slot's descriptor words: [0] cached positions, [1] the last kept final row (0: none), then a layer's pair.
    fn descriptor(p: *Plan, slot: usize) []u64 {
        const at = Layout.start(p.layers, p.streams + 1) + slot * Layout.desc(p.layers);
        return p.host.slice(u64)[at / 2 ..][0 .. Layout.desc(p.layers) / 2];
    }

    fn descAt(p: *const Plan, slot: usize) u64 {
        return p.dev.base() + 4 * (Layout.start(p.layers, p.streams + 1) + slot * Layout.desc(p.layers));
    }

    /// Linear layer `layer`'s section-major k, v, gate and beta over the round's row `at`.
    pub fn keptOf(p: *const Plan, layer: usize, at: usize) struct { k: u64, v: u64, gate: u64, beta: u64 } {
        const kw = p.key_heads * p.key_dim;
        const vw = p.value_heads * p.value_dim;
        const rows = p.rows;
        const base = p.kept.base() + layer * p.kept_stride;
        const k = base + 4 * at * kw;
        const v = base + 4 * rows * kw + 4 * at * vw;
        const gate = base + 4 * rows * (kw + vw) + 4 * at * p.value_heads;
        return .{ .k = k, .v = v, .gate = gate, .beta = gate + 4 * rows * p.value_heads };
    }

    /// The conv samples of layer `layer` (the round's `channels` a row).
    pub fn samplesOf(p: *const Plan, layer: usize) u64 {
        return p.samples.base() + layer * p.samples_stride;
    }

    /// Slot `slot`'s kept-row array, the one its keep reads.
    fn keepWords(p: *const Plan, slot: usize) u64 {
        return p.dev.base() + 4 * (p.layout().keep + slot * p.layout().slots);
    }

    /// The round's words and a descriptor a stream of `segs`, on the host: the copy goes up with `send`.
    pub fn fill(p: *Plan, segs: []const forward.Seg) !void {
        if (segs.len == 0 or segs.len > p.streams) return error.TooManyStreams;
        var rows: usize = 0;
        for (segs) |g| rows += g.tokens.len;
        if (rows > p.room) return error.TooManyRows;
        for (segs) |g| if (g.tokens.len == 0 or g.tokens.len > window_rows) return error.TooManyRows;
        const l = p.layout();
        const w = p.host.slice(u32);
        const ptr = p.host.slice(u64);
        const c0 = segs[0].c;
        for (0..p.layers) |i| {
            // a sparse layer keeps its conv samples and replay inputs zero: the kernels skip it
            ptr[l.snaps / 2 + 2 * i] = if (c0.linear[i]) p.samplesOf(i) else 0;
            ptr[l.snaps / 2 + 2 * i + 1] = if (c0.linear[i]) p.kept.base() + i * p.kept_stride else 0;
        }
        // every keep array starts with nothing kept: a stream's keep writes its own slot of its own array
        @memset(w[l.keep..][0 .. l.slots * l.slots], std.math.maxInt(u32));
        var at: usize = 0;
        for (segs, 0..) |g, s| {
            const desc = p.descriptor(s);
            for (0..p.layers) |i| {
                desc[2 + 2 * i ..][0..2].* = if (g.c.linear[i]) .{ g.c.linearAt(i).conv, g.c.linearAt(i).state } else .{ 0, 0 };
            }
            // a round keeps no final row: a keep takes the conv tail and the state alone
            desc[0] = 0;
            desc[1] = 0;
            w[l.first + s] = @intCast(at);
            w[l.count + s] = @intCast(g.tokens.len);
            ptr[l.ptr / 2 + s] = p.descAt(s);
            p.ids[s] = g.c.id;
            p.counts[s] = g.tokens.len;
            at += g.tokens.len;
        }
        // the empty slot runs nothing and keeps nothing: the kernels stop at its zero row count
        const empty = segs.len;
        @memset(p.descriptor(empty), 0);
        w[l.first + empty] = @intCast(at);
        w[l.count + empty] = 0;
        ptr[l.ptr / 2 + empty] = p.descAt(empty);
        p.rows = rows;
        p.n = segs.len;
        if (logOn()) std.debug.print("gdn fill n={d} rows={d} ids={any} counts={any}\n", .{ p.n, p.rows, p.ids[0..p.n], p.counts[0..p.n] });
    }

    /// The words up to the device: a graph holds the copy, so its every replay sends the round's own words.
    pub fn send(p: *Plan, s: hip.abi.Stream) !void {
        try hip.raw.upload(p.dev, 0, p.host.bytes[0 .. 4 * Layout.total(p.layers, p.streams + 1)], s);
    }

    /// Cache slot `id`'s first `n` rows of the last round kept: the conv tail from their samples, the state replayed.
    pub fn keep(p: *Plan, o: Ops, id: u32, n: usize) !void {
        const s = for (p.ids[0..p.n], 0..) |x, i| {
            if (x == id) break i;
        } else return error.KeepNotInRound;
        if (logOn()) std.debug.print("gdn keep id={d} slot={d} n={d} count={d}\n", .{ id, s, n, p.counts[s] });
        // a window of one row left the state and its conv tail as the round wrote them
        if (p.counts[s] <= 1 or n == 0) return;
        const l = p.layout();
        const words = p.host.slice(u32)[l.keep + s * l.slots ..][0..l.slots];
        words[s] = @bitCast(@as(i32, @intCast(n)));
        try hip.raw.upload(p.dev, 4 * (l.keep + s * l.slots), std.mem.sliceAsBytes(words), o.stream);
        try pops.keep(o, p.ref(), .{
            .keep = p.keepWords(s),
            .hidden = 0,
            .hidden_words = 0,
            .channels = p.channels,
            .taps = p.conv - 1,
            .layers = p.layers,
        });
        try pops.gdnReplay(o, p.keepWords(s), p.ref(), p.key_heads, p.value_heads, p.key_dim, p.value_dim, p.layers);
    }

    /// The words as launches take them: every stream's rows from its first, and the empty slot.
    pub fn ref(p: *const Plan) pops.Plan {
        const l = p.layout();
        const at = p.dev.base();
        return .{
            .args = .{
                .first = at + 4 * l.first,
                .count = at + 4 * l.count,
                .desc = at + 4 * l.ptr,
                .snaps = at + 4 * l.snaps,
            },
            .rows = p.rows,
            .slots = p.n + 1,
        };
    }
};

/// A round's rows through the linear attention from its plan: one conv split and delta rule over every stream, the norm after.
pub fn linear(
    o: Ops,
    m: *const model.Model,
    l: model.Linear,
    index: usize,
    qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    rows: usize,
    p: *const Plan,
) !Tensor {
    if (rows != p.rows) return error.CachesNotPlanned;
    const d = m.dims;
    const ch = d.convWidth();
    const kw = d.key_heads * d.key_dim;
    const vw = d.value_heads * d.value_dim;
    const y = try o.arena.of(f32, rows * vw);
    const qn = try o.arena.of(f32, rows * kw);
    const kept = p.keptOf(index, 0);
    try pops.convSplit(
        o,
        qkv,
        l.conv,
        qn,
        kept.k,
        kept.v,
        p.ref(),
        index,
        ch,
        d.conv,
        kw,
        vw,
        .{ .q = m.qk.q, .k = m.qk.k, .eps = m.qk.eps },
        .{
            .a = a,
            .b = b,
            .a_log = l.a_log,
            .dt_bias = l.dt_bias,
            .gate = kept.gate,
            .beta = kept.beta,
            .count = rows * d.value_heads,
            .heads = d.value_heads,
        },
    );
    try pops.gatedDelta(o, qn, kept.k, kept.v, kept.gate, kept.beta, y, p.ref(), index, d.key_heads, d.value_heads, d.key_dim, d.value_dim);
    const out = try take(o, m.act, rows * vw);
    try o.gnormOut(y, l.norm, z, out, rows * d.value_heads, d.value_dim, d.eps, d.gate);
    return o.project(out, l.out, rows, l.out.partial);
}

test "the words keep a keep array a slot and put the pointers on even words" {
    for ([_]usize{ 2, 4, 6, 65 }) |slots| {
        const l = Layout.of(48, slots);
        try std.testing.expect(l.ptr % 2 == 0);
        try std.testing.expect(l.ptr >= l.keep + slots * slots);
        try std.testing.expect(l.snaps == l.ptr + 2 * slots);
        try std.testing.expect(l.words == l.snaps + 4 * 48);
        try std.testing.expect(Layout.start(48, slots) % 2 == 0);
        try std.testing.expectEqual(Layout.start(48, slots) + slots * Layout.desc(48), Layout.total(48, slots));
    }
    try std.testing.expectEqual(@as(usize, 2 * (2 + 2 * 48)), Layout.desc(48));
}

test "the plan path is the default: Flash Next's shapes take it, the shapes its kernels cannot take stay per-row" {
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const d = try model.Dims.of(&c, 1);
    try std.testing.expectEqual(@as(usize, 16), d.key_heads);
    try std.testing.expectEqual(@as(usize, 48), d.value_heads);
    try std.testing.expectEqual(@as(usize, 4), d.conv);
    // the shapes the plan kernels are: no variable asks for it
    try std.testing.expect(supported(d, 48));
    // every other shape keeps the per-row snapshots
    var narrow = d;
    narrow.key_dim = 64;
    try std.testing.expect(!supported(narrow, 48));
    var odd = d;
    odd.value_dim = 100;
    try std.testing.expect(!supported(odd, 48));
    odd = d;
    odd.value_heads = 50;
    try std.testing.expect(!supported(odd, 48));
    var long = d;
    long.conv = 9;
    try std.testing.expect(!supported(long, 48));
    try std.testing.expect(!supported(d, 0));
}

test "one layer's kept sections sit one after another over the window's rows" {
    const kw = 16 * 128;
    const vw = 48 * 128;
    const vh = 48;
    var p: Plan = undefined;
    p.rows = 4;
    p.room = window_rows;
    p.streams = 1;
    p.key_heads = 16;
    p.key_dim = 128;
    p.value_heads = vh;
    p.value_dim = 128;
    p.kept_stride = 4 * p.room * (kw + vw + 2 * vh);
    p.kept = .{ .r = undefined, .ptr = @ptrFromInt(0x1000), .len = 0 };
    const first = p.keptOf(0, 0);
    const second = p.keptOf(0, 1);
    try std.testing.expectEqual(first.k + 4 * kw, second.k);
    try std.testing.expectEqual(first.k, first.v - 4 * p.rows * kw);
    try std.testing.expectEqual(first.v, first.gate - 4 * p.rows * vw);
    try std.testing.expectEqual(first.beta, first.gate + 4 * p.rows * vh);
}
