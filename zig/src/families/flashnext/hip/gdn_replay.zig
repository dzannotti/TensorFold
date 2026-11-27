//! A round's DeltaNet over every stream's window in two launches a layer (the CUDA family's fn_gdn_lanes_*), KEEP_LATE:
//! the chain writes the window's end state to a spare, a whole keep swaps it in, a partial keep leaves its rows in a log.
const std = @import("std");
const hip = @import("hip");

const cache = @import("cache.zig");
const config = @import("../config.zig");
const forward = @import("forward.zig");
const model = @import("model.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;
const launches = hip.launches;

/// Rows one window holds at most: the pending token and its drafts.
pub const window_rows = cache.max_window;

const slots = forward.max_segs;

/// Whether this model's shapes are the lane kernels': key dim 128, gated heads, a conv of at most eight taps.
pub fn supported(d: model.Dims, layers: usize) bool {
    if (d.key_dim != 128 or d.value_dim != 128) return false;
    if (d.key_heads < 1 or d.value_heads % d.key_heads != 0) return false;
    if (d.conv < 1 or d.conv > 8) return false;
    return layers > 0;
}

/// TF_GDN_LATE=0: a partly kept window's rows are applied to the state at once (the intermediate step before KEEP_LATE).
fn lateOn() bool {
    const v = std.c.getenv("TF_GDN_LATE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The round's words in four-byte units: first and rows a slot, [pending, log first, kept, 0] a slot, then the slots'
/// tables (five addresses a layer) behind an address a slot.
const Layout = struct {
    first: usize,
    count: usize,
    ctl: usize,
    ptrs: usize,
    tables: usize,
    words: usize,

    fn of(layers: usize) Layout {
        const first = 0;
        const count = slots;
        const ctl = 2 * slots;
        const ptrs = std.mem.alignForward(usize, ctl + 4 * slots, 2);
        const tables = ptrs + 2 * slots;
        return .{ .first = first, .count = count, .ctl = ctl, .ptrs = ptrs, .tables = tables, .words = tables + 2 * slots * 5 * layers };
    }

    fn table(l: Layout, layers: usize, slot: usize) usize {
        return l.tables + 2 * 5 * layers * slot;
    }
};

const Words = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,

    fn make(d: *const hip.Runtime, bytes: usize) !Words {
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        const dev = try hip.DeviceBuffer.alloc(d, bytes);
        @memset(host.bytes, 0);
        return .{ .host = host, .dev = dev };
    }

    fn free(w: *Words) void {
        w.dev.free();
        w.host.free();
    }
};

pub const Lanes = struct {
    r: *const hip.Runtime,
    layers: usize,
    channels: usize,
    taps: usize,
    key_heads: usize,
    value_heads: usize,
    /// The round's words, and the words a flush of pending rows runs from.
    main: Words,
    side: Words,
    late: bool,
    cs: [slots]*cache.Caches = undefined,
    n: usize = 0,
    ids: [slots]u32 = @splat(0),
    rows: [slots]u32 = @splat(0),
    open: [slots]bool = @splat(false),
    batch: bool = false,
    dirty: bool = false,

    pub fn init(d: *const hip.Runtime, dims: model.Dims) !Lanes {
        if (!supported(dims, dims.layers)) return error.UnsupportedPlan;
        const size = 4 * Layout.of(dims.layers).words;
        var main = try Words.make(d, size);
        errdefer main.free();
        const side = try Words.make(d, size);
        return .{ .r = d, .layers = dims.layers, .channels = dims.convWidth(), .taps = dims.conv - 1, .key_heads = dims.key_heads, .value_heads = dims.value_heads, .main = main, .side = side, .late = lateOn() };
    }

    pub fn deinit(p: *Lanes) void {
        p.side.free();
        p.main.free();
    }

    /// Device bytes of the round's words.
    pub fn bytes(dims: model.Dims) usize {
        return 4 * Layout.of(dims.layers).words;
    }

    fn layout(p: *const Lanes) Layout {
        return Layout.of(p.layers);
    }

    pub fn args(p: *const Lanes) launches.LaneArgs {
        return p.argsOf(&p.main);
    }

    fn argsOf(p: *const Lanes, w: *const Words) launches.LaneArgs {
        const l = p.layout();
        const at = w.dev.base();
        return .{ .first = at + 4 * l.first, .count = at + 4 * l.count, .ctl = at + 4 * l.ctl, .tab = at + 4 * l.ptrs };
    }

    fn fillSlot(p: *Lanes, w: *Words, i: usize, c: *const cache.Caches, inplace: bool) void {
        const l = p.layout();
        const ptr = w.host.slice(u64);
        const t = ptr[l.table(p.layers, i) / 2 ..];
        for (0..p.layers) |li| {
            if (!c.linear[li]) {
                t[5 * li ..][0..5].* = @splat(0);
                continue;
            }
            const live = c.linearAt(li);
            t[5 * li ..][0..5].* = .{ live.conv, live.state, if (inplace) live.state else c.spareState(li), c.logAt(li), c.samplesAt(li) };
        }
        ptr[l.ptrs / 2 + i] = w.dev.base() + 4 * l.table(p.layers, i);
    }

    /// The round's windows: their rows, each stream's pending log rows and the buffers its chain reads and writes.
    pub fn fill(p: *Lanes, segs: []const forward.Seg) !void {
        const l = p.layout();
        const w = p.main.host.slice(u32);
        var at: u32 = 0;
        for (segs, 0..) |g, i| {
            const n: u32 = @intCast(g.tokens.len);
            if (n == 0 or n > window_rows) return error.TooManyRows;
            if (g.c.late.pending + n > cache.log_rows) return error.LogFull;
            w[l.first + i] = at;
            w[l.count + i] = n;
            w[l.ctl + 4 * i ..][0..4].* = .{ g.c.late.pending, g.c.late.first, std.math.maxInt(u32), 0 };
            p.fillSlot(&p.main, i, g.c, false);
            p.cs[i] = g.c;
            p.ids[i] = g.c.id;
            p.rows[i] = n;
            p.open[i] = true;
            at += n;
        }
        p.n = segs.len;
    }

    /// The words up to the device: a graph holds the copy, so its every replay sends the round's own words.
    pub fn send(p: *Lanes, s: hip.abi.Stream) !void {
        try hip.raw.upload(p.main.dev, 0, p.main.host.bytes, s);
    }

    /// Stream `c`'s window kept `kept` rows: its conv window moves on at the next flush, its state swaps or its rows pend.
    pub fn keep(p: *Lanes, o: Ops, c: *cache.Caches, kept: usize) !void {
        const slot = for (0..p.n) |i| {
            if (p.open[i] and p.ids[i] == c.id) break i;
        } else return error.NoWindow;
        if (kept == 0 or kept > p.rows[slot]) return error.BadKeep;
        p.open[slot] = false;
        p.main.host.slice(u32)[p.layout().ctl + 4 * slot + 2] = @intCast(kept);
        c.late.kept(kept, p.rows[slot]);
        p.dirty = true;
        if (!p.batch) try p.flushKeeps(o);
    }

    /// Every kept window's conv window rolled on, in one launch; without KEEP_LATE the pending rows applied at once.
    pub fn flushKeeps(p: *Lanes, o: Ops) !void {
        if (!p.dirty) return;
        p.dirty = false;
        const l = p.layout();
        const w = p.main.host.slice(u32);
        try p.send(o.stream);
        try o.l.laneRoll(p.args(), p.n, p.layers, p.channels, p.taps, o.stream);
        try hip.runtime.check(p.r.api.hipStreamSynchronize(o.stream));
        for (0..p.n) |i| w[l.ctl + 4 * i + 2] = std.math.maxInt(u32);
        if (p.late) return;
        var held: [slots]*cache.Caches = undefined;
        var count: usize = 0;
        for (p.cs[0..p.n], p.open[0..p.n]) |c, open| {
            if (open or c.late.pending == 0) continue;
            held[count] = c;
            count += 1;
        }
        if (count > 0) try p.flush(o, held[0..count]);
    }

    /// Slots of `caches` with their pending rows applied to their live states in place; the live buffer then the main one.
    pub fn flush(p: *Lanes, o: Ops, caches: []const *cache.Caches) !void {
        const l = p.layout();
        const w = p.side.host.slice(u32);
        var any = false;
        for (caches, 0..) |c, i| {
            w[l.first + i] = 0;
            w[l.count + i] = 0;
            w[l.ctl + 4 * i ..][0..4].* = .{ c.late.pending, c.late.first, std.math.maxInt(u32), 0 };
            p.fillSlot(&p.side, i, c, true);
            any = any or c.late.pending > 0;
        }
        if (any) {
            try hip.raw.upload(p.side.dev, 0, p.side.host.bytes, o.stream);
            for (0..p.layers) |li| {
                if (!caches[0].linear[li]) continue;
                try o.l.laneChain(0, 0, p.argsOf(&p.side), caches.len, li, cache.log_rows, std.math.maxInt(u31), p.key_heads, p.value_heads, o.stream);
            }
        }
        for (caches) |c| {
            c.late.pending = 0;
            c.late.first = 0;
            if (!c.late.flip) continue;
            for (c.linear, 0..) |is_linear, li| {
                if (!is_linear) continue;
                try hip.runtime.check(p.r.api.hipMemcpyDtoDAsync(
                    @ptrFromInt(c.spareState(li)),
                    @ptrFromInt(c.linearAt(li).state),
                    c.layout.state_bytes,
                    o.stream,
                ));
            }
            c.late.flip = false;
        }
        try hip.runtime.check(p.r.api.hipStreamSynchronize(o.stream));
    }
};

/// A round's windows through the linear attention: one conv front and one chain for every stream, the norm after.
pub fn linear(
    o: Ops,
    m: *const model.Model,
    l: model.Linear,
    segs: []const forward.Seg,
    index: usize,
    qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    rows: usize,
    s: *Lanes,
) !Tensor {
    const d = m.dims;
    const kw = d.key_heads * d.key_dim;
    const vw = d.value_heads * d.value_dim;
    const y = try o.arena.of(f32, rows * vw);
    const qn = try o.arena.of(f32, rows * kw);
    var widest: usize = 0;
    for (segs) |g| widest = @max(widest, g.tokens.len);
    try o.l.laneFront(.{
        .x = qkv.ptr,
        .kind = @backingInt(m.act),
        .weight = l.conv,
        .qw = m.qk.q,
        .kw_w = m.qk.k,
        .eps = m.qk.eps,
        .qn = qn,
        .ga = a.ptr,
        .gb = b.ptr,
        .a_log = l.a_log,
        .dt_bias = l.dt_bias,
        .channels = @intCast(d.convWidth()),
        .kernel = @intCast(d.conv),
        .kw = @intCast(kw),
        .vw = @intCast(vw),
        .heads = @intCast(d.value_heads),
        .layer = @intCast(index),
        .log_rows = cache.log_rows,
        .lanes = s.args(),
    }, segs.len, widest, o.stream);
    try o.l.laneChain(qn, y, s.args(), segs.len, index, cache.log_rows, cache.late_store, d.key_heads, d.value_heads, o.stream);
    const out = try take(o, m.act, rows * vw);
    try o.gnormOut(y, l.norm, z, out, rows * d.value_heads, d.value_dim, d.eps, d.gate);
    return o.project(out, l.out, rows, l.out.partial);
}

test "the lane kernels take Flash Next's shapes and refuse the rest" {
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const d = try model.Dims.of(&c, 1);
    try std.testing.expect(supported(d, 48));
    var narrow = d;
    narrow.key_dim = 64;
    try std.testing.expect(!supported(narrow, 48));
    var long = d;
    long.conv = 9;
    try std.testing.expect(!supported(long, 48));
    try std.testing.expect(!supported(d, 0));
}

test "the words keep every slot's table on an even word" {
    const l = Layout.of(48);
    try std.testing.expect(l.ptrs % 2 == 0 and l.tables % 2 == 0);
    try std.testing.expectEqual(l.tables + 2 * 5 * 48, l.table(48, 1));
}
