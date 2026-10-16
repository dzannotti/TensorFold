//! The DFlash2 drafter on HIP behind core/lanes/drafter.zig: a ring of tapped keys a stream, every stream's block a hold.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("../engine/engine.zig").Engine;
const config = @import("config.zig");
const weights = @import("weights.zig");
const graph_cache = hip.graph_cache;
const dft = hip.launches.dft;

pub const Config = config.Config;
const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
pub const Linear = weights.Linear;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const Shape = lanes.shape.Shape;

/// Rows one absorb pass projects at once (a longer absorb runs several).
const absorb_rows = 128;
/// Most lanes a tree holds.
const max_lanes = 64;
/// The splits block attention walks a ring in (the merge joins them).
const splits = 16;
/// The draft head's vocabulary when the target's is Qwen3.8's: whole tiles of the common ids and the specials, as Metal's
/// runtime_head.zig; the head reads 98592 of 248320 rows. Any other vocabulary drafts over the whole head.
const spans = [_][2]u32{ .{ 0, 98304 }, .{ 248032, 248320 } };
const stock_vocab = 248320;

/// Acceptance by depth until a stream has its own (#836's, measured on the reference's chat text).
const prior = [_]f64{ 0.75, 0.65, 0.58, 0.52, 0.47, 0.43, 0.4 };

/// A stream's context: its rings over positions [begin, end), and what its last hold left for `held`.
const Ctx = struct {
    ring: hip.DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    /// The last hold: its row in the batch, pending token, drafts asked for, and its tree (null: a chain).
    slot: ?usize = null,
    pending: u32 = 0,
    depth: usize = 0,
    parents: [max_lanes]i32 = undefined,
    ranks: [max_lanes]u8 = undefined,
    depths: [max_lanes]u8 = undefined,
    n_lanes: ?usize = null,
    /// The drafts chosen from it (`held` chooses once, after the hold's results reach the host).
    held: [max_lanes]u32 = undefined,
    held_n: ?usize = null,

    fn reset(c: *Ctx) void {
        c.* = .{ .ring = c.ring };
    }
};

/// Byte offsets of the staging area (pinned, mirrored on the device): uploads first, the hold's results after.
const Layout = struct { desc: usize, ids: usize, rows: usize, top_ids: usize, top_vals: usize, hp: usize, total: usize };

/// The hold's launches by streams held at once.
const Key = struct { streams: u32 };

pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *Engine,
    c: Config,
    w: weights.Weights,
    /// The target's head over the draft vocabulary, its rows copied and tiled.
    head: weights.Q4,
    head_buf: ?hip.DeviceBuffer,
    restricted: bool,
    /// Streams one hold takes at most, and rows an absorb's table holds.
    cap: usize,
    table_rows: usize,
    arena: hip.Arena,
    stage: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// top-k counts a hold row: 0 for each block's pending row.
    ks: hip.DeviceBuffer,
    /// The staging areas a queued copy may still read: written again only after a sync.
    absorb_busy: bool = false,
    hold_busy: bool = false,
    /// The last hold's results reached the host.
    synced: bool = true,
    held_rows: usize = 0,
    hp_dev: u64 = 0,
    graphs: graph_cache.Cache(Key, u64),
    use_graphs: bool,
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    /// Contexts between streams, allocated and zeroed once (`reserve`).
    pool: std.ArrayList(*Ctx) = .empty,
    /// One more draft's cost: a hold timed at load over its block's drafts.
    step_ms: f64 = 1.0,
    /// The hold's GPU time at load (events) and the hold with `held` of a 16-lane tree (wall), for the logs.
    gpu_ms: f64 = 0,
    tree_ms: f64 = 0,

    /// The drafter in `dir` over the target engine `e`, `bits` 4 (affine-4) or 0 (bf16), for up to `cap` streams a hold.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, bits: u8, cap: usize) !*Draft {
        if (e.o.world > 1) return error.DrafterNeedsOneRank;
        if (e.act != .bf16) return error.DrafterNeedsBf16;
        const m = e.model();
        const c = try Config.read(gpa, io, dir, .{ .hidden = m.spec.hidden, .vocab = m.head.n, .layers = m.spec.n_layers });
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .c = c, .w = undefined, .head = undefined, .head_buf = null, .restricted = false, .cap = @max(cap, 1), .table_rows = 0, .arena = undefined, .stage = undefined, .dev = undefined, .ks = undefined, .graphs = graph_cache.Cache(Key, u64).init(gpa), .use_graphs = e.headGraphs() };
        errdefer d.graphs.deinit();
        d.table_rows = @max(c.window, d.cap * 17);
        d.w = try weights.load(gpa, io, &e.driver, dir, c, bits);
        errdefer d.w.deinit(gpa);
        try d.draftHead();
        errdefer if (d.head_buf) |*b| b.free();
        const at = d.layout();
        d.stage = try hip.HostBuffer.alloc(&e.driver, at.total);
        errdefer d.stage.free();
        d.dev = try hip.DeviceBuffer.alloc(&e.driver, at.total);
        errdefer d.dev.free();
        try d.dev.fill8(0);
        d.ks = try hip.DeviceBuffer.alloc(&e.driver, d.cap * c.block * 4);
        errdefer d.ks.free();
        const ks = try gpa.alloc(i32, d.cap * c.block);
        defer gpa.free(ks);
        for (ks, 0..) |*k, r| k.* = if (r % c.block == 0) 0 else @intCast(c.topk);
        try d.ks.upload(0, std.mem.sliceAsBytes(ks));
        d.arena = try hip.Arena.init(&e.driver, try d.scratch());
        errdefer d.arena.deinit();
        try d.time();
        return d;
    }

    pub fn deinit(d: *Draft) void {
        d.e.drain();
        var it = d.streams.valueIterator();
        while (it.next()) |ctx| d.drop(ctx.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |ctx| d.drop(ctx);
        d.pool.deinit(d.gpa);
        d.graphs.deinit();
        d.ks.free();
        d.dev.free();
        d.stage.free();
        d.arena.deinit();
        if (d.head_buf) |*b| b.free();
        d.w.deinit(d.gpa);
        d.gpa.destroy(d);
    }

    /// Device bytes the drafter in `dir` holds beside the target: weights and draft head, then each stream's rings.
    pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: config.Target, bits: u8) !struct { fixed: usize, stream: usize } {
        const c = try Config.read(gpa, io, dir, target);
        const head: usize = if (target.vocab == stock_vocab) width() * target.hidden * 5 / 8 else 0; // 4-bit rows, g64 tables
        return .{ .fixed = try weights.deviceBytes(gpa, io, dir, c, bits) + head + (256 << 20), .stream = c.ringBytes() };
    }

    fn width() usize {
        var n: usize = 0;
        for (spans) |s| n += s[1] - s[0];
        return n;
    }

    /// The draft head: the target's head rows of `spans` side by side (the stock vocabulary, else every row), in the row
    /// kernel's tiled layout. An affine-4 group-64 head with bf16 tables is tiled as it is, an EXL3 one dequantized and
    /// quantized to affine-4 first; any other is refused.
    fn draftHead(d: *Draft) !void {
        const full = d.e.model().head;
        if (full.partial or full.n % 32 != 0) return error.UnsupportedDraftHead;
        d.restricted = full.n == stock_vocab;
        const all = [_][2]u32{.{ 0, full.n }};
        const parts: []const [2]u32 = if (d.restricted) &spans else &all;
        const k: usize = full.k;
        const groups = k / 64;
        var n: usize = 0;
        for (parts) |s| n += s[1] - s[0];
        const gpa = d.gpa;
        const words = try gpa.alloc(u32, n * k / 8);
        defer gpa.free(words);
        const scales = try gpa.alloc(u16, n * groups);
        defer gpa.free(scales);
        const biases = try gpa.alloc(u16, n * groups);
        defer gpa.free(biases);
        switch (full.handle) {
            .mlx => |h| {
                if (h.bits != 4 or h.group != 64 or h.tables != .bf16) return error.UnsupportedDraftHead;
                var row: usize = 0;
                for (parts) |s| {
                    const rows = s[1] - s[0];
                    try download(&d.e.driver, h.words + s[0] * k / 2, std.mem.sliceAsBytes(words[row * k / 8 ..][0 .. rows * k / 8]));
                    try download(&d.e.driver, h.scale + s[0] * groups * 2, std.mem.sliceAsBytes(scales[row * groups ..][0 .. rows * groups]));
                    try download(&d.e.driver, h.bias + s[0] * groups * 2, std.mem.sliceAsBytes(biases[row * groups ..][0 .. rows * groups]));
                    row += rows;
                }
            },
            .exl3 => |h| {
                const rows = try gpa.alloc(u16, n * k);
                defer gpa.free(rows);
                try d.exl3Rows(h, full.n, parts, rows);
                weights.quantize(rows, n, k, words, scales, biases);
            },
            .dense => return error.UnsupportedDraftHead,
        }
        const tw = try gpa.alloc(u32, n * k / 8);
        defer gpa.free(tw);
        const ts = try gpa.alloc(u16, n * groups);
        defer gpa.free(ts);
        const tb = try gpa.alloc(u16, n * groups);
        defer gpa.free(tb);
        if (weights.mlx_layout) {
            @memcpy(tw, words);
            @memcpy(ts, scales);
            @memcpy(tb, biases);
        } else weights.tile(u16, n, k, words, scales, biases, tw, ts, tb);
        const scale_at = std.mem.alignForward(usize, n * k / 2, 256);
        const bias_at = std.mem.alignForward(usize, scale_at + n * groups * 2, 256);
        var buf = try hip.DeviceBuffer.alloc(&d.e.driver, bias_at + n * groups * 2);
        errdefer buf.free();
        try buf.upload(0, std.mem.sliceAsBytes(tw));
        try buf.upload(scale_at, std.mem.sliceAsBytes(ts));
        try buf.upload(bias_at, std.mem.sliceAsBytes(tb));
        d.head = .{ .words = buf.base(), .scales = buf.base() + scale_at, .biases = buf.base() + bias_at, .n = n, .k = k };
        d.head_buf = buf;
    }

    /// The head's rows `parts` as bf16 (rows, K): the EXL3 product of identity rows, 32 inputs a pass, is W's rows.
    fn exl3Rows(d: *Draft, h: hip.quant.exl3.Matrix, n_all: u32, parts: []const [2]u32, out: []u16) !void {
        const k: u32 = d.e.model().head.k;
        const chunk = 32;
        const e = d.e;
        var x = try hip.DeviceBuffer.alloc(&e.driver, chunk * k * 2);
        defer x.free();
        var y = try hip.DeviceBuffer.alloc(&e.driver, chunk * @as(usize, n_all) * 4);
        defer y.free();
        var work = try hip.DeviceBuffer.alloc(&e.driver, hip.launches.Launcher.exl3Scratch(chunk, k, n_all));
        defer work.free();
        const eye = try d.gpa.alloc(u16, chunk * k);
        defer d.gpa.free(eye);
        const got = try d.gpa.alloc(f32, chunk * @as(usize, n_all));
        defer d.gpa.free(got);
        var c: u32 = 0;
        while (c < k) : (c += chunk) {
            @memset(eye, 0);
            for (0..chunk) |i| eye[i * k + c + i] = 0x3F80; // bf16 1.0
            try x.upload(0, std.mem.sliceAsBytes(eye));
            try e.lib.exl3Project(x.base(), 1, h, n_all, k, chunk, y.base(), 0, work.base(), e.stream.handle);
            e.drain();
            try download(&e.driver, y.base(), std.mem.sliceAsBytes(got));
            var row: usize = 0;
            for (parts) |s| for (s[0]..s[1]) |col| {
                for (0..chunk) |i| out[row * k + c + i] = toBf16(got[i * n_all + col]);
                row += 1;
            };
        }
    }

    /// The token of draft-head column `col`.
    fn token(d: *const Draft, col: u32) u32 {
        if (!d.restricted) return col;
        var at: u32 = 0;
        for (spans) |s| {
            if (col < at + s[1] - s[0]) return s[0] + col - at;
            at += s[1] - s[0];
        }
        return col;
    }

    fn layout(d: *const Draft) Layout {
        const rows = d.cap * d.c.block;
        const a = std.mem.alignForward;
        const desc: usize = 0;
        const ids = a(usize, desc + d.cap * @sizeOf(dft.Desc), 256);
        const table = a(usize, ids + rows * 4, 256);
        const top_ids = a(usize, table + d.table_rows * @sizeOf(dft.Row), 256);
        const top_vals = a(usize, top_ids + rows * d.c.topk * 4, 256);
        const hp = a(usize, top_vals + rows * d.c.topk * 2, 256);
        return .{ .desc = desc, .ids = ids, .rows = table, .top_ids = top_ids, .top_vals = top_vals, .hp = hp, .total = hp + rows * d.c.rank * 2 };
    }

    fn ops(d: *Draft, prefill: bool) Ops {
        return .{ .l = &d.e.lib, .bf16 = true, .stream = d.e.stream.handle, .arena = &d.arena, .prefill = prefill };
    }

    /// The arena the widest absorb and hold take, counted on the counting stream.
    fn scratch(d: *Draft) !usize {
        var most: usize = 0;
        for ([_]bool{ true, false }) |absorb| {
            d.arena = hip.Arena.counting();
            var o = d.ops(absorb);
            o.stream = hip.counting;
            if (absorb) try d.absorbPass(o, 0, absorb_rows) else _ = try d.holdPass(o, d.cap);
            most = @max(most, d.arena.peak);
        }
        return most;
    }

    fn drop(d: *Draft, ctx: *Ctx) void {
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    fn create(d: *Draft) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try hip.DeviceBuffer.alloc(&d.e.driver, d.c.ringBytes()) };
        errdefer ctx.ring.free();
        try ctx.ring.fill8(0);
        return ctx;
    }

    /// `n` contexts ready in the pool, for the streams the server admits.
    pub fn reserve(d: *Draft, n: usize) !void {
        try d.pool.ensureTotalCapacity(d.gpa, n);
        while (d.pool.items.len < n) d.pool.appendAssumeCapacity(try d.create());
    }

    /// Waits for what the staging areas' queued copies read, once.
    fn settle(d: *Draft) !void {
        try d.e.stream.synchronize();
        d.absorb_busy = false;
        d.hold_busy = false;
        d.synced = true;
    }

    // absorb: target rows' taps -> fc -> hidden norm -> every layer's k and v -> the rings

    /// Rows [first, first + n) of the uploaded table, projected and put in their streams' rings.
    fn absorbPass(d: *Draft, o: Ops, first: usize, n: usize) !void {
        const c = d.c;
        const table = d.dev.base() + d.layout().rows + first * @sizeOf(dft.Row);
        const taps = try o.arena.take(n * c.n_taps * c.hidden * 2);
        try o.l.dftGather(table, n, c.n_taps * c.hidden * 2, taps, o.stream);
        const th = try proj(o, .{ .ptr = taps, .kind = .bf16 }, d.w.fc, n);
        const thn: Tensor = .{ .ptr = try o.arena.take(n * c.hidden * 2), .kind = .bf16 };
        try o.rms(th, d.w.hidden_norm, thn, n, c.hidden, c.eps);
        const kv = try proj(o, thn, d.w.kv, n);
        try o.l.dftRingPut(kv.ptr, d.w.kv.rows(), table, n, d.w.k_norms, c.layers, c.kv_heads, c.ringRows(), c.eps, c.theta, o.stream);
    }

    /// Each item's rows (from `rows` when given) at start.. into its context, the last `window` of them at most.
    fn absorbAll(d: *Draft, items: []const dr.Absorb) !void {
        const c = d.c;
        const row_bytes = c.n_taps * c.hidden * 2;
        if (d.absorb_busy) try d.settle();
        const table = std.mem.bytesAsSlice(dft.Row, d.stage.bytes[d.layout().rows..][0 .. d.table_rows * @sizeOf(dft.Row)]);
        var n: usize = 0;
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.ready != .none or f.row_bytes != row_bytes) return error.UnsupportedFeatures;
            const count: usize = if (it.rows) |rs| rs.len else f.rows;
            if (it.rows) |rs| for (rs) |r| if (r >= f.rows) return error.BadDraftRows;
            var start = it.start;
            if (start < ctx.end) ctx.end = start; // a rollback: rows past it are rewritten or never read
            if (start > ctx.end or start < ctx.begin) ctx.begin = start; // a gap (a prompt restored whole): the context starts here
            // only the last window of rows can be seen: earlier ones are never read (a target may keep just a tail)
            const skip = count -| c.window;
            start += skip;
            if (skip > 0) ctx.begin = start;
            for (skip..count) |i| {
                if (n == table.len) return error.TooManyDraftRows;
                const src_row: usize = if (it.rows) |rs| rs[i] else i;
                const at = if (f.ring > 0) ((f.offset + src_row) % f.ring) * row_bytes else f.offset + src_row * row_bytes;
                table[n] = .{ .src = f.buffer + at, .ring = ctx.ring.base(), .pos = @intCast(it.start + i) };
                n += 1;
            }
            ctx.end = start + (count - skip);
            if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
            ctx.slot = null;
        }
        if (n == 0) return;
        const at = d.layout().rows;
        try hip.raw.upload(d.dev, at, d.stage.bytes[at..][0 .. n * @sizeOf(dft.Row)], d.e.stream.handle);
        d.absorb_busy = true;
        var first: usize = 0;
        while (first < n) : (first += absorb_rows) {
            d.arena.reset();
            const rows = @min(absorb_rows, n - first);
            try d.absorbPass(d.ops(rows > 16), first, rows);
        }
    }

    // hold: [pending, mask x (block - 1)] a stream -> the layers over the rings -> the draft head's top-k and the selector

    /// Every launch of a hold of `streams` streams; the selector's projected rows' address.
    fn holdPass(d: *Draft, o: Ops, streams: usize) !u64 {
        const c = d.c;
        const B = c.block;
        const M = streams * B;
        const at = d.layout();
        const base = d.dev.base();
        const desc = base + at.desc;
        const H = c.hidden;
        const h: Tensor = .{ .ptr = try o.arena.take(M * H * 2), .kind = .bf16 };
        try o.embedRows(d.e.model().embed, base + at.ids, M, h);
        const x: Tensor = .{ .ptr = try o.arena.take(M * H * 2), .kind = .bf16 };
        const xin: Tensor = .{ .ptr = try o.arena.take(M * H * 2), .kind = .bf16 };
        const h2: Tensor = .{ .ptr = try o.arena.take(M * H * 2), .kind = .bf16 };
        const q = try o.arena.take(M * c.qWidth() * 2);
        const kb = try o.arena.take(M * c.kvWidth() * 2);
        const att = try o.arena.take(M * c.qWidth() * 2);
        const part = try o.arena.take(dft.partBytes(streams, c.kv_heads, splits));
        const qkv_w = c.qWidth() + 2 * c.kvWidth();
        const mark = o.arena.mark();
        for (d.w.layers, 0..) |l, li| {
            o.arena.release(mark);
            try o.rms(h, l.input_norm, x, M, H, c.eps);
            const dyn = try proj(o, x, l.attn_conv, M);
            try o.l.dftDynConv(x.ptr, dyn.ptr, l.attn_base, 0, xin.ptr, M, H, c.group, 0, B, o.stream);
            const qkv = try proj(o, xin, l.qkv, M);
            try o.l.dftNormRope(qkv.ptr, qkv_w, l.q_norm, q, desc, B, M, c.heads, c.eps, c.theta, o.stream);
            try o.l.dftNormRope(qkv.ptr + c.qWidth() * 2, qkv_w, l.k_norm, kb, desc, B, M, c.kv_heads, c.eps, c.theta, o.stream);
            try o.l.dftAttn(.{ .q = q, .kb = kb, .vb = qkv.ptr + (c.qWidth() + c.kvWidth()) * 2, .vstride = qkv_w, .desc = desc, .layer = li, .streams = streams, .block = B, .heads = c.heads, .kv_heads = c.kv_heads, .ring_rows = c.ringRows(), .window = c.window, .splits = splits, .part = part, .out = att }, o.stream);
            const ao = try proj(o, .{ .ptr = att, .kind = .bf16 }, l.o, M);
            try o.l.dftDynConv(ao.ptr, dyn.ptr, l.attn_base, h.ptr, h2.ptr, M, H, c.group, 1, B, o.stream);
            try o.rms(h2, l.post_norm, x, M, H, c.eps);
            const mdyn = try proj(o, x, l.mlp_conv, M);
            try o.l.dftDynConv(x.ptr, mdyn.ptr, l.mlp_base, 0, xin.ptr, M, H, c.group, 0, B, o.stream);
            const gu = [2]Tensor{ try proj(o, xin, l.gate, M), try proj(o, xin, l.up, M) };
            try o.siluMul(gu[0], gu[1], gu[0], M * c.intermediate);
            const down = try proj(o, gu[0], l.down, M);
            try o.l.dftDynConv(down.ptr, mdyn.ptr, l.mlp_base, h2.ptr, h.ptr, M, H, c.group, 1, B, o.stream);
        }
        o.arena.release(mark);
        // the mask rows' best tokens by the draft head, and every row's selector projection (the pending rows' unused)
        try o.rms(h, d.w.final_norm, x, M, H, c.eps);
        const logits = try proj(o, x, .{ .q4 = d.head }, M);
        try o.topkRows(logits, M, d.head.n, d.ks.base(), c.topk, base + at.top_ids, base + at.top_vals);
        const hp = try proj(o, x, d.w.selector, M);
        return hp.ptr;
    }

    const Recording = struct { d: *Draft, o: Ops, streams: usize };

    fn record(r: Recording) anyerror!u64 {
        return r.d.holdPass(r.o, r.streams);
    }

    /// The hold's launches: replayed from its shape's graph, captured the first time it is met, or eager.
    fn launchHold(d: *Draft, streams: usize) !u64 {
        d.arena.reset();
        const o = d.ops(false);
        const stream = d.e.stream;
        d.graphs.rounds += 1;
        const entry = if (d.use_graphs) try d.graphs.find(.{ .streams = @intCast(streams) }) else null;
        if (entry) |e| switch (e.state) {
            .ready => {
                try e.exec.?.launchOn(stream);
                d.graphs.replayed += 1;
                return e.out;
            },
            .seen => {
                if (try graph_cache.Cache(Key, u64).record(stream, Recording{ .d = d, .o = o, .streams = streams }, record)) |rec| {
                    if (d.graphs.keep(e, rec.graph, stream, rec.out)) {
                        try e.exec.?.launchOn(stream);
                        return e.out;
                    } else |_| {}
                }
                e.state = .failed;
                d.arena.reset();
            },
            .failed => {},
        };
        return d.holdPass(o, streams);
    }

    /// Every item's block over its context as one batch; the results come back with the next `held`.
    fn holdAll(d: *Draft, items: []const dr.Hold) !void {
        const c = d.c;
        const B = c.block;
        if (items.len > d.cap) return error.TooManyDraftStreams;
        if (d.hold_busy) try d.settle();
        const at = d.layout();
        const desc = std.mem.bytesAsSlice(dft.Desc, d.stage.bytes[at.desc..][0 .. d.cap * @sizeOf(dft.Desc)]);
        const ids = std.mem.bytesAsSlice(i32, d.stage.bytes[at.ids..][0 .. d.cap * B * 4]);
        for (items, 0..) |it, i| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            const pos0 = it.position - 1;
            if (ctx.end != pos0 or it.pending >= c.vocab) return error.DraftContextBehind;
            desc[i] = .{ .ring = ctx.ring.base(), .begin = @intCast(ctx.begin), .pos0 = @intCast(pos0) };
            ids[i * B] = @intCast(it.pending);
            @memset(ids[i * B + 1 .. (i + 1) * B], @intCast(c.mask));
            ctx.slot = i;
            ctx.pending = it.pending;
            ctx.depth = @min(it.depth, B - 1);
            ctx.held_n = null;
            ctx.n_lanes = null;
            if (it.lanes) |l| {
                if (l.parents.len > max_lanes) return error.TooManyLanes;
                for (l.parents, l.depths, l.ranks, 0..) |par, dep, rank, j| {
                    if (dep >= B - 1 or rank >= c.topk or par >= @as(i32, @intCast(j)) or par < -1) return error.BadLanes;
                    ctx.parents[j] = par;
                    ctx.depths[j] = dep;
                    ctx.ranks[j] = rank;
                }
                ctx.n_lanes = l.parents.len;
            }
        }
        // the descriptors and ids up in one copy, the batch, its results down: no wait until `held`
        try hip.raw.upload(d.dev, 0, d.stage.bytes[0..at.rows], d.e.stream.handle);
        d.hold_busy = true;
        const hp = try d.launchHold(items.len);
        const rows = items.len * B;
        try hip.raw.download(d.dev, at.top_ids, d.stage.bytes[at.top_ids..][0 .. rows * c.topk * 4], d.e.stream.handle);
        try hip.raw.download(d.dev, at.top_vals, d.stage.bytes[at.top_vals..][0 .. rows * c.topk * 2], d.e.stream.handle);
        try hipDownload(d, hp, d.stage.bytes[at.hp..][0 .. rows * c.rank * 2]);
        d.synced = false;
        d.held_rows = rows;
    }

    fn hipDownload(d: *Draft, src: u64, out: []u8) !void {
        try hip.runtime.check(d.e.driver.api.hipMemcpyDtoHAsync(out.ptr, @ptrFromInt(src), out.len, d.e.stream.handle));
    }

    /// Row `r` of the last hold's `rank`-th candidate after `pred`: its draft-head logit + pred . projected . successor,
    /// ties to the head's earlier one.
    fn choice(d: *const Draft, r: usize, pred: u32, rank: usize) u32 {
        const c = d.c;
        const at = d.layout();
        const top_ids = std.mem.bytesAsSlice(i32, d.stage.bytes[at.top_ids..][r * c.topk * 4 ..][0 .. c.topk * 4]);
        const top_vals = std.mem.bytesAsSlice(u16, d.stage.bytes[at.top_vals..][r * c.topk * 2 ..][0 .. c.topk * 2]);
        const hp = std.mem.bytesAsSlice(u16, d.stage.bytes[at.hp..][r * c.rank * 2 ..][0 .. c.rank * 2]);
        var ph: [1024]f32 = undefined;
        const p = d.w.predecessor[@as(usize, pred) * c.rank ..][0..c.rank];
        for (ph[0..c.rank], p, hp) |*x, a, b| x.* = bf16(a) * bf16(b);
        var scores: [64]f32 = undefined;
        var tokens: [64]u32 = undefined;
        for (0..c.topk) |j| {
            tokens[j] = d.token(@intCast(top_ids[j]));
            const s = d.w.successor[@as(usize, tokens[j]) * c.rank ..][0..c.rank];
            var edge: f32 = 0;
            for (ph[0..c.rank], s) |a, b| edge += a * bf16(b);
            scores[j] = bf16(top_vals[j]) + edge;
        }
        var taken: [64]bool = @splat(false);
        var pick: usize = 0;
        for (0..rank + 1) |_| {
            var best: f32 = -std.math.inf(f32);
            pick = 0;
            for (0..c.topk) |j| if (!taken[j] and (scores[j] > best or (scores[j] == best and tokens[j] < tokens[pick]))) {
                best = scores[j];
                pick = j;
            };
            taken[pick] = true;
        }
        return tokens[pick];
    }

    /// The context's drafts from its last hold: the path through each row's best after the one before, or the tree.
    fn choose(d: *Draft, ctx: *Ctx) void {
        const B = d.c.block;
        const first = (ctx.slot orelse return) * B + 1;
        if (ctx.n_lanes) |n| {
            for (0..n) |i| {
                const pred = if (ctx.parents[i] < 0) ctx.pending else ctx.held[@intCast(ctx.parents[i])];
                ctx.held[i] = d.choice(first + ctx.depths[i], pred, ctx.ranks[i]);
            }
            ctx.held_n = n;
            return;
        }
        var pred = ctx.pending;
        for (0..B - 1) |r| {
            ctx.held[r] = d.choice(first + r, pred, 0);
            pred = ctx.held[r];
        }
        ctx.held_n = ctx.depth;
    }

    /// One hold of `streams` streams over freshly absorbed contexts, timed: wall ms (with `held`) and the GPU's ms.
    fn timeHold(d: *Draft, ctxs: []const *Ctx, tree: ?*const Shape) !struct { wall: f64, gpu: f64 } {
        var ss: [64]Stream = undefined;
        var holds: [64]dr.Hold = undefined;
        var ptrs: [64]*Stream = undefined;
        var outs: [64][]u32 = undefined;
        var bufs: [64][max_lanes]u32 = undefined;
        for (ctxs, 0..) |ctx, i| {
            ss[i] = undefined;
            ptrs[i] = &ss[i];
            try d.streams.put(d.gpa, &ss[i], ctx);
            holds[i] = .{ .stream = &ss[i], .pending = 1, .position = ctx.end + 1, .depth = @intCast(d.c.block - 1), .lanes = tree };
            outs[i] = bufs[i][0..if (tree) |t| t.parents.len else d.c.block - 1];
        }
        defer for (ptrs[0..ctxs.len]) |s| {
            _ = d.streams.remove(s);
        };
        var start = try hip.Event.init(&d.e.driver, true);
        defer start.deinit();
        var stop = try hip.Event.init(&d.e.driver, true);
        defer stop.deinit();
        var wall: f64 = std.math.inf(f64);
        var gpu: f64 = std.math.inf(f64);
        for (0..10) |_| {
            const t0 = std.Io.Clock.awake.now(d.io);
            try start.record(d.e.stream);
            try d.holdAll(holds[0..ctxs.len]);
            try stop.record(d.e.stream);
            try d.heldAll(ptrs[0..ctxs.len], outs[0..ctxs.len]);
            wall = @min(wall, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds())) / 1e6);
            gpu = @min(gpu, try hip.Event.elapsedMs(start, stop));
        }
        return .{ .wall = wall, .gpu = gpu };
    }

    /// The hold's cost on a lone stream with a full window: one draft's share is the round rule's step_ms.
    fn time(d: *Draft) !void {
        const c = d.c;
        const ctx = try d.create();
        defer d.drop(ctx);
        // a window of rows from zeroed taps (the staging's own device area does: any rows will)
        var taps = try hip.DeviceBuffer.alloc(&d.e.driver, c.n_taps * c.hidden * 2);
        defer taps.free();
        try taps.fill8(0);
        var s: Stream = undefined;
        try d.streams.put(d.gpa, &s, ctx);
        defer _ = d.streams.remove(&s);
        const rows = c.window;
        // every row reads the same tap row: features with a zero row stride are not a contract shape, so absorb row by row
        var at: u64 = 0;
        while (at < rows) : (at += absorb_rows) {
            var picks: [absorb_rows]u32 = @splat(0);
            const n: usize = @min(absorb_rows, rows - at);
            try d.absorbAll(&.{.{ .stream = &s, .features = .{ .buffer = taps.base(), .rows = 1, .row_bytes = @intCast(c.n_taps * c.hidden * 2), .space = .device, .dtype = .bf16 }, .start = at, .follow = &.{}, .rows = picks[0..n] }});
        }
        try d.settle();
        _ = d.streams.remove(&s);
        const chain = try d.timeHold(&.{ctx}, null);
        // a 16-lane tree: the four best first drafts, the best's own best three, then a chain to the block's end
        var parents = [_]i32{ -1, -1, -1, -1, 0, 0, 1, 4, 4, 5, 7, 10, 11, 12, 8, 6 };
        var ranks = [_]u8{ 0, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
        var depths = [_]u8{ 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 3, 2 };
        const tree: Shape = .{ .parents = &parents, .ranks = &ranks, .depths = &depths, .expected = 0 };
        const t = try d.timeHold(&.{ctx}, &tree);
        d.step_ms = chain.wall / @as(f64, @floatFromInt(c.block - 1));
        d.gpu_ms = chain.gpu;
        d.tree_ms = t.wall;
        std.log.info("drafter: a hold of {d} drafts takes {d:.2} ms ({d:.2} ms on the GPU), a 16-lane tree {d:.2} ms; {d:.3} ms a draft", .{ c.block - 1, chain.wall, chain.gpu, t.wall, d.step_ms });
    }

    fn heldAll(d: *Draft, streams: []const *Stream, out: []const []u32) !void {
        if (!d.synced) try d.settle();
        for (streams, out) |s, o| {
            const ctx = d.streams.get(s) orelse return error.UnknownStream;
            if (ctx.held_n == null) d.choose(ctx);
            const n = ctx.held_n orelse return error.TooFewDrafts;
            if (o.len > n) return error.TooFewDrafts;
            @memcpy(o, ctx.held[0..o.len]);
        }
    }

    // the Drafter contract

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(p: *anyopaque) *Draft {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        const d = cast(p);
        return d.c.taps[0..d.c.n_taps];
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = cast(p);
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = d.step_ms, .prior = prior[0..@min(prior.len, d.c.block - 1)], .plain_guard = true, .batched = true, .stream_bytes = d.c.ringBytes(), .trees = true };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        if (d.streams.count() >= d.cap) return error.TooManyDraftStreams;
        const ctx = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, ctx) catch d.drop(ctx);
        ctx.reset();
        try d.streams.put(d.gpa, s, ctx);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        return cast(p).absorbAll(items);
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        return cast(p).holdAll(items);
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        return cast(p).heldAll(streams, out);
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }
};

/// A product on either weight format: affine-4 through the drafter's tiled row kernel, or bf16 through its own.
pub fn proj(o: Ops, x: Tensor, l: Linear, rows: usize) !Tensor {
    switch (l) {
        .q4 => |p| {
            if (weights.mlx_layout) return o.affine(x, .{ .words = p.words, .scale = p.scales, .bias = p.biases, .tables = .bf16, .n = @intCast(p.n), .k = @intCast(p.k), .bits = 4, .group = 64 }, rows, false);
            const out = try o.arena.take(rows * p.n * 2);
            try o.l.dftQ4Rows(x.ptr, p.words, p.scales, p.biases, out, rows, p.n, p.k, o.stream);
            return .{ .ptr = out, .kind = .bf16 };
        },
        .bf16 => |b| {
            const out = try o.arena.take(rows * b.n * 2);
            try o.l.dftBf16Rows(x.ptr, b.w, out, rows, b.n, b.k, o.stream);
            return .{ .ptr = out, .kind = .bf16 };
        },
    }
}

/// `bytes` of device memory at `src` on the host (a load-time copy).
fn download(r: *const hip.Runtime, src: u64, bytes: []u8) !void {
    try hip.runtime.check(r.api.hipMemcpyDtoH(bytes.ptr, @ptrFromInt(src), bytes.len));
}

fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "the draft head's columns name the stock vocabulary's two spans" {
    var d: Draft = undefined;
    d.restricted = true;
    try std.testing.expectEqual(@as(usize, 98592), Draft.width());
    try std.testing.expectEqual(@as(u32, 0), d.token(0));
    try std.testing.expectEqual(@as(u32, 98303), d.token(98303));
    try std.testing.expectEqual(@as(u32, 248032), d.token(98304));
    try std.testing.expectEqual(@as(u32, 248319), d.token(98591));
}
