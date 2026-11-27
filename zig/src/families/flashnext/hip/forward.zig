//! Flash Next's forward on a HIP rank: a stream's rows through every layer, a window's each with a serial step's bits.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const tables = @import("tables.zig");
const mixers = @import("mixers.zig");
const ngram = @import("ngram.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const Kind = hip.ops.Kind;

/// What a forward reads from the host: the pinned staging its uploads go through, and the n-gram tables.
pub const Host = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    staging: hip.HostBuffer,
    /// Bytes of `staging` this forward used: the stream is synchronized before a forward starts again.
    used: usize = 0,
    tables: *tables.Tables,
    /// The staged rows' n-gram ids, kept for a gather a prompt's forward makes after its first launches.
    hashed: std.ArrayList([ngram.heads]u32) = .empty,

    pub fn deinit(h: *Host) void {
        h.staging.free();
        h.hashed.deinit(h.gpa);
    }

    /// `bytes` copied through the staging into fresh arena memory, queued on the stream.
    pub fn put(h: *Host, o: Ops, bytes: []const u8) !u64 {
        const from = std.mem.alignForward(usize, h.used, 16);
        if (from + bytes.len > h.staging.bytes.len) return error.StagingTooSmall;
        const dst = try o.arena.take(bytes.len);
        @memcpy(h.staging.bytes[from..][0..bytes.len], bytes);
        h.used = from + bytes.len;
        if (o.stream != hip.counting and bytes.len > 0) try hip.runtime.check(h.d.api.hipMemcpyHtoDAsync(
            @ptrFromInt(dst),
            h.staging.bytes[from..].ptr,
            bytes.len,
            o.stream,
        ));
        return dst;
    }
};

/// A device buffer of `n` values of `kind` from the arena.
pub fn take(o: Ops, kind: Kind, n: usize) !Tensor {
    return .{ .ptr = try o.arena.take(n * kind.size()), .kind = kind };
}

/// `t` advanced by `n` values.
pub fn at(t: Tensor, n: usize) Tensor {
    return .{ .ptr = t.ptr + n * t.kind.size(), .kind = t.kind };
}

/// One call's rows: their tokens, the position of the first, and whether they are a window (else a prompt's).
pub const Call = struct { tokens: []const u32, pos: usize, window: bool, marks: []const Mark = &.{} };

/// One stream's rows in a forward: its caches, its tokens and the position of its first row.
pub const Seg = struct { c: *cache.Caches, tokens: []const u32, pos: usize, marks: []const Mark = &.{} };

/// A kept row inside a prompt chunk (a multiple of 64): its linear and n-gram state to `side`, as a chunk ending there.
pub const Mark = struct { row: usize, side: u64 };

/// The final mixer's rows (rows, hidden) for the head, and the residual streams (rows, wide) before it.
pub const Out = struct { mixed: Tensor, streams: Tensor };

/// A hyper-connection's read of the normed streams `x`: the block input, and the inject gates when `inject` is set.
pub fn hcPre(o: Ops, m: *const model.Model, hc: model.Hc, x: Tensor, rows: usize, inject: ?Tensor) !Tensor {
    if (try readout(o, m, hc, x, rows, inject)) |mixed| return mixed;
    const d = m.dims;
    const z = try o.project(x, hc.down, rows, true);
    const act = try take(o, m.act, rows * d.low);
    const kind = @backingInt(m.act);
    try o.l.fn_hc_gates(
        @ptrFromInt(z.ptr),
        @ptrFromInt(act.ptr),
        if (inject) |i| @ptrFromInt(i.ptr) else null,
        kind,
        @intCast(rows),
        @intCast(d.low),
        @intCast(d.streams),
        o.stream,
    );
    const g = try o.project(act, hc.up, rows, true);
    const mixed = try take(o, m.act, rows * d.hidden);
    try o.l.fn_hc_mix(
        @ptrFromInt(g.ptr),
        @ptrFromInt(x.ptr),
        @ptrFromInt(mixed.ptr),
        kind,
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        o.stream,
    );
    return mixed;
}

/// The bf16 [n][k] rows of a BF16-fold projection (FP4-Local's hyper-connection weights), or null.
fn bf16Rows(p: hip.ops.Projection) ?u64 {
    return switch (p.handle) {
        .dense => |h| if (h.fold and h.dtype == .bf16) h.weight else null,
        else => null,
    };
}

/// A prompt's streams take the Loom read-out (launches/hc_readout.zig): hipcc's bytes, its gates and mix fused.
fn fused(o: Ops, m: *const model.Model) bool {
    const d = m.dims;
    return o.prefill and m.act == .bf16 and o.l.hcr.takes(d.streams, d.hidden, d.low, d.low);
}

/// hcPre on the Loom read-out: the down with the gates, the up with the mix; null where it does not take `hc`.
fn readout(o: Ops, m: *const model.Model, hc: model.Hc, x: Tensor, rows: usize, inject: ?Tensor) !?Tensor {
    const d = m.dims;
    if (!fused(o, m) or !o.l.hcr.takes(d.streams, d.hidden, d.low, hc.down.n)) return null;
    if (hc.down.k != d.wide() or hc.up.n != d.wide() or hc.up.k != d.low or (inject != null) != (hc.down.n > d.low)) return null;
    const down = bf16Rows(hc.down) orelse return null;
    const up = bf16Rows(hc.up) orelse return null;
    const act = try take(o, m.act, rows * d.low);
    try o.l.hcr.down(o.l.r, x.ptr, down, act.ptr, if (inject) |i| i.ptr else null, rows, hc.down.n, d.low, o.stream);
    const mixed = try take(o, m.act, rows * d.hidden);
    try o.l.hcr.upMix(o.l.r, act.ptr, up, x.ptr, mixed.ptr, rows, d.low, o.stream);
    return mixed;
}

/// A block's output into the streams, then their norm: a tp rank's fp32 share is gathered and summed in the kernel.
pub fn hcBranch(o: Ops, m: *const model.Model, h: Tensor, branch: Tensor, inject: Tensor, x: Tensor, scale: u64, rows: usize) !void {
    if (branch.kind != .f32) return hcNorm(o, m, h, branch, inject, x, scale, rows);
    const d = m.dims;
    const c = m.tp.?;
    const n = rows * d.hidden;
    const parts = try o.arena.of(f32, c.world * n);
    try c.allGather(branch.ptr, parts, n, .f32, o.stream);
    try o.l.fn_hc_norm_shares(@ptrFromInt(h.ptr), @ptrFromInt(parts), @ptrFromInt(inject.ptr), @ptrFromInt(h.ptr), @ptrFromInt(x.ptr), @ptrFromInt(scale), @backingInt(m.act), @intCast(rows), @intCast(d.streams), @intCast(d.hidden), d.eps, @intCast(c.world), @intCast(n), o.stream);
}

/// h += round(branch * inject) per stream when a branch is given, then x = the streams normed with `scale`.
pub fn hcNorm(
    o: Ops,
    m: *const model.Model,
    h: Tensor,
    branch: ?Tensor,
    inject: ?Tensor,
    x: Tensor,
    scale: u64,
    rows: usize,
) !void {
    const d = m.dims;
    if (fused(o, m) and (branch == null or inject != null)) {
        const b: ?u64 = if (branch) |t| t.ptr else null;
        const i: ?u64 = if (inject) |t| t.ptr else null;
        return o.l.hcr.norm(o.l.r, h.ptr, b, if (b != null) i else null, scale, x.ptr, rows, d.eps, o.stream);
    }
    try o.l.fn_hc_norm(
        @ptrFromInt(h.ptr),
        if (branch) |b| @ptrFromInt(b.ptr) else null,
        if (inject) |i| @ptrFromInt(i.ptr) else null,
        if (branch != null) @ptrFromInt(h.ptr) else null,
        @ptrFromInt(x.ptr),
        @ptrFromInt(scale),
        @backingInt(m.act),
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        d.eps,
        o.stream,
    );
}

/// A forward's host inputs, laid out in one span of the staging (byte offsets into it) and sent up in one copy.
pub const Staged = struct {
    /// The staging bytes the span starts at (another staging's when a forward takes inputs staged ahead).
    src: [*]const u8 = undefined,
    from: usize = 0,
    len: usize = 0,
    /// The n-gram rows not gathered yet: a prompt's forward gathers them while its first layers run (`gather`).
    late: bool = false,
    ids: usize = 0,
    pos: usize = 0,
    ends: usize = 0,
    complete: usize = 0,
    local: usize = 0,
    words: usize = 0,
    scales: usize = 0,
    biases: usize = 0,
    /// Blocks a window's scores walk: all its caches can hold, so the launch is the same at any position.
    bound: usize = 0,
};

/// The staged inputs on the device: token ids (one a stream), positions, ends, complete blocks, n-gram rows.
pub const Inputs = struct {
    ids: u64,
    pos: u64,
    ends: u64,
    complete: u64,
    local: u64,
    words: u64,
    scales: u64,
    biases: u64,
    bound: usize,
};

/// The staged span of a forward over `rows` rows: each input's offset in it and its length.
pub fn layout(m: *const model.Model, t: *const tables.Tables, rows: usize) Staged {
    const n16 = rows * ngram.heads;
    var s: Staged = .{};
    var used: usize = 0;
    inline for (.{
        .{ "ids", rows * m.dims.streams * 4 },
        .{ "pos", rows * 4 },
        .{ "ends", rows * 4 },
        .{ "complete", rows * 4 },
        .{ "local", n16 * 4 },
        .{ "words", n16 * t.wordBytes() },
        .{ "scales", n16 * t.tableBytes() },
        .{ "biases", n16 * t.tableBytes() },
    }) |f| {
        @field(s, f[0]) = used;
        used = std.mem.alignForward(usize, used + f[1], 16);
    }
    s.len = used;
    return s;
}

/// Every host input of a forward of `segs` written into the staging, n-gram rows hashed (a window's gathered too).
pub fn stage(m: *const model.Model, host: *Host, segs: []const Seg, window: bool) !Staged {
    const d = m.dims;
    const t = host.tables;
    if (d.ple_dim / t.dims() != ngram.heads) return error.UnsupportedNgramHeads;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    const n16 = rows * ngram.heads;
    var s = layout(m, t, rows);
    s.from = std.mem.alignForward(usize, host.used, 16);
    const used = s.len;
    if (s.from + used > host.staging.bytes.len) return error.StagingTooSmall;
    const span = host.staging.bytes[s.from..][0..used];
    s.src = span.ptr;
    const ints = struct {
        fn of(b: []u8, off: usize, n: usize) []i32 {
            return @as([*]i32, @ptrCast(@alignCast(b[off..].ptr)))[0..n];
        }
    }.of;
    const ids = ints(span, s.ids, rows * d.streams);
    const pos = ints(span, s.pos, rows);
    const ends = ints(span, s.ends, rows);
    const complete = ints(span, s.complete, rows);
    for (ints(span, s.local, n16), 0..) |*x, i| x.* = @intCast(i);
    try host.hashed.resize(host.gpa, rows);
    const hashed = host.hashed.items;
    var row: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        for (g.tokens, 0..) |tok, r| {
            if (tok >= d.vocab) return error.InvalidToken; // embed reads the table at id unchecked
            for (0..d.streams) |k| ids[(row + r) * d.streams + k] = @intCast(tok);
            pos[row + r] = @intCast(g.pos + r);
            ends[row + r] = @intCast(g.pos + r + 1);
            complete[row + r] = @intCast((g.pos + r + 1) / d.index_ratio);
        }
        t.hasher.ids(g.c.history, g.tokens, hashed[row..][0..n]);
        row += n;
    }
    s.late = true;
    if (window) try gather(host, &s);
    // a window's scores walk every block its caches hold: the launch is the same at any position
    if (window) for (segs) |g| {
        s.bound = @max(s.bound, g.c.layout.cap / d.index_ratio);
    };
    host.used = s.from + used;
    return s;
}

/// The n-gram rows of the span `stage` last wrote into `host` gathered into it (no launch).
pub fn gather(host: *Host, s: *Staged) !void {
    if (!s.late) return;
    const t = host.tables;
    const n16 = host.hashed.items.len * ngram.heads;
    const span = host.staging.bytes[s.from..][0..s.len];
    if (span.ptr != s.src) return error.StagingMismatch;
    try t.gather(
        @as([*]const u32, @ptrCast(host.hashed.items.ptr))[0..n16],
        span[s.words..][0 .. n16 * t.wordBytes()],
        span[s.scales..][0 .. n16 * t.tableBytes()],
        span[s.biases..][0 .. n16 * t.tableBytes()],
    );
    s.late = false;
}

/// The staged span up to arena memory, queued (inside a graph when one records); late n-gram rows go up later.
pub fn send(o: Ops, host: *Host, s: Staged) !Inputs {
    const dst = try o.arena.take(s.len);
    if (o.stream != hip.counting) try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(
        @ptrFromInt(dst),
        s.src,
        if (s.late) s.words else s.len,
        o.stream,
    ));
    return .{
        .ids = dst + s.ids,
        .pos = dst + s.pos,
        .ends = dst + s.ends,
        .complete = dst + s.complete,
        .local = dst + s.local,
        .words = dst + s.words,
        .scales = dst + s.scales,
        .biases = dst + s.biases,
        .bound = s.bound,
    };
}

/// The n-gram rows' keys and values of every row of a forward in one product: the matrix is read once a forward.
fn pleKv(o: Ops, m: *const model.Model, host: *Host, rows: usize, in: Inputs) !Tensor {
    const d = m.dims;
    const t = host.tables;
    // FP4-Local's rows arrive decoded: the staged bf16 rows are the embedding (token-major, heads in turn)
    const emb: Tensor = switch (t.store) {
        .nvfp4 => if (m.act == .bf16) .{ .ptr = in.words, .kind = .bf16 } else return error.UnsupportedNgramHeads,
        .mlx => |r| e: {
            const e = try take(o, m.act, rows * d.ple_dim);
            try o.l.tf_embed_rows(
                @ptrFromInt(in.words),
                @ptrFromInt(in.scales),
                @ptrFromInt(in.biases),
                @backingInt(Kind.bf16),
                @ptrFromInt(in.local),
                @intCast(rows * ngram.heads),
                r.bits,
                @intCast(r.group),
                @intCast(r.dims),
                @ptrFromInt(e.ptr),
                @backingInt(m.act),
                o.stream,
            );
            break :e e;
        },
    };
    return o.project(emb, m.ple.kv, rows, false);
}

/// The n-gram layer over a segment's rows, `kv` their rows of `pleKv`: h += gated + silu(conv) over `cin`.
fn ple(o: Ops, m: *const model.Model, host: *Host, c: *cache.Caches, h: Tensor, call: Call, cin: u64, kv: Tensor) !void {
    const d = m.dims;
    const rows = call.tokens.len;
    const mark = o.arena.mark();
    defer o.arena.release(mark);
    const gated = try take(o, m.act, rows * d.wide());
    const tail = c.layout.tail_rows * d.wide() * m.act.size();
    try hip.raw.copy(.{ .r = host.d, .ptr = @ptrFromInt(cin), .len = tail }, 0, c.tail(), tail, o.stream);
    const kind = @backingInt(m.act);
    try o.l.fn_ple_gate(
        @ptrFromInt(kv.ptr),
        @ptrFromInt(h.ptr),
        @ptrFromInt(m.ple.key_scale),
        @ptrFromInt(m.ple.query_scale),
        @ptrFromInt(m.ple.conv_scale),
        @ptrFromInt(gated.ptr),
        @ptrFromInt(cin + tail),
        kind,
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        d.eps,
        o.stream,
    );
    try o.l.fn_ple_conv(
        @ptrFromInt(cin),
        @ptrFromInt(m.ple.conv),
        @ptrFromInt(gated.ptr),
        @ptrFromInt(h.ptr),
        @ptrFromInt(h.ptr),
        kind,
        @intCast(rows),
        @intCast(d.wide()),
        @intCast(d.ple_taps),
        @intCast(d.ngram),
        o.stream,
    );
    // a prompt's rows are kept: the tail moves on now; a window's waits for its keep
    if (!call.window) {
        try hip.raw.copy(
            .{ .r = host.d, .ptr = @ptrFromInt(c.tail()), .len = tail },
            0,
            cin + rows * d.wide() * m.act.size(),
            tail,
            o.stream,
        );
        for (call.marks) |mk| try hip.raw.copy(
            .{ .r = host.d, .ptr = @ptrFromInt(c.sideTail(mk.side)), .len = tail },
            0,
            cin + mk.row * d.wide() * m.act.size(),
            tail,
            o.stream,
        );
        c.history = ngram.after(c.history, call.tokens);
    }
}

/// Every segment's rows through every layer: one stream's prompt chunk, or several streams' windows together.
pub fn run(ops: Ops, m: *const model.Model, host: *Host, segs: []const Seg, window: bool, inputs: Staged) !Out {
    var staged = inputs;
    var o = ops;
    o.prefill = !window;
    o.window = window;
    const d = m.dims;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    const in = try send(o, host, staged);
    const dev_ids = in.ids;
    const dev_pos = in.pos;
    const h = try take(o, m.act, rows * d.wide());
    const x = try take(o, m.act, rows * d.wide());
    try o.embedRows(m.embed, dev_ids, rows * d.streams, h);
    var cins: [max_segs]u64 = undefined;
    if (segs.len > max_segs) return error.TooManyStreams;
    for (segs, cins[0..segs.len]) |g, *cin| cin.* = if (window)
        g.c.cin.base()
    else
        try o.arena.take((g.c.layout.tail_rows + g.tokens.len) * d.wide() * m.act.size());
    // a window's words go up inside the round: a graph holds the copy, so every replay sends this round's own
    if (window) if (m.gdn) |s| for (segs) |g| {
        const p = s.planFor(g.c.id) orelse return error.CachesNotPlanned;
        try p.send(o.stream);
    };
    for (m.layers, 0..) |layer, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        if (i == m.ple.layer) {
            // the n-gram rows gathered while the layers before run, then sent behind them
            if (staged.late) {
                try gather(host, &staged);
                if (o.stream != hip.counting) try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(
                    @ptrFromInt(in.words),
                    staged.src + staged.words,
                    staged.len - staged.words,
                    o.stream,
                ));
            }
            const kv = try pleKv(o, m, host, rows, in);
            var off: usize = 0;
            for (segs, cins[0..segs.len]) |g, cin| {
                try ple(
                    o,
                    m,
                    host,
                    g.c,
                    at(h, off * d.wide()),
                    .{ .tokens = g.tokens, .pos = g.pos, .window = window, .marks = g.marks },
                    cin,
                    at(kv, off * m.ple.kv.n),
                );
                off += g.tokens.len;
            }
        }
        if (i == 0 or i == m.ple.layer) try hcNorm(o, m, h, null, null, x, layer.ahc.scale, rows);
        const inject = try take(o, m.act, rows * d.streams);
        const mixed = try hcPre(o, m, layer.ahc, x, rows, inject);
        const branch = switch (layer.mixer) {
            .linear => |l| try mixers.linear(o, m, l, segs, i, mixed, rows, window),
            .sparse => |s| try mixers.sparse(
                o,
                m,
                d,
                s,
                segs,
                i,
                mixed,
                rows,
                dev_pos,
                host,
                .{ .ends = in.ends, .complete = in.complete, .bound = in.bound },
            ),
        };
        try hcBranch(o, m, h, branch, inject, x, layer.mhc.scale, rows);
        const mixed2 = try hcPre(o, m, layer.mhc, x, rows, inject);
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].ahc.scale else m.mix.scale;
        if (fused(o, m) and d.streams == 4 and d.hidden == 2560 and hip.moe.defers(o, m.act, layer.moe, rows)) {
            const p = try hip.moe.pending(o, m.act, layer.moe, mixed2, rows);
            try o.l.fn_hc_moe_norm(@ptrFromInt(h.ptr), @ptrFromInt(p.y), @ptrFromInt(p.wts), @ptrFromInt(p.shared), @ptrFromInt(inject.ptr), @ptrFromInt(next), @ptrFromInt(x.ptr), @intCast(rows), @intCast(p.slots), d.eps, o.stream);
            continue;
        }
        const y = try hip.moe.run(o, m.act, layer.moe, mixed2, rows, !window);
        try hcBranch(o, m, h, y, inject, x, next, rows);
    }
    // a window's count moves in `Engine.round`, which a replay runs too: the launches change nothing on the host
    if (!window) for (segs) |g| {
        g.c.pooled = (g.pos + g.tokens.len) / d.index_ratio;
    };
    return .{ .mixed = try hcPre(o, m, m.mix, x, rows, null), .streams = h };
}

/// Streams a forward takes at once at most.
pub const max_segs = 64;
