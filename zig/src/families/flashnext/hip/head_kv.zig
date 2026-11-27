//! The head's sparse layer on prompt rows as CUDA's lean prefill (cuda_batch_cache.zig): K/V and indexer keys only.
const hip = @import("hip");
const model = @import("model.zig");
const forward = @import("forward.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;

/// Each segment's roped keys, values, raw indexer keys and pooled blocks from `x`; no queries, attention or output.
pub fn store(o: Ops, m: *const model.Model, d: model.Dims, s: model.Sparse, segs: []const forward.Seg, x: Tensor, rows: usize, dev_pos: u64) !void {
    const hd = d.head_dim;
    const di = d.index_dim;
    const width = (d.index_heads + 1) * di;
    const kind = @backingInt(m.act);
    var outs: [3]Tensor = undefined;
    if (!try o.projectGroup(x, &.{ s.k, s.v, s.index }, rows, &outs)) {
        for (&outs, [_]model.Projection{ s.k, s.v, s.index }) |*t, p| t.* = try o.project(x, p, rows, false);
    }
    const keys, const values, const idx = outs;
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const kv = g.c.sparse(0);
        const cap = g.c.layout.cap;
        const pos = g.pos;
        const at_pos = dev_pos + off * 4;
        try o.qkRope(forward.at(keys, off * d.kv_heads * hd), d.kv_heads * hd, hd, s.k_norm, d.eps, n, d.kv_heads, hd, d.rotary, d.theta, at_pos, null, kv.k, cap);
        try o.kvWrite(forward.at(values, off * d.kv_heads * hd), kv.v, n, d.kv_heads, hd, cap, pos);
        try o.copyCols(forward.at(idx, off * width), width, d.index_heads * di, kv.raw + pos * di * m.act.size(), n, di);
        const blocks = (pos + n) / d.index_ratio;
        const pooled = g.c.pooled;
        if (blocks > pooled) try o.l.fn_idx_pool(
            @ptrFromInt(kv.raw),
            @ptrFromInt(kv.pooled + pooled * di * m.act.size()),
            @ptrFromInt(s.ik_norm),
            kind,
            @intCast(pooled),
            @intCast(blocks - pooled),
            @intCast(di),
            @intCast(d.index_ratio),
            @intCast(d.rotary),
            d.theta,
            d.eps,
            o.stream,
        );
        off += n;
    }
}

