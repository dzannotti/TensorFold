//! A prompt's sparse attention: the Loom indexer (launches/loom_attention.zig), the WMMA core, whole chunks a launch.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");

const Ops = hip.ops.Ops;

/// The most fp32 indexer scores a launch keeps (rows x blocks): past it the rows go in several launches.
const score_budget = 64 << 20;

/// One stream's `rows` prompt rows from `pos`, ends and blocks from `staged` if given; false where Loom does not fit.
pub fn attend(o: Ops, m: *const model.Model, d: model.Dims, kv: cache.SparseView, cap: usize, q32: u64, iq: u64, att: u64, rows: usize, pos: usize, host: *forward.Host, staged: ?mixers.Rows) !bool {
    const k = &o.l.attn;
    if (!o.prefill or m.act != .bf16 or !k.takes(d.heads, d.kv_heads, d.head_dim, d.index_heads, d.index_dim)) return false;
    const key_stride = d.index_top * d.index_ratio + d.index_ratio;
    const blocks_max = (pos + rows) / d.index_ratio;
    const per = @min(rows, @max(64, score_budget / @max(1, blocks_max)));
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(d.head_dim))));
    const a = std.heap.smp_allocator;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += per) {
        const n = @min(per, rows - r0);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const complete = try a.alloc(i32, n);
        defer a.free(complete);
        const ends = try a.alloc(i32, n);
        defer a.free(ends);
        var sparse_rows = false;
        for (complete, ends, 0..) |*c, *e, r| {
            e.* = @intCast(pos + r0 + r + 1);
            c.* = @divFloor(e.*, @as(i32, @intCast(d.index_ratio)));
            sparse_rows = sparse_rows or c.* > d.index_top;
        }
        const dev_complete = if (staged) |st| st.complete + r0 * 4 else try host.put(o, std.mem.sliceAsBytes(complete));
        const dev_ends = if (staged) |st| st.ends + r0 * 4 else try host.put(o, std.mem.sliceAsBytes(ends));
        var ids: u64 = 0;
        var count = dev_ends;
        var flags: u64 = 0;
        if (sparse_rows) {
            const nb: usize = @intCast(complete[n - 1]);
            const scores = try o.arena.of(f32, n * nb);
            const iq_rows: [*]const f32 = @ptrFromInt(iq + r0 * d.index_heads * d.index_dim * 4);
            try k.scores(o.l.r, iq_rows, @ptrFromInt(kv.pooled), @ptrFromInt(dev_complete), @ptrFromInt(scores), @intCast(n), @intCast(nb), @intCast(nb), @intCast(d.index_top), o.stream);
            ids = try o.arena.of(i32, n * key_stride);
            const counts = try o.arena.of(i32, n);
            flags = try o.arena.of(i32, n);
            try k.select(o.l.r, @ptrFromInt(scores), @intCast(nb), @ptrFromInt(dev_complete), @ptrFromInt(dev_ends), @ptrFromInt(ids), @intCast(key_stride), @ptrFromInt(counts), @ptrFromInt(flags), @intCast(n), @intCast(d.index_top), @intCast(d.index_ratio), o.stream);
            count = counts;
        } else {
            const zeros = try a.alloc(i32, n);
            defer a.free(zeros);
            @memset(zeros, 0);
            flags = try host.put(o, std.mem.sliceAsBytes(zeros));
            ids = flags;
        }
        // the hipcc twin of Loom's sparse_attend (same bytes, launches/flashnext.zig) runs faster on gfx1151
        try o.l.fn_sparse_attention(
            @ptrFromInt(q32 + r0 * d.heads * d.head_dim * 4),
            @ptrFromInt(kv.k),
            @ptrFromInt(kv.v),
            2,
            @ptrFromInt(ids),
            @intCast(key_stride),
            @ptrFromInt(count),
            @ptrFromInt(flags),
            null,
            null,
            @ptrFromInt(att + r0 * d.heads * d.head_dim * 4),
            @intCast(n),
            @intCast(d.heads),
            @intCast(d.kv_heads),
            @intCast(d.head_dim),
            @intCast(cap),
            scale,
            @intCast(hip.launches.loom_attention.parts),
            o.stream,
        );
    }
    return true;
}
