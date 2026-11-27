//! A prompt's sparse attention on the Loom kernels (launches/loom_attention.zig): hipcc's bytes, whole chunks a launch.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const mixers = @import("mixers.zig");

const Ops = hip.ops.Ops;

/// The fewest rows of a launch that take gufo's cohort kernel: below it the per-row kernels are faster.
const gufo_rows = 64;
/// tf_fn_idx_mask's most mask words a row.
const gufo_mask_words = 4096;

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
        const words = ((pos + r0 + n + d.index_ratio - 1) / d.index_ratio + 31) / 32;
        if (n >= gufo_rows and d.kv_heads == 2 and d.index_ratio == 4 and words <= gufo_mask_words) {
            const mask = try o.arena.of(u32, n * words);
            try o.l.fn_idx_mask(@ptrFromInt(ids), @intCast(key_stride), @ptrFromInt(count), @ptrFromInt(flags), @ptrFromInt(dev_complete), mask, @intCast(words), @intCast(d.index_ratio), @intCast(n), o.stream);
            try o.l.fn_gufo_attention(
                @ptrFromInt(q32 + r0 * d.heads * d.head_dim * 4),
                @ptrFromInt(kv.k),
                @ptrFromInt(kv.v),
                mask,
                @intCast(words),
                @ptrFromInt(att + r0 * d.heads * d.head_dim * 4),
                @intCast(pos + r0),
                @intCast(n),
                @intCast(d.kv_heads),
                @intCast(cap),
                o.stream,
            );
            continue;
        }
        try k.attend(
            o.l.r,
            @ptrFromInt(q32 + r0 * d.heads * d.head_dim * 4),
            @ptrFromInt(kv.k),
            @ptrFromInt(kv.v),
            @ptrFromInt(ids),
            @intCast(key_stride),
            @ptrFromInt(count),
            @ptrFromInt(flags),
            @ptrFromInt(att + r0 * d.heads * d.head_dim * 4),
            @intCast(n),
            @intCast(d.heads),
            @intCast(d.kv_heads),
            @intCast(cap),
            scale,
            o.stream,
        );
    }
    return true;
}
