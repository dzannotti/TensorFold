//! The DFlash2 drafter's launches (draft/draft.hip): tap rows gathered, rings written, dynamic conv, block attention.

const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const invalid = util.invalid;

pub const Fns = struct {
    gather: Function,
    ring_put: Function,
    norm_rope: Function,
    norm_rope_qk: Function,
    dyn_conv: Function,
    attn: Function,
    attn_merge: Function,
    bf16_rows: Function,
    q4_rows: Function,
    topk_merge: Function,
    /// The MLX products (draft.hip dft_mlx_tile): 4 or 8 waves a tile; 4 waves two tiles; 4 waves gate|up with silu.
    mlx: [Mlx.len]Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .gather = try m.function("tf_dft_gather"),
            .ring_put = try m.function("tf_dft_ring_put"),
            .norm_rope = try m.function("tf_dft_norm_rope"),
            .norm_rope_qk = try m.function("tf_dft_norm_rope_qk"),
            .dyn_conv = try m.function("tf_dft_dyn_conv"),
            .attn = try m.function("tf_dft_attn"),
            .attn_merge = try m.function("tf_dft_attn_merge"),
            .bf16_rows = try m.function("tf_dft_bf16_rows"),
            .q4_rows = try m.function("tf_dft_q4_rows"),
            .topk_merge = try m.function("tf_dft_topk_merge"),
            .mlx = .{ try m.function("tf_dft_mlx_w4c1"), try m.function("tf_dft_mlx_w8c1"), try m.function("tf_dft_mlx_w4c2"), try m.function("tf_dft_mlx_w4act"), try m.function("tf_dft_mlx_w4conv"), try m.function("tf_dft_mlx_w8conv"), try m.function("tf_dft_mlx_w2c1"), try m.function("tf_dft_mlx_w2conv"), try m.function("tf_dft_mlx_w4pair") },
        };
    }
};

/// The kernels' head width.
pub const head_dim = 128;
/// The queries one block attention workgroup takes: a stream's block rows times a key head's query heads.
pub const queries = 64;

/// draft.hip's DftRow: an absorbed row's features, its stream's rings and its position.
pub const Row = extern struct { src: u64, ring: u64, pos: i64 };

/// draft.hip's DftDesc: a held stream's rings, the first position its context keeps, its block's first position.
pub const Desc = extern struct { ring: u64, begin: i64, pos0: i64 };

/// `n` rows of `width` bytes (a multiple of 16) from rows[i].src into `out`, back to back.
pub fn gather(l: *const Launcher, rows: u64, n: usize, width: usize, out: u64, s: S) Error!void {
    if (width % 16 != 0) return invalid("dft gather");
    if (n == 0) return;
    var a: Args = .{};
    try a.add(rows);
    try a.add(@as(c_int, @intCast(width)));
    try a.add(out);
    try l.go(l.dft.gather, dim(n, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// Every layer's keys (normed with knorm (layers, 128) fp32, rotated) and values of `n` rows into their streams' rings.
pub fn ringPut(l: *const Launcher, kv: u64, stride: usize, rows: u64, n: usize, knorm: u64, layers: usize, kv_heads: usize, ring_rows: usize, eps: f32, theta: f32, s: S) Error!void {
    if (n == 0) return;
    var a: Args = .{};
    try a.add(kv);
    try a.add(@as(c_longlong, @intCast(stride)));
    try a.add(rows);
    try a.add(knorm);
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(@as(c_int, @intCast(ring_rows)));
    try a.add(eps);
    try a.add(theta);
    try l.go(l.dft.ring_put, dim(n, layers * kv_heads, 1), dim(32, 1, 1), 0, s, &a);
}

/// `rows` block rows' `heads` heads (x rows `stride` apart) normed and rotated at their streams' positions, packed.
pub fn normRope(l: *const Launcher, x: u64, stride: usize, weight: u64, out: u64, desc: u64, block: usize, rows: usize, heads: usize, eps: f32, theta: f32, s: S) Error!void {
    var a: Args = .{};
    try a.add(x);
    try a.add(@as(c_longlong, @intCast(stride)));
    try a.add(weight);
    try a.add(out);
    try a.add(desc);
    try a.add(@as(c_int, @intCast(block)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(eps);
    try a.add(theta);
    try l.go(l.dft.norm_rope, dim(rows * heads, 1, 1), dim(32, 1, 1), 0, s, &a);
}

/// normRope of the q heads (weight wq, into qo) and of the k heads `koff` elements on (wk, into ko), in one launch.
pub fn normRopeQk(l: *const Launcher, x: u64, stride: usize, koff: usize, wq: u64, wk: u64, qo: u64, ko: u64, desc: u64, block: usize, rows: usize, heads: usize, kv_heads: usize, eps: f32, theta: f32, s: S) Error!void {
    var a: Args = .{};
    try a.add(x);
    try a.add(@as(c_longlong, @intCast(stride)));
    try a.add(@as(c_longlong, @intCast(koff)));
    try a.add(wq);
    try a.add(wk);
    try a.add(qo);
    try a.add(ko);
    try a.add(desc);
    try a.add(@as(c_int, @intCast(block)));
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(kv_heads)));
    try a.add(eps);
    try a.add(theta);
    try l.go(l.dft.norm_rope_qk, dim(rows * (heads + kv_heads), 1, 1), dim(32, 1, 1), 0, s, &a);
}

/// The two-lag dynamic convolution of `rows` rows (`block` a stream), branch 0 or 1, `res` (0: none) added last.
pub fn dynConv(l: *const Launcher, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, width: usize, group: usize, branch: u1, block: usize, s: S) Error!void {
    if (width % group != 0) return invalid("dft dyn_conv");
    var a: Args = .{};
    try a.add(x);
    try a.add(dyn);
    try a.add(base);
    try a.add(res);
    try a.add(out);
    try a.add(@as(c_int, @intCast(width)));
    try a.add(@as(c_int, @intCast(group)));
    try a.add(@as(c_int, branch));
    try a.add(@as(c_int, @intCast(block)));
    try l.go(l.dft.dyn_conv, dim(rows, (width + 255) / 256, 1), dim(256, 1, 1), 0, s, &a);
}

/// What one block attention launch reads: a layer of each stream's rings by `desc`, and the scratch its splits write.
pub const Attn = struct {
    q: u64, // (rows, heads, 128) bf16
    kb: u64, // (rows, kv_heads, 128) bf16
    vb: u64, // rows `vstride` apart
    vstride: usize,
    desc: u64,
    layer: usize,
    streams: usize,
    block: usize,
    heads: usize,
    kv_heads: usize,
    ring_rows: usize,
    window: usize,
    splits: usize,
    part: u64, // partBytes
    out: u64, // (rows, heads, 128) bf16, rows `ostride` apart
    ostride: usize = 0, // elements (0: heads x 128)
};

/// Scratch the splits of one launch write: partial outputs then their (max, sum).
pub fn partBytes(streams: usize, kv_heads: usize, splits: usize) usize {
    return streams * kv_heads * splits * queries * (head_dim + 2) * 4;
}

pub fn attn(l: *const Launcher, x: Attn, s: S) Error!void {
    if (x.block * (x.heads / x.kv_heads) > queries or x.heads % x.kv_heads != 0 or x.ring_rows % 16 != 0 or x.splits == 0) return invalid("dft attn");
    const stats = x.part + x.streams * x.kv_heads * x.splits * queries * head_dim * 4;
    var a: Args = .{};
    try a.add(x.q);
    try a.add(x.kb);
    try a.add(x.vb);
    try a.add(@as(c_longlong, @intCast(x.vstride)));
    try a.add(x.desc);
    try a.add(@as(c_int, @intCast(x.layer)));
    try a.add(@as(c_int, @intCast(x.block)));
    try a.add(@as(c_int, @intCast(x.heads)));
    try a.add(@as(c_int, @intCast(x.kv_heads)));
    try a.add(@as(c_int, @intCast(x.ring_rows)));
    try a.add(@as(c_int, @intCast(x.window)));
    try a.add(x.part);
    try a.add(stats);
    try l.go(l.dft.attn, dim(x.streams * x.kv_heads, x.splits, 1), dim(2 * queries, 1, 1), 0, s, &a);
    var b: Args = .{};
    try b.add(x.part);
    try b.add(stats);
    try b.add(@as(c_int, @intCast(x.splits)));
    try b.add(@as(c_int, @intCast(x.block)));
    try b.add(@as(c_int, @intCast(x.heads)));
    try b.add(@as(c_int, @intCast(x.kv_heads)));
    try b.add(x.out);
    try b.add(@as(c_longlong, @intCast(if (x.ostride != 0) x.ostride else x.heads * head_dim)));
    try l.go(l.dft.attn_merge, dim(x.streams * x.kv_heads, queries, 1), dim(head_dim, 1, 1), 0, s, &b);
}

/// out (rows, n) = x (rows, k) times bf16 w (n, k) transposed, rounded to bf16; 64 rows a launch.
pub fn bf16Rows(l: *const Launcher, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, s: S) Error!void {
    if (k % 32 != 0 or n == 0) return invalid("dft bf16_rows");
    var at: usize = 0;
    while (at < rows) : (at += 64) {
        const m = @min(64, rows - at);
        var a: Args = .{};
        try a.add(x + at * k * 2);
        try a.add(w);
        try a.add(out + at * n * 2);
        try a.add(@as(c_int, @intCast(m)));
        try a.add(@as(c_int, @intCast(n)));
        try a.add(@as(c_int, @intCast(k)));
        try l.go(l.dft.bf16_rows, dim((n + 127) / 128, 1, 1), dim(128, 1, 1), 0, s, &a);
    }
}

/// out (rows, n) = x (rows, k) times an MLX affine-4 group-64 matrix with bf16 tables, rounded to bf16; 64 rows a launch.
pub fn q4Rows(l: *const Launcher, x: u64, words: u64, scales: u64, biases: u64, out: u64, rows: usize, n: usize, k: usize, s: S) Error!void {
    if (k % 64 != 0 or n == 0 or words % 16 != 0) return invalid("dft q4_rows");
    var at: usize = 0;
    while (at < rows) : (at += 64) {
        const m = @min(64, rows - at);
        var a: Args = .{};
        try a.add(x + at * k * 2);
        try a.add(words);
        try a.add(scales);
        try a.add(biases);
        try a.add(out + at * n * 2);
        try a.add(@as(c_int, @intCast(m)));
        try a.add(@as(c_int, @intCast(n)));
        try a.add(@as(c_int, @intCast(k)));
        try l.go(l.dft.q4_rows, dim((n + 31) / 32, 1, 1), dim(128, 1, 1), 0, s, &a);
    }
}

/// Each row's ks[r] best (value desc, id asc) from its `chunks` chunks' top k each (ci, cv: (rows, chunks, k)), ids
/// as the row's (chunk c's id + c x `chunk`), best first, `stride` apart.
pub fn topkMerge(l: *const Launcher, ci: u64, cv: u64, rows: usize, chunks: usize, k: usize, chunk: usize, ks: u64, stride: usize, ids: u64, values: u64, s: S) Error!void {
    if (chunks * k > 1024 or rows == 0) return invalid("dft topk merge");
    var a: Args = .{};
    try a.add(ci);
    try a.add(cv);
    try a.add(@as(c_int, @intCast(chunks)));
    try a.add(@as(c_int, @intCast(k)));
    try a.add(@as(c_int, @intCast(chunk)));
    try a.add(ks);
    try a.add(@as(c_int, @intCast(stride)));
    try a.add(ids);
    try a.add(values);
    try l.go(l.dft.topk_merge, dim(rows, 1, 1), dim(chunks * k, 1, 1), 0, s, &a);
}

/// draft.hip's DftMat: one MLX affine-4 group-64 matrix with bf16 tables and its bf16 output.
pub const Mat = extern struct { words: u64, scale: u64, bias: u64, out: u64, n: c_int, ld: c_int = 0 };

/// The MLX product tiles, by index into `Fns.mlx`.
pub const Mlx = enum(u4) {
    w4,
    w8,
    w4x2,
    act,
    /// w4 and w8 with dynConv's branch 1 and residual on the rounded product (`Conv`)
    w4conv,
    w8conv,
    w2,
    w2conv,
    /// w4 over a's tiles then b's (two products of the same x)
    pair,
    pub const len = 9;
};

/// The dynamic convolution a `.w4conv` or `.w8conv` product ends with: dynConv(product, dyn, base, res) on branch 1.
pub const Conv = struct { dyn: u64 = 0, base: u64 = 0, res: u64 = 0, group: usize = 1, block: usize = 1 };

/// out (m <= 16 rows, a.n) = x (m, k) times `a`, rounded to bf16, the verify split tile's bits; with `.act`, a.out =
/// silu(x a) * (x b) as tf_silu_mul rounds them; with a conv variant, `cv` after. k a multiple of 256.
pub fn mlx(l: *const Launcher, v: Mlx, x: u64, m: usize, k: usize, ldx: usize, a: Mat, b: Mat, cv: Conv, s: S) Error!void {
    if (m == 0 or m > 16 or k % 256 != 0 or a.n <= 0 or (v == .act and b.n != a.n)) return invalid("dft mlx");
    if ((v == .w4conv or v == .w8conv or v == .w2conv) and (cv.res == 0 or @as(usize, @intCast(a.n)) % cv.group != 0)) return invalid("dft mlx conv");
    var p: Args = .{};
    try p.add(extern struct { x: u64, m: c_int, k: c_int, a: Mat, b: Mat, dyn: u64, base: u64, res: u64, group: c_int, block: c_int, ldx: c_int }{ .ldx = @intCast(ldx), .x = x, .m = @intCast(m), .k = @intCast(k), .a = a, .b = b, .dyn = cv.dyn, .base = cv.base, .res = cv.res, .group = @intCast(cv.group), .block = @intCast(cv.block) });
    const n: usize = @intCast(a.n);
    const blocks = switch (v) {
        .w4x2 => (n + 31) / 32,
        .pair => (n + 15) / 16 + (@as(usize, @intCast(b.n)) + 15) / 16,
        else => (n + 15) / 16,
    };
    const waves: usize = switch (v) {
        .w8, .w8conv => 8,
        .w2, .w2conv => 2,
        else => 4,
    };
    try l.go(l.dft.mlx[@intFromEnum(v)], dim(blocks, 1, 1), dim(32 * waves, 1, 1), 0, s, &p);
}
