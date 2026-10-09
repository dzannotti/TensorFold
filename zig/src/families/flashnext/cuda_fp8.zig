//! 128x128-block FP8 linears (ModelOpt FP8_PB_WO: e4m3 `weight` + fp32 `weight_scale_inv`) on CUDA, as the Python
//! engine runs them (cuda/nvfp4/linear.py Fp8BlockLinear): e4m3 bytes in the FP8 GEMM's fragment order, an fp32
//! scale per (64 inputs, output column), and the lane matmul in mode FP8G (qmmf.cu) for decode and prompt rows
//! alike: each 64 inputs' bf16 MMA products scaled into an fp32 sum in block order, K slices fixed by the shape, a
//! row's bits independent of every other row and of M. zig/kernels/cuda/fn_qmmf.cu is the device code copied by
//! zig/tests/cuda/copies.py (SASS-equal to the Python extension tensorfold_nvfp4_v3); `matmul` is linear._matmul and
//! qmmf_cuda's host logic. The host layout helpers make the loader's buffers byte for byte as from_checkpoint does.
//! HIP (gfx1151): zig/kernels/hip/fn_qmmf{,_ld}.hip, the same layouts and arguments, slices added in the block;
//! from `wide_rows` rows the LDS-staged wide tile (qmmw_kernel), each row's arithmetic unchanged: the same bits.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold, cuda/nvfp4/linear.py and qmmf.cu, authored by
//! Ash Hart (ashhart)).
const std = @import("std");
const cuda = @import("cuda");

/// Fp8BlockLinear: w8 uint8 [npad/64][K/64][8][32][2][8] (fragment order), bs fp32 [npad/64][K/64][64].
pub const Linear = struct { w8: u64 = 0, bs: u64 = 0, n: u32 = 0, k: u32 = 0, npad: u32 = 0 };

/// linear.py FUSED_ROWS (TF_QMMF_FUSED_ROWS default): rows from which a block sums its tile's K slices itself.
pub const fused_rows: usize = 256;

/// Rows of output padding Fp8Linear / Fp8BlockLinear take: ceil(n / 128) * 128.
pub fn npadOf(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// qmm.split_k(n, k, gs=64, target=192): K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// A HIP build (the runtime's `cuda.hip`): fn_qmmf.hip's launch shapes, no clusters.
const hip_build = cuda.hip;

/// zig/kernels/hip/fn_qmmf.hip on gfx1151: no clusters; a block holds up to 4 K slices side by side (512 threads)
/// and adds them in LDS in slice order (the cluster's sum without the cluster).
pub const hip_slices: usize = 4;

/// K slices on HIP: split_k capped at the slices a block holds; still a function of the shape alone.
pub fn splitKHip(n: usize, k: usize) usize {
    return @min(splitK(n, k), hip_slices);
}

/// HIP: rows from which the wide tile (64 or 128 rows, 128 columns or 64 with K slices) replaces the 16/32-row tiles.
pub const wide_rows: usize = 33;

/// HIP's wide tile for `m` rows and `sk` slices: (rows, columns); rows 64 up to 64, else 128.
pub fn wideTile(m: usize, sk: usize) [2]usize {
    return .{ if (m <= 64) 64 else 128, if (sk > 1) 64 else 128 };
}

/// qmm.bucket: the row tile.
pub fn bucket(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

/// qmmf.cu Tile<FP8G, BM, 64, 1, 4, 4>::SMEM: max(4 stages, the cluster partials).
pub fn smem(bm: usize) u32 {
    const stage = std.mem.alignForward(usize, bm * 128 + 64 * 64 + 64 * 4, 128);
    const partials = (bm / 16) * 2 * 4 * 128 * 4;
    return @intCast(@max(4 * stage, partials));
}

/// qmmf.cu launch's L2 band: row tiles whose inputs fill about 12 MB.
pub fn l2Group(rows_t: usize, bm: usize, k: usize) usize {
    return @max(1, @min(rows_t, (12 << 20) / (bm * k * 2)));
}

/// HIP's wide-tile band: 4 row tiles (measured 2-7% faster than l2Group's 12 MB band at 512-2048 rows, inputs and
/// weights cold, notes/prefill.md); tile order only, no bits.
pub fn wideGroup(rows_t: usize) usize {
    return @min(rows_t, 4);
}

/// Mangled names of the FP8G instantiations (fn_qmmf.cu's footer): [F32][cluster][bm 16, 32, 64], and the fused [F32].
pub const sym = struct {
    fn name(comptime bm: u32, comptime f32_out: bool, comptime cluster: bool, comptime fuse: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10tf_fn_qmmf11qmmf_kernelILi3ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ bm, @intFromBool(f32_out), @intFromBool(cluster), @intFromBool(fuse) });
    }
    pub const tiled = blk: {
        var out: [2][2][3][:0]const u8 = undefined;
        for (0..2) |f| for (0..2) |c| for (0..3) |b| {
            out[f][c][b] = name(16 << b, f == 1, c == 1, false);
        };
        break :blk out;
    };
    pub const fused = [2][:0]const u8{ name(64, false, false, true), name(64, true, false, true) };
    /// fn_qmmf_ld.cu's (tools/zig/flashnext_qmmf_ld.py): bf16 out with a row stride, [cluster][bm 16, 32, 64] and fused
    fn ldName(comptime bm: u32, comptime cluster: bool, comptime fuse: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN13tf_fn_qmmf_ld11qmmf_kernelILi3ELi{d}ELi64ELi1ELi4ELi4ELb0ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiiii", .{ bm, @intFromBool(cluster), @intFromBool(fuse) });
    }
    pub const ld_tiled = blk: {
        var out: [2][3][:0]const u8 = undefined;
        for (0..2) |c| for (0..3) |b| {
            out[c][b] = ldName(16 << b, c == 1, false);
        };
        break :blk out;
    };
    pub const ld_fused = ldName(64, false, true);
    /// fn_qmmf.hip's wide tile, [F32][64x128, 64x64, 128x128, 128x64] (HIP builds only)
    fn wideName(comptime ns: []const u8, comptime bm: u32, comptime bn: u32, comptime f32_out: bool, comptime tail: []const u8) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN{s}11qmmw_kernelILi{d}ELi{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii{s}", .{ ns, bm, bn, @intFromBool(f32_out), tail });
    }
    pub const wide = blk: {
        var out: [2][4][:0]const u8 = undefined;
        for (0..2) |f| for (0..4) |i| {
            out[f][i] = wideName("10tf_fn_qmmf", 64 << (i / 2), 128 >> (i % 2), f == 1, "");
        };
        break :blk out;
    };
    /// fn_qmmf.hip's SwiGLU tiles (16, 32 rows; HIP builds only): the shared expert's interleaved gate|up into its act
    pub const swiglu = [2][:0]const u8{ "_ZN10tf_fn_qmmf18qmmf_swiglu_kernelILi16EEEvPK13__nv_bfloat16PKhS5_fPviiiiii", "_ZN10tf_fn_qmmf18qmmf_swiglu_kernelILi32EEEvPK13__nv_bfloat16PKhS5_fPviiiiii" };
    pub const ld_wide = blk: {
        var out: [4][:0]const u8 = undefined;
        for (0..4) |i| out[i] = wideName("13tf_fn_qmmf_ld", 64 << (i / 2), 128 >> (i % 2), false, "i");
        break :blk out;
    };
};

pub const Kernels = struct {
    module: cuda.Module,
    tiled: [2][2][3]cuda.Function, // [bf16, fp32][one slice, cluster][16, 32, 64]
    fused: [2]cuda.Function,
    /// fn_qmmf_ld: the same kernels writing bf16 rows `ldo` apart
    ld_module: cuda.Module,
    ld_tiled: [2][3]cuda.Function,
    ld_fused: cuda.Function,
    /// HIP: the wide tile, [bf16, fp32][wideIndex] and its `ldo` form
    wide: [2][4]cuda.Function = undefined,
    ld_wide: [4]cuda.Function = undefined,
    /// HIP: `matmulSwiglu`'s 16- and 32-row tiles (null: a build without them)
    swiglu: [2]?cuda.Function = .{ null, null },
    major: c_int,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.module = try cuda.Module.load(ctx.d, cuda.kernels.fn_qmmf);
        errdefer k.module.unload();
        // HIP: no cluster instantiations (their slots repeat the one-slice kernels) and static LDS only
        for (0..2) |f| {
            for (0..2) |c| for (0..3) |b| {
                k.tiled[f][c][b] = try k.module.function(sym.tiled[f][if (hip_build) 0 else c][b]);
                if (!hip_build) try k.tiled[f][c][b].allowDynamicShared(smem(@as(usize, 16) << @intCast(b)));
            };
            k.fused[f] = try k.module.function(sym.fused[f]);
            if (!hip_build) try k.fused[f].allowDynamicShared(smem(64));
        }
        k.ld_module = try cuda.Module.load(ctx.d, cuda.kernels.fn_qmmf_ld);
        errdefer k.ld_module.unload();
        for (0..2) |c| for (0..3) |b| {
            k.ld_tiled[c][b] = try k.ld_module.function(sym.ld_tiled[if (hip_build) 0 else c][b]);
            if (!hip_build) try k.ld_tiled[c][b].allowDynamicShared(smem(@as(usize, 16) << @intCast(b)));
        };
        k.ld_fused = try k.ld_module.function(sym.ld_fused);
        if (!hip_build) try k.ld_fused.allowDynamicShared(smem(64));
        if (hip_build) for (0..4) |i| {
            for (0..2) |f| k.wide[f][i] = try k.module.function(sym.wide[f][i]);
            k.ld_wide[i] = try k.ld_module.function(sym.ld_wide[i]);
        };
        k.swiglu = .{ null, null };
        if (hip_build) for (0..2) |i| {
            k.swiglu[i] = k.module.function(sym.swiglu[i]) catch null;
        };
        k.major = try ctx.attribute(.compute_capability_major);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.ld_module.unload();
        k.module.unload();
    }
};

fn bucketIndex(bm: usize) usize {
    return switch (bm) {
        16 => 0,
        32 => 1,
        else => 2,
    };
}

/// linear._matmul(FP8G, ...) and qmmf_cuda: x (m, K) bf16 rows `x_stride` elements apart -> out (m, n) bf16 (fp32
/// with `f32_out`), contiguous. K slices from the shape alone (split_k), added in a cluster (sm_90 on) or, from
/// FUSED_ROWS rows, by each block in slice order: the same bits either way, a row's bits never depend on m.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize) !void {
    return matmulAt(k, s, x, x_stride, l, out, f32_out, m, null);
}

/// `matmul`, bf16 out, the rows `ldo` elements apart (fn_qmmf_ld: the same kernels and values, only the store
/// address differs; a projection's block-FP8 columns written straight into its wider rows).
pub fn matmulLd(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, ldo: usize, m: usize) !void {
    if (ldo < l.n or ldo % 2 != 0 or out % 4 != 0) return error.Invalid;
    return matmulAt(k, s, x, x_stride, l, out, false, m, ldo);
}

fn matmulAt(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize, ldo: ?usize) !void {
    if (m < 1) return error.Invalid;
    const n: usize = l.n;
    const kk: usize = l.k;
    if (kk % 64 != 0 or l.npad < n) return error.Invalid;
    const sk = if (hip_build) splitKHip(n, kk) else splitK(n, kk);
    if (hip_build and m >= wide_rows and l.npad % 128 == 0) return wide(k, s, x, x_stride, l, out, f32_out, m, ldo, sk);
    const bm: usize = if (sk > 1 and m >= fused_rows) 0 else bucket(m);
    const fused = bm == 0;
    const cluster = !hip_build and !fused and sk > 1 and sk <= 8 and k.major >= 9;
    // sm_89 and older would add slices in the reduce (part + reduce_kernel): never on GB10, not ported; HIP adds
    // them in the block
    if (!hip_build and !fused and sk > 1 and !cluster) return error.SplitWithoutCluster;
    const tile: usize = if (fused) 64 else bm;
    const rows_t = (m + tile - 1) / tile;
    const f = if (ldo != null) (if (fused) k.ld_fused else k.ld_tiled[@intFromBool(cluster)][bucketIndex(bm)]) else if (fused) k.fused[@intFromBool(f32_out)] else k.tiled[@intFromBool(f32_out)][@intFromBool(cluster)][bucketIndex(bm)];
    var a: cuda.Args = .{};
    a.add(x);
    a.add(l.w8);
    a.add(l.bs);
    a.add(@as(f32, 1.0));
    a.add(out);
    a.add(@as(u64, 0));
    for ([_]usize{ m, n, kk, sk, l.npad, if (m == 1) kk else x_stride, l2Group(rows_t, tile, kk) }) |v| a.add(@as(c_int, @intCast(v)));
    if (ldo) |ld| a.add(@as(c_int, @intCast(ld)));
    // HIP: one block a tile, its K slices side by side (128 threads each), static LDS
    const z: u32 = if (fused or hip_build) 1 else @intCast(sk);
    try cuda.launch.launch(f, .{
        .grid = .{ .x = @intCast(rows_t * ((n + 63) / 64)), .y = 1, .z = z },
        .block = .{ .x = if (hip_build and !fused) @intCast(128 * sk) else 128 },
        .shared = if (hip_build) 0 else smem(tile),
        .cluster = if (cluster) .{ .x = 1, .y = 1, .z = @intCast(sk) } else null,
    }, s, &a);
}

/// The rows `matmulSwiglu` takes (its 16- and 32-row tiles; wider windows run `matmul` and fn_ops' interleaved SwiGLU).
pub fn swigluRows(k: *const Kernels, m: usize) bool {
    return hip_build and m >= 1 and m < wide_rows and k.swiglu[bucketIndex(bucket(m))] != null;
}

/// The shared expert's gate|up `l` (its rows interleaved: 32 gate rows, then the same 32 up rows, a 64-row tile;
/// weights.interleave) with fn_ops' SwiGLU in the epilogue: act [m, n / 2] bf16 = the bytes `matmul` then
/// tf_fn_shared_swiglu give on the plain rows. `swigluRows(m)` first.
pub fn matmulSwiglu(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, act: u64, m: usize) !void {
    if (!swigluRows(k, m) or l.n % 64 != 0 or l.k % 64 != 0) return error.Invalid;
    const sk = splitKHip(l.n, l.k);
    const bm = bucket(m);
    const rows_t = (m + bm - 1) / bm;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(l.w8);
    a.add(l.bs);
    a.add(@as(f32, 1.0));
    a.add(act);
    for ([_]usize{ m, l.n, l.k, sk, if (m == 1) l.k else x_stride, l2Group(rows_t, bm, l.k) }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(k.swiglu[bucketIndex(bm)].?, .{ .grid = .{ .x = @intCast(rows_t * (l.n / 64)), .y = 1, .z = 1 }, .block = .{ .x = @intCast(128 * sk) } }, s, &a);
}

/// HIP's wide tile: a block of 2 * rows threads, static LDS, the L2 band over its row tiles.
fn wide(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, l: Linear, out: u64, f32_out: bool, m: usize, ldo: ?usize, sk: usize) !void {
    const t = wideTile(m, sk);
    const i = @as(usize, if (t[0] == 128) 2 else 0) + @intFromBool(t[1] == 64);
    const rows_t = (m + t[0] - 1) / t[0];
    var a: cuda.Args = .{};
    a.add(x);
    a.add(l.w8);
    a.add(l.bs);
    a.add(@as(f32, 1.0));
    a.add(out);
    a.add(@as(u64, 0));
    for ([_]usize{ m, l.n, l.k, sk, l.npad, x_stride, wideGroup(rows_t) }) |v| a.add(@as(c_int, @intCast(v)));
    if (ldo) |ld| a.add(@as(c_int, @intCast(ld)));
    try cuda.launch.launch(if (ldo != null) k.ld_wide[i] else k.wide[@intFromBool(f32_out)][i], .{
        .grid = .{ .x = @intCast(rows_t * ((l.n + t[1] - 1) / t[1])), .y = 1, .z = 1 },
        .block = .{ .x = @intCast(2 * t[0]) },
    }, s, &a);
}

// ---- host layouts (Fp8BlockLinear.from_checkpoint) ------------------------------------------------------------------

/// Fp8BlockLinear.column_scales: fp32 `scale_inv` [ceil(n/128), k/128] -> `out` [n, k/64], each (row, 64 inputs)'s
/// block scale.
pub fn columnScales(scale_inv: []const f32, n: usize, k: usize, out: []f32) !void {
    if (k % 128 != 0 or scale_inv.len != ((n + 127) / 128) * (k / 128) or out.len != n * (k / 64)) return error.BadScaleShape;
    const kb = k / 128;
    for (0..n) |r| for (0..k / 64) |j| {
        out[r * (k / 64) + j] = scale_inv[(r / 128) * kb + j / 2];
    };
}

/// The input offset within a 32-input step of fragment byte `byte` of lane `lane` (fragment_index's `kin`).
pub fn kin(lane: usize, byte: usize) usize {
    return 16 * (byte / 4) + 2 * (lane % 4) + (byte % 2) + 8 * ((byte % 4) / 2);
}

/// _fragment_order: e4m3 bytes [n, k] (rows past n zero up to npad) -> [npad/64][k/64][8][32][2][8].
pub fn fragmentOrder(codes: []const u8, n: usize, k: usize, npad: usize, out: []u8) !void {
    if (k % 64 != 0 or npad % 64 != 0 or npad < n or codes.len != n * k or out.len != npad * k) return error.BadFp8Shape;
    var i: usize = 0;
    for (0..npad / 64) |a| for (0..k / 64) |b| for (0..8) |c| for (0..32) |lane| for (0..2) |h| for (0..8) |byte| {
        const row = a * 64 + c * 8 + lane / 4;
        const col = b * 64 + h * 32 + kin(lane, byte);
        out[i] = if (row < n) codes[row * k + col] else 0;
        i += 1;
    };
}

/// from_rows' bs: fp32 column scales [n, k/64] (ones for rows past n up to npad) -> [npad/64][k/64][64].
pub fn tileScales(cols: []const f32, n: usize, k: usize, npad: usize, out: []f32) !void {
    const kg = k / 64;
    if (k % 64 != 0 or npad % 64 != 0 or npad < n or cols.len != n * kg or out.len != npad * kg) return error.BadFp8Shape;
    for (0..npad / 64) |a| for (0..kg) |g| for (0..64) |r| {
        const row = a * 64 + r;
        out[(a * kg + g) * 64 + r] = if (row < n) cols[row * kg + g] else 1.0;
    };
}

test "split_k, buckets and shared memory follow qmm.py and qmmf.cu" {
    // the INT4-AutoRound checkpoint's block-FP8 shapes, one GPU and a TP=2 rank
    try std.testing.expectEqual(@as(usize, 1), splitK(16384, 2560));
    try std.testing.expectEqual(@as(usize, 1), splitK(13312, 2560));
    try std.testing.expectEqual(@as(usize, 8), splitK(2560, 6144));
    try std.testing.expectEqual(@as(usize, 4), splitK(2560, 2560));
    try std.testing.expectEqual(@as(usize, 2), splitK(2560, 1280));
    try std.testing.expectEqual(@as(usize, 4), splitK(2560, 3072));
    try std.testing.expectEqual(@as(usize, 1), splitK(2560, 640));
    try std.testing.expectEqual(@as(usize, 4), splitK(1280, 2560));
    try std.testing.expectEqual(@as(usize, 4), splitKHip(2560, 6144));
    try std.testing.expectEqual(@as(usize, 2), splitKHip(2560, 1280));
    try std.testing.expectEqual(@as(usize, 1), splitKHip(16384, 2560));
    try std.testing.expectEqual(@as(usize, 25600), smem(16));
    try std.testing.expectEqual(@as(usize, 33792), smem(32));
    try std.testing.expectEqual(@as(usize, 50176), smem(64));
    try std.testing.expectEqual(@as(usize, 13312), npadOf(13312));
    try std.testing.expectEqual(@as(usize, 128), npadOf(96));
    try std.testing.expectEqual(@as(usize, 64), bucket(33));
    try std.testing.expectEqual(@as(usize, 1), l2Group(1, 16, 2560));
    try std.testing.expectEqual(@as(usize, 38), l2Group(64, 64, 2560));
    try std.testing.expectEqual(@as(usize, 4), wideGroup(16));
    try std.testing.expectEqual(@as(usize, 1), wideGroup(1));
    try std.testing.expectEqual([2]usize{ 64, 128 }, wideTile(33, 1));
    try std.testing.expectEqual([2]usize{ 128, 64 }, wideTile(65, 4));
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmw_kernelILi128ELi64ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.wide[1][3]);
    try std.testing.expectEqualStrings("_ZN13tf_fn_qmmf_ld11qmmw_kernelILi64ELi128ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiiii", sym.ld_wide[0]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.tiled[1][1][1]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_qmmf11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", sym.fused[0]);
}

test "fragment order is fragment_index's (k, n) of every byte" {
    // fragment_index: p = 16 (byte / 4) + 4 (lane % 4) + byte % 4; kin = 16 (p / 16) + 2 ((p % 16) / 4) + p % 2 + 8 ((p % 4) / 2)
    for (0..32) |lane| for (0..8) |byte| {
        const p = 16 * (byte / 4) + 4 * (lane % 4) + (byte % 4);
        try std.testing.expectEqual(16 * (p / 16) + 2 * ((p % 16) / 4) + (p % 2) + 8 * ((p % 4) / 2), kin(lane, byte));
    };
    // each 32-input step's lane bytes cover 32 inputs of 8 rows once (a permutation)
    for (0..4) |q| {
        var seen: [32]bool = @splat(false);
        for (0..8) |byte| {
            const v = kin(q, byte);
            try std.testing.expect(!seen[v]);
            seen[v] = true;
        }
        for (0..8) |byte| try std.testing.expectEqual(kin(q, byte), kin(q + 4 * 5, byte));
    }
    // hand cases: lane 0 byte 0 -> 0, byte 1 -> 1, byte 2 -> 8, byte 3 -> 9, byte 4 -> 16; lane 1 byte 0 -> 2
    try std.testing.expectEqual(@as(usize, 0), kin(0, 0));
    try std.testing.expectEqual(@as(usize, 1), kin(0, 1));
    try std.testing.expectEqual(@as(usize, 8), kin(0, 2));
    try std.testing.expectEqual(@as(usize, 9), kin(0, 3));
    try std.testing.expectEqual(@as(usize, 16), kin(0, 4));
    try std.testing.expectEqual(@as(usize, 2), kin(1, 0));
    try std.testing.expectEqual(@as(usize, 31), kin(7, 7)); // quad 3, byte 7: 16 + 6 + 1 + 8
}

test "fragment order and tile scales on a small matrix" {
    const gpa = std.testing.allocator;
    const n = 70;
    const k = 128;
    const npad = npadOf(n);
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @intCast((i * 7 + 3) % 251);
    const out = try gpa.alloc(u8, npad * k);
    defer gpa.free(out);
    try fragmentOrder(codes, n, k, npad, out);
    // byte (a, b, c, lane, h, byte) at ((((a * kg + b) * 8 + c) * 32 + lane) * 2 + h) * 8 + byte
    const at = struct {
        fn f(a: usize, b: usize, c: usize, lane: usize, h: usize, byte: usize) usize {
            return ((((a * (k / 64) + b) * 8 + c) * 32 + lane) * 2 + h) * 8 + byte;
        }
    }.f;
    try std.testing.expectEqual(codes[0], out[at(0, 0, 0, 0, 0, 0)]);
    try std.testing.expectEqual(codes[1 * k + 64 + 32 + 9], out[at(0, 1, 0, 4, 1, 3)]); // row 1, col 64 + 32 + kin(4, 3) = 9
    try std.testing.expectEqual(codes[69 * k + 0], out[at(1, 0, 0, 20, 0, 0)]); // row 64 + 5 = 69, lane 20: quad 0, kin 0
    try std.testing.expectEqual(@as(u8, 0), out[at(1, 0, 0, 24, 0, 0)]); // row 70: padding
    var inv = [_]f32{ 0.5, 2.0 }; // [ceil(70/128) = 1, 128/128 = 1] ... one block row, one block column
    var cols: [n * 2]f32 = undefined;
    try std.testing.expectError(error.BadScaleShape, columnScales(&inv, n, k, &cols));
    try columnScales(inv[0..1], n, k, &cols);
    for (cols) |v| try std.testing.expectEqual(@as(f32, 0.5), v);
    cols[69 * 2 + 1] = 3;
    var bs: [128 * 2]f32 = undefined;
    try tileScales(&cols, n, k, npad, &bs);
    try std.testing.expectEqual(@as(f32, 3), bs[(1 * 2 + 1) * 64 + 5]);
    try std.testing.expectEqual(@as(f32, 1), bs[(1 * 2 + 0) * 64 + 6]); // row 70: padding scales are one
    try std.testing.expectEqual(@as(f32, 0.5), bs[(0 * 2 + 1) * 64 + 63]);
}

test "column scales repeat 128 rows and two 64-input groups" {
    var inv = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]: n 200 -> 2 block rows, k 384 -> 3 block columns
    var cols: [200 * 6]f32 = undefined;
    try columnScales(&inv, 200, 384, &cols);
    try std.testing.expectEqual(@as(f32, 1), cols[0]);
    try std.testing.expectEqual(@as(f32, 1), cols[1]);
    try std.testing.expectEqual(@as(f32, 2), cols[2]);
    try std.testing.expectEqual(@as(f32, 3), cols[127 * 6 + 5]);
    try std.testing.expectEqual(@as(f32, 4), cols[128 * 6 + 0]);
    try std.testing.expectEqual(@as(f32, 6), cols[199 * 6 + 4]);
}

// ---- fp8-check: the port against Python's Fp8BlockLinear (tools/zig/check_fp8block.py's cases) ---------------------

fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, suffix: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}{s}", .{ dir, name, suffix });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
}

fn hex(bytes: []const u8) [64]u8 {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
    return std.fmt.bytesToHex(sum, .lower);
}

/// Every case of `dir` (cases.json + each case's codes, scale_inv and input rows): the host layouts' sha256 against
/// Python's, then `matmul` at every row count (bf16, and fp32 where Python wrote it) against Python's outputs, a
/// row's bits against its one-row call, and the time at 1 and 4096 rows. True when every byte agrees.
pub fn check(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !bool {
    const d = ctx.d;
    var k = try Kernels.load(ctx);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const text = try readFile(gpa, io, dir, "cases", ".json");
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var ok = true;
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    for (parsed.value.object.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const name = c.get("name").?.string;
        const n: usize = @intCast(c.get("n").?.integer);
        const kk: usize = @intCast(c.get("k").?.integer);
        const npad = npadOf(n);
        const codes = try readFile(gpa, io, dir, name, "_codes.bin");
        defer gpa.free(codes);
        const inv_b = try readFile(gpa, io, dir, name, "_inv.bin");
        defer gpa.free(inv_b);
        const xb = try readFile(gpa, io, dir, name, "_x.bin");
        defer gpa.free(xb);
        const inv = try gpa.alloc(f32, inv_b.len / 4);
        defer gpa.free(inv);
        @memcpy(std.mem.sliceAsBytes(inv), inv_b);
        // the layouts
        const cols = try gpa.alloc(f32, n * (kk / 64));
        defer gpa.free(cols);
        try columnScales(inv, n, kk, cols);
        const bs = try gpa.alloc(f32, npad * (kk / 64));
        defer gpa.free(bs);
        try tileScales(cols, n, kk, npad, bs);
        const w8 = try gpa.alloc(u8, npad * kk);
        defer gpa.free(w8);
        try fragmentOrder(codes, n, kk, npad, w8);
        const w8_ok = std.mem.eql(u8, &hex(w8), c.get("w8").?.string);
        const bs_ok = std.mem.eql(u8, &hex(std.mem.sliceAsBytes(bs)), c.get("bs").?.string);
        ok = ok and w8_ok and bs_ok;
        var dw = try cuda.DeviceBuffer.fromHost(d, w8);
        defer dw.free();
        var dbs = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(bs));
        defer dbs.free();
        var dx = try cuda.DeviceBuffer.fromHost(d, xb);
        defer dx.free();
        const max_m = xb.len / (2 * kk);
        var dy = try cuda.DeviceBuffer.alloc(d, max_m * n * 4);
        defer dy.free();
        var dy1 = try cuda.DeviceBuffer.alloc(d, n * 4);
        defer dy1.free();
        const host = try gpa.alloc(u8, max_m * n * 4);
        defer gpa.free(host);
        const one = try gpa.alloc(u8, n * 4);
        defer gpa.free(one);
        const l: Linear = .{ .w8 = dw.ptr, .bs = dbs.ptr, .n = @intCast(n), .k = @intCast(kk), .npad = @intCast(npad) };
        var equal: usize = 0;
        var total: usize = 0;
        var inv_ok = true;
        for ([_][]const u8{ "bf16", "fp32" }) |kind| {
            const f32_out = kind[0] == 'f';
            const es: usize = if (f32_out) 4 else 2;
            var it = c.get(kind).?.object.iterator();
            while (it.next()) |kv| {
                const m = try std.fmt.parseInt(usize, kv.key_ptr.*, 10);
                try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, f32_out, m);
                try stream.synchronize();
                try dy.download(0, host[0 .. m * n * es]);
                total += 1;
                // "" (a ROCm generator run: no CUDA qmmf there): no reference bits; layouts, invariance and Ld still
                // count, the numerics are tools/rocm/fp8_check.py's
                if (kv.value_ptr.string.len == 0) {
                    total -= 1;
                } else if (std.mem.eql(u8, &hex(host[0 .. m * n * es]), kv.value_ptr.string)) {
                    equal += 1;
                } else {
                    ok = false;
                    std.debug.print("  {s} {s} m {d}: DIFFER\n", .{ name, kind, m });
                }
                // row invariance: rows of this call against one-row calls
                for ([_]usize{ 0, 1, m / 2, m - 1 }) |r| if (r < m) {
                    try matmul(&k, stream, dx.ptr + r * kk * 2, kk, l, dy1.ptr, f32_out, 1);
                    try stream.synchronize();
                    try dy1.download(0, one[0 .. n * es]);
                    if (!std.mem.eql(u8, one[0 .. n * es], host[r * n * es ..][0 .. n * es])) {
                        inv_ok = false;
                        std.debug.print("  {s} {s} m {d} row {d}: differs from its one-row call\n", .{ name, kind, m, r });
                    }
                };
            }
        }
        ok = ok and inv_ok;
        // matmulLd (fn_qmmf_ld): every bf16 row count's rows written `ldo` apart hold the contiguous call's bytes,
        // and the gaps between them stay untouched
        {
            const ldo = n + 72;
            var dl = try cuda.DeviceBuffer.alloc(d, max_m * ldo * 2);
            defer dl.free();
            const strided = try gpa.alloc(u8, max_m * ldo * 2);
            defer gpa.free(strided);
            var ld_ok = true;
            var ld_n: usize = 0;
            var it = c.get("bf16").?.object.iterator();
            while (it.next()) |kv| {
                const m = try std.fmt.parseInt(usize, kv.key_ptr.*, 10);
                @memset(strided, 0xAB);
                try dl.upload(0, strided);
                try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
                try matmulLd(&k, stream, dx.ptr, kk, l, dl.ptr, ldo, m);
                try stream.synchronize();
                try dy.download(0, host[0 .. m * n * 2]);
                try dl.download(0, strided[0 .. m * ldo * 2]);
                ld_n += 1;
                for (0..m) |r| {
                    const row = strided[r * ldo * 2 ..][0 .. ldo * 2];
                    if (!std.mem.eql(u8, row[0 .. n * 2], host[r * n * 2 ..][0 .. n * 2]) or !std.mem.allEqual(u8, row[n * 2 ..], 0xAB)) {
                        ld_ok = false;
                        std.debug.print("  {s} bf16 m {d} row {d}: matmulLd differs from matmul (or wrote past its columns)\n", .{ name, m, r });
                        break;
                    }
                }
            }
            std.debug.print("  {s}: matmulLd (rows {d} apart) {s} matmul at {d} row counts\n", .{ name, ldo, if (ld_ok) "byte-equal to" else "DIFFERS from", ld_n });
            ok = ok and ld_ok;
        }
        // time at 1 and max_m rows (bf16 out)
        var us: [2]f64 = undefined;
        for ([_]usize{ 1, max_m }, 0..) |m, i| {
            const reps: usize = if (m == 1) 200 else 20;
            try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try e0.record(stream);
            for (0..reps) |_| try matmul(&k, stream, dx.ptr, kk, l, dy.ptr, false, m);
            try e1.record(stream);
            try e1.synchronize();
            us[i] = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(reps));
        }
        const wbytes: f64 = @floatFromInt(npad * kk + npad * (kk / 64) * 4);
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(max_m * n * kk));
        std.debug.print("{s} {s}: n {d} k {d} split_k {d}: layouts w8 {s} bs {s}, outputs {d}/{d} equal, rows {s} one-row calls; 1 row {d:.1} us ({d:.0} GB/s weights), {d} rows {d:.0} us ({d:.1} TFLOP/s)\n", .{
            if (w8_ok and bs_ok and equal == total and inv_ok) "PASS" else "FAIL", name, n, kk, if (hip_build) splitKHip(n, kk) else splitK(n, kk),
            if (w8_ok) "equal" else "DIFFER", if (bs_ok) "equal" else "DIFFER", equal, total, if (inv_ok) "equal" else "DIFFER",
            us[0], wbytes / us[0] / 1000.0, max_m, us[1], flops / us[1] / 1e6,
        });
    }
    return ok;
}
