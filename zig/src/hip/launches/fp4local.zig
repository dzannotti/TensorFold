//! FP4-Local's products (fp4local/fp4local.hip): stream tile for decode, gfx11 matrix tiles for a prompt.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const Caps = @import("../caps.zig").Caps;
const affine = @import("affine.zig");
const stream_pick = @import("stream_pick.zig");
const stream_layout = @import("stream_layout.zig");
const util = @import("util.zig");
const mxfp8_gemv = @import("mxfp8_gemv.zig");
const nvfp4_experts = @import("nvfp4_experts.zig");
const bf16_gemm = @import("bf16_gemm.zig");
const kernels = @import("../kernels.zig");

const Error = runtime.Error;
pub const Arg = affine.Arg;
pub const StreamSides = stream_layout.StreamSides;
pub const GlobalLayout = affine.GlobalLayout;

/// The decoders as the kernels number them (quant/fp4local_tiles.hpp Fp4LocalDecoder).
pub const Format = enum(u2) {
    nvfp4 = 0,
    mxfp8 = 1,
    bf16 = 2,

    /// Code words of a 32-input chunk: what bounds a lane's columns (stream_pick's `bits`).
    pub fn chunkWords(f: Format) c_int {
        return switch (f) {
            .nvfp4 => 4,
            .mxfp8 => 8,
            .bf16 => 16,
        };
    }
};

/// The BF16 prompt fold: 128 inputs where K takes it, else the widest of 64 and 32 that divides K (HC up's K is 320).
pub fn bf16Fold(k: c_int) c_int {
    return if (@rem(k, 128) == 0) 128 else if (@rem(k, 64) == 0) 64 else 32;
}

/// Rows of a decode launch: the stream tile; past it the prompt tile.
pub const decode_rows = 8;
/// A routed prompt whose items average at most this many rows takes the 64-row item tile (NVFP4 only).
pub const item_rows = 64;

const rows_of = affine.stream_rows;
const cbs = affine.stream_cbs;

pub const Kernels = struct {
    /// stream[format][row tile][columns a lane][pair]: null where the format's chunk words or 16 sums do not fit.
    stream: [3][rows_of.len][cbs.len][2]?Function,
    wmma: [3]Function,
    item: Function,
    /// The (gate | up) prompt tile with silu(gate) x up in its epilogue: NVFP4 and MXFP8.
    pair: [2]Function,
    caps: Caps,
    /// gfx11: the matrix tiles run; elsewhere every prompt is refused.
    matrix: bool,
    /// The Loom MXFP8 GEMV: the stream tile's bytes, at the shapes it is faster; empty in a build without Loom.
    gemv: mxfp8_gemv.Kernels,
    /// The Loom NVFP4 grouped-expert prompt kernels: the routed prompt tiles' bytes; empty without Loom.
    experts: nvfp4_experts.Kernels,
    /// The Loom dense bf16 prompt GEMMs: fp4l_wmma's BF16 bytes, at the shapes they are faster; empty without Loom.
    dense: bf16_gemm.Kernels,

    /// fp4local.hip's kernels in `m`, and the Loom kernels in `loom` the build has (`manifests` check them).
    pub fn load(m: Module, caps: Caps, loom: [kernels.loom_count]?Module, manifests: [kernels.loom_count][]const u8) Error!Kernels {
        const gemv = @backingInt(kernels.Loom.mxfp8_gemv);
        const gu = @backingInt(kernels.Loom.nvfp4_expert_gate_up);
        const dn = @backingInt(kernels.Loom.nvfp4_expert_down);
        const experts = try nvfp4_experts.Kernels.load(.{ loom[gu], loom[dn] }, .{ manifests[gu], manifests[dn] });
        const d0 = @backingInt(kernels.Loom.bf16_gemm_128x128);
        const d1 = @backingInt(kernels.Loom.bf16_gemm_128x256);
        const dense = try bf16_gemm.Kernels.load(.{ loom[d0], loom[d1] }, .{ manifests[d0], manifests[d1] });
        @setEvalBranchQuota(100_000);
        var k: Kernels = .{ .stream = @splat(@splat(@splat(@splat(null)))), .wmma = undefined, .item = undefined, .pair = undefined, .caps = caps, .matrix = caps.matrix == .wmma11, .gemv = try mxfp8_gemv.Kernels.load(loom[gemv], manifests[gemv]), .experts = experts, .dense = dense };
        inline for (0..3) |f| {
            const words = (Format.chunkWords(@fromBackingInt(@intCast(f))));
            inline for (rows_of, 0..) |rows, ri| inline for (cbs, 0..) |cb, ci| inline for (0..2) |pair| {
                const fits = cb * rows <= 16 and cb * words <= affine.stream_code_words and !(pair == 1 and f == 2);
                if (fits) k.stream[f][ri][ci][pair] = try m.function(std.fmt.comptimePrint(
                    "_ZN2tf4rocm11fp4l_streamILi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineENS0_11StreamSidesEi",
                    .{ f, rows, cb, pair },
                ));
            };
            k.wmma[f] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm9fp4l_wmmaILi{d}EEEvNS0_6AffineE", .{f}));
        }
        k.item = try m.function("_ZN2tf4rocm9fp4l_itemILi0EEEvNS0_6AffineE");
        k.pair = .{ try m.function("_ZN2tf4rocm9fp4l_pairILi0EEEvNS0_6AffineEf"), try m.function("_ZN2tf4rocm9fp4l_pairILi1EEEvNS0_6AffineEf") };
        return k;
    }

    fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, threads: c_uint, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == util.counting) return;
        try hl.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, .{ .r = r, .handle = s }, args);
    }

    /// Lanes a column (log2) from K alone, so its sum order is fixed: the most, up to 32, that split the chunks evenly.
    pub fn lanes(kk: c_int) u5 {
        const chunks = kk >> 5;
        var l: u5 = 0;
        while (l < 5 and @rem(chunks, @as(c_int, 2) << l) == 0) l += 1;
        return l;
    }

    /// The decode tile's choice: its function, lanes a column (log2) and outputs a block.
    pub const Pick = struct { f: Function, lpc_log2: u5, per_block: c_int, rows: c_int };

    /// The stream tile for `a` (m rows, total_n outputs over `items`); `pair`: outputs are silu(gate) x up.
    pub fn pickStream(k: *const Kernels, fmt: Format, a: Arg, total_n: c_int, items: c_int, pair: bool) ?Pick {
        if (a.m < 1 or a.k < 32 or @rem(a.k, 32) != 0) return null;
        const lpc_log2 = lanes(a.k);
        const t = stream_pick.pick(k.caps.generation, a.m, lpc_log2, fmt.chunkWords(), total_n, items, pair);
        const ri = stream_pick.rowTile(a.m);
        const f = k.stream[@backingInt(fmt)][ri][t.at][@intFromBool(pair)] orelse return null;
        return .{ .f = f, .lpc_log2 = lpc_log2, .per_block = t.per_block, .rows = rows_of[ri] };
    }

    /// The stream tile over `a` (or a plan's `items`, or `sides`): fp32 `a.out`, or the activation type in `a.out16`.
    pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, sides_in: StreamSides, items: c_int, s: abi.Stream) Error!void {
        const plain = sides_in.pair_cols == 0 and items == 1 and a.route.items == 0;
        if (fmt == .mxfp8 and plain) {
            if (sides_in.count == 0) if (k.gemv.pick(a.n, a.k, a.m, if (a.out16 != 0) .bf16 else .f32)) |at| {
                const half = a.out16 != 0;
                return k.gemv.go(r, at, a.m, a.n, a.words, a.scale.p, a.x, if (half) a.out16 else a.out, half, s);
            };
            if (stitched(sides_in, a.k)) |st| if (k.gemv.pick(st.n, a.k, a.m, .sides)) |at| {
                return k.gemv.goSides(r, at, a.m, st.n, st.cuts, st.words, st.scale, a.x, st.out, s);
            };
        }
        return k.tileGo(r, fmt, a, sides_in, items, s);
    }

    /// A group of sides as one product: its planes, n, the column cuts between sides and their outputs in that order.
    pub const Stitched = struct { words: u64, scale: u64, n: c_int, cuts: [3]c_int, out: [4]u64 };

    /// A bf16 group whose MXFP8 sides follow one another in memory, each starting at a multiple of 8 rows, or null.
    pub fn stitched(sides: StreamSides, kk: c_int) ?Stitched {
        const count: usize = @intCast(@max(sides.count, 0));
        if (count < 2 or count > 4 or sides.out_half == 0 or kk < 32) return null;
        const k: u64 = @intCast(kk);
        var order: [4]usize = .{ 0, 1, 2, 3 };
        std.mem.sort(usize, order[0..count], sides, struct {
            fn less(s: StreamSides, a: usize, b: usize) bool {
                return s.words[a] < s.words[b];
            }
        }.less);
        const first = order[0];
        var st: Stitched = .{ .words = sides.words[first], .scale = sides.scale[first], .n = 0, .cuts = undefined, .out = @splat(sides.out[first]) };
        for (0..4) |i| {
            if (i < count) {
                const s = order[i];
                const rows: u64 = @intCast(st.n);
                if (sides.n[s] < 1 or @rem(st.n, 8) != 0 or sides.words[s] != st.words + rows * k or sides.scale[s] != st.scale + rows * (k / 32)) return null;
                st.out[i] = sides.out[s];
                st.n += sides.n[s];
            }
            if (i < 3) st.cuts[i] = st.n;
        }
        return st;
    }

    /// `streamGo` on fp4local.hip's stream tile alone: the kernel the Loom GEMV's bytes are checked against.
    pub fn tileGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, sides_in: StreamSides, items: c_int, s: abi.Stream) Error!void {
        const pair = sides_in.pair_cols > 0;
        var total: c_int = if (pair) sides_in.pair_cols else a.n;
        if (sides_in.count > 0) {
            total = 0;
            for (sides_in.n[0..@intCast(sides_in.count)]) |n| total += n;
        }
        const p = k.pickStream(fmt, a, total, items, pair) orelse return refuse("stream shape");
        var sides = sides_in;
        var blocks: c_uint = @intCast(util.cdiv(total, @as(usize, @intCast(p.per_block))));
        if (sides.count > 0) {
            blocks = 0;
            for (0..@intCast(sides.count)) |i| {
                sides.first[i] = @intCast(blocks);
                blocks += @intCast(util.cdiv(sides.n[i], @as(usize, @intCast(p.per_block))));
            }
            sides.first[@intCast(sides.count)] = @intCast(blocks);
        }
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(sides);
        try args.add(@as(c_int, p.lpc_log2));
        const grid = stream_layout.grid(true, blocks, @intCast(util.cdiv(a.m, @as(usize, @intCast(p.rows)))), items);
        try go(r, p.f, grid, 32 * affine.stream_waves, s, &args);
    }

    /// The prompt tile over `a` (or every item of a plan): 128 x 128 blocks, or the 64-row item tile for small items.
    pub fn promptGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, items: c_int, expert_rows: c_int, s: abi.Stream) Error!void {
        if (!k.matrix) return refuse("the FP4-Local prompt tile needs gfx11's matrix cores");
        if (a.m < 1 or a.n < 1 or @rem(a.k, fold(fmt, a)) != 0 or a.x % 16 != 0) return refuse("prompt shape");
        if (fmt == .nvfp4 and k.experts.takesDown(a)) return k.experts.downGo(r, a, items, s);
        if (fmt == .bf16) if (k.dense.pick(a)) |t| return k.dense.go(r, t, a, s);
        return k.tilePromptGo(r, fmt, a, items, expert_rows, s);
    }

    /// `promptGo` on fp4local.hip's matrix tiles alone: what the Loom prompt kernels are checked against.
    pub fn tilePromptGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, items: c_int, expert_rows: c_int, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        const small = fmt == .nvfp4 and a.route.items != 0 and expert_rows <= item_rows;
        const rows: usize = if (small) item_rows else 128;
        const grid = util.dim(util.cdiv(a.n, 128), util.cdiv(a.m, rows), items);
        try go(r, if (small) k.item else k.wmma[@backingInt(fmt)], grid, 256, s, &args);
    }

    /// silu(gate) x up of a (gate | up) stack into a.out16 (pairs, n / 2): promptGo's sums, then moe_act's bytes.
    pub fn pairGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, items: c_int, limit: f32, s: abi.Stream) Error!void {
        if (!k.matrix) return refuse("the FP4-Local prompt tile needs gfx11's matrix cores");
        if (fmt == .bf16 or a.m < 1 or @rem(a.n, 2) != 0 or @rem(a.k, 128) != 0 or a.x % 16 != 0 or a.out16 == 0) return refuse("pair shape");
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(limit);
        const grid = util.dim(util.cdiv(@divExact(a.n, 2), 64), util.cdiv(a.m, 128), items);
        try go(r, k.pair[@backingInt(fmt)], grid, 256, s, &args);
    }

    /// The inputs a prompt output's fold covers.
    pub fn fold(fmt: Format, a: Arg) c_int {
        return if (fmt == .bf16) a.group else 128;
    }

    fn refuse(what: []const u8) Error {
        std.log.err("fp4local: {s}", .{what});
        return error.Invalid;
    }
};

test "a column's lanes split K's chunks evenly, at most 32" {
    try std.testing.expectEqual(@as(u5, 4), Kernels.lanes(2560));
    try std.testing.expectEqual(@as(u5, 5), Kernels.lanes(6144));
    try std.testing.expectEqual(@as(u5, 2), Kernels.lanes(640));
    try std.testing.expectEqual(@as(u5, 1), Kernels.lanes(320));
}

test "the BF16 fold is 128 where K takes it, else 64 or 32" {
    try std.testing.expectEqual(@as(c_int, 128), bf16Fold(2560));
    try std.testing.expectEqual(@as(c_int, 64), bf16Fold(320));
    try std.testing.expectEqual(@as(c_int, 32), bf16Fold(96));
}

test "a group of stitched MXFP8 rows is one product in memory order; a gap or fp32 outputs are not" {
    const k: u64 = 2560;
    var g: StreamSides = .{ .count = 4, .out_half = 1 };
    const n = [_]c_int{ 10240, 6144, 48, 48 };
    const at = [_]u64{ 0, 10240, 16432, 16384 };
    for (0..4) |i| {
        g.n[i] = n[i];
        g.words[i] = 0x1000 + at[i] * k;
        g.scale[i] = 0x9000 + at[i] * (k / 32);
        g.out[i] = 0x100 * (i + 1);
    }
    const st = Kernels.stitched(g, 2560).?;
    try std.testing.expectEqual(@as(c_int, 16480), st.n);
    try std.testing.expectEqual([3]c_int{ 10240, 16384, 16432 }, st.cuts);
    try std.testing.expectEqual([4]u64{ 0x100, 0x200, 0x400, 0x300 }, st.out);
    var gap = g;
    gap.words[3] += k;
    try std.testing.expect(Kernels.stitched(gap, 2560) == null);
    var wide = g;
    wide.out_half = 0;
    try std.testing.expect(Kernels.stitched(wide, 2560) == null);
}
