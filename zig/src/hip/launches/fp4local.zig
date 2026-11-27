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

    pub fn load(m: Module, caps: Caps) Error!Kernels {
        @setEvalBranchQuota(100_000);
        var k: Kernels = .{ .stream = @splat(@splat(@splat(@splat(null)))), .wmma = undefined, .item = undefined, .pair = undefined, .caps = caps, .matrix = caps.matrix == .wmma11 };
        inline for (0..3) |f| {
            const words = (Format.chunkWords(@enumFromInt(f)));
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

    /// The decode tile's choice: its function, lanes a column (log2) and outputs a block.
    pub const Pick = struct { f: Function, lpc_log2: u5, per_block: c_int, rows: c_int };

    /// The stream tile for `a` (m rows, total_n outputs over `items`); `pair`: outputs are silu(gate) x up.
    pub fn pickStream(k: *const Kernels, fmt: Format, a: Arg, total_n: c_int, items: c_int, pair: bool) ?Pick {
        if (a.m < 1 or a.k < 32 or @rem(a.k, 32) != 0) return null;
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const t = stream_pick.pick(k.caps.generation, a.m, lpc_log2, fmt.chunkWords(), total_n, items, pair);
        const ri = stream_pick.rowTile(a.m);
        const f = k.stream[@intFromEnum(fmt)][ri][t.at][@intFromBool(pair)] orelse return null;
        return .{ .f = f, .lpc_log2 = lpc_log2, .per_block = t.per_block, .rows = rows_of[ri] };
    }

    /// The stream tile over `a` (or a plan's `items`, or `sides`): fp32 `a.out`, or the activation type in `a.out16`.
    pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, sides_in: StreamSides, items: c_int, s: abi.Stream) Error!void {
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
        var args: hl.Args = .{};
        try args.add(a);
        const small = fmt == .nvfp4 and a.route.items != 0 and expert_rows <= item_rows;
        const rows: usize = if (small) item_rows else 128;
        const grid = util.dim(util.cdiv(a.n, 128), util.cdiv(a.m, rows), items);
        try go(r, if (small) k.item else k.wmma[@intFromEnum(fmt)], grid, 256, s, &args);
    }

    /// silu(gate) x up of a (gate | up) stack into a.out16 (pairs, n / 2): promptGo's sums, then moe_act's bytes.
    pub fn pairGo(k: *const Kernels, r: *const runtime.Runtime, fmt: Format, a: Arg, items: c_int, limit: f32, s: abi.Stream) Error!void {
        if (!k.matrix) return refuse("the FP4-Local prompt tile needs gfx11's matrix cores");
        if (fmt == .bf16 or a.m < 1 or @rem(a.n, 2) != 0 or @rem(a.k, 128) != 0 or a.x % 16 != 0 or a.out16 == 0) return refuse("pair shape");
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(limit);
        const grid = util.dim(util.cdiv(@divExact(a.n, 2), 64), util.cdiv(a.m, 128), items);
        try go(r, k.pair[@intFromEnum(fmt)], grid, 256, s, &args);
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

test "the BF16 fold is 128 where K takes it, else 64 or 32" {
    try std.testing.expectEqual(@as(c_int, 128), bf16Fold(2560));
    try std.testing.expectEqual(@as(c_int, 64), bf16Fold(320));
    try std.testing.expectEqual(@as(c_int, 32), bf16Fold(96));
}
