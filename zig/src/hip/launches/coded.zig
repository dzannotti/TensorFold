//! The coded decoders' tiles (kernels/hip/quant/coded_tiles.hpp): block FP8 and BF16 weights in groups of 128, on the
//! stream tile (a window's products whose K is not a multiple of 256), the WMMA verify tile and its split (the rest of
//! a window's), and the dequantizing prompt tile (a prompt's). Which tile a product takes depends on its shape and on
//! whether it is a prompt's, never on its row count within a window, so a row's bits are the same at every count.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const quant = @import("core").quant;

const Error = runtime.Error;
const Arg = affine.Arg;
const StreamSides = affine.StreamSides;

pub const Code = quant.mlx.Code;

const names = [_][]const u8{ "10Fp8Decoder", "11Bf16Decoder" };
const stream_rows = [_]c_int{ 1, 2, 4 };
const stream_cbs = [_]c_int{ 2, 4 };
const stream_waves = 4;
const group = 128;

pub const Kernels = struct {
    /// [decoder][rows 1, 2, 4][columns a lane 2, 4]
    stream: [2][stream_rows.len][stream_cbs.len]Function,
    /// [decoder][waves 2, 4][16-row tiles 1, 2]; BF16 has only the two-tile block (its one-tile block spills)
    verify: [2][2][2]?Function,
    /// [decoder][waves 4, 8]
    split: [2][2]Function,
    /// [decoder][128 x 256, 256 x 128]; BF16 has only the 256-row block
    dq: [2][2]?Function,

    pub fn load(tiles: Module, matrix: bool) Error!?Kernels {
        if (!matrix) return null;
        var k: Kernels = undefined;
        const r = "_ZN2tf4rocm";
        inline for (names, 0..) |d, di| {
            inline for (stream_rows, 0..) |rows, ri| inline for (stream_cbs, 0..) |cb, ci| {
                k.stream[di][ri][ci] = try tiles.function(std.fmt.comptimePrint(r ++ "17coded_dot2_streamINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ d, rows, cb }));
            };
            inline for (.{ 2, 4 }, 0..) |wv, wi| inline for (.{ 1, 2 }, 0..) |mt, ti| {
                k.verify[di][wi][ti] = if (di == 1 and mt == 1) null else try tiles.function(std.fmt.comptimePrint(r ++ "17coded_wmma_verifyINS0_{s}ELi{d}ELi128ELi{d}EEEvNS0_6AffineENS0_11StreamSidesE", .{ d, mt, wv }));
            };
            inline for (.{ 4, 8 }, 0..) |wv, wi| {
                k.split[di][wi] = try tiles.function(std.fmt.comptimePrint(r ++ "23coded_wmma_verify_splitINS0_{s}ELi128ELi{d}EEEvNS0_6AffineENS0_11StreamSidesE", .{ d, wv }));
            }
            inline for (.{ 128, 256 }, 0..) |bm, bi| {
                k.dq[di][bi] = if (di == 1 and bm == 128) null else try tiles.function(std.fmt.comptimePrint(r ++ "15coded_prompt_dqINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ d, bm }));
            }
        }
        return k;
    }

    fn index(code: Code) Error!usize {
        return switch (code) {
            .fp8 => 0,
            .bf16 => 1,
            .affine => error.Invalid,
        };
    }

    fn cdiv(n: c_int, by: c_int) c_uint {
        return @intCast(@divTrunc(n + by - 1, by));
    }

    fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == @import("util.zig").counting) return;
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .r = r, .handle = s }, args);
    }

    /// Whether `a` is a shape these tiles take: groups of 128, K a multiple of 32, bf16 x.
    pub fn takes(a: Arg) bool {
        return a.group == group and @mod(a.k, 32) == 0 and a.fp16 == 0 and a.route.items == 0;
    }

    /// `a` (out or out16 set) on the tile its shape and path take: a prompt's on the dequantizing tile, a window's on
    /// the verify family when K is a multiple of 256, else on the stream tile.
    pub fn run(k: *const Kernels, r: *const runtime.Runtime, code: Code, a: Arg, prefill: bool, s: abi.Stream) Error!void {
        if (!takes(a)) return error.Invalid;
        const d = try index(code);
        var args: hl.Args = .{};
        try args.add(a);
        if (prefill) {
            const tall = d == 1 or a.k >= 3 * a.n;
            const bm: c_int = if (tall) 256 else 128;
            return go(r, k.dq[d][@intFromBool(tall)].?, .{ .x = cdiv(a.n, @divExact(32768, bm)) * cdiv(a.m, bm) }, .{ .x = 256 }, s, &args);
        }
        if (@mod(a.k, 256) == 0) {
            const sides: StreamSides = .{};
            try args.add(sides);
            // up to 16 rows: K over a block's waves, 16 columns a block (8 waves while the columns are few)
            if (a.m <= 8 or (a.m <= 16 and a.k <= 8192)) {
                const wi: usize = @intFromBool(a.n <= 32768);
                return go(r, k.split[d][wi], .{ .x = cdiv(a.n, 16), .y = cdiv(a.m, 16) }, .{ .x = @intCast(32 * @as(c_int, if (wi == 1) 8 else 4)) }, s, &args);
            }
            const t: usize = if (a.m <= 16 and d == 0) 0 else 1;
            const wi: usize = if (a.n <= 6144 and a.k <= 8192) 0 else 1;
            const cols: c_int = 16 * @as(c_int, if (wi == 0) 2 else 4);
            return go(r, k.verify[d][wi][t].?, .{ .x = cdiv(a.n, cols), .y = cdiv(a.m, 16 * @as(c_int, if (t == 0) 1 else 2)) }, .{ .x = @intCast(cols * 2) }, s, &args);
        }
        // the stream tile: lanes a column by K's chunks, columns a lane by the code words a lane holds
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const ri: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else 2;
        const ci: usize = if (d == 0) 1 else 0;
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * stream_cbs[ci];
        try args.add(StreamSides{});
        try args.add(@as(c_int, lpc_log2));
        try args.add(@as(c_int, 2)); // a chunk's group is chunk >> 2 (128 / 32)
        try go(r, k.stream[d][ri][ci], .{ .x = cdiv(a.n, per_block), .y = cdiv(a.m, stream_rows[ri]) }, .{ .x = 32 * stream_waves }, s, &args);
    }
};
