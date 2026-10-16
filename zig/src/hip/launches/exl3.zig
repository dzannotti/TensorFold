//! EXL3 projections (quant/exl3.hip): the input rotation and the row-invariant WMMA product.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const exl3 = @import("core").quant.exl3;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const invalid = util.invalid;

/// quant/exl3.hip's Exl3, by value.
pub const Arg = extern struct {
    xh: u64,
    trellis: u64,
    stride_k: i64,
    stride_nb: i64,
    svh: u64,
    bias: u64,
    out: u64,
    partial: u64,
    m: c_int,
    k: c_int,
    n: c_int,
    splits: c_int,
    out_kind: c_int,
};

/// quant/exl3.hip's kExl3Counters: (column block, row block) pairs a split launch may take.
const arrivals = 65536;

/// A launch's tile: 16-row tiles (1 and 2 for a round, 8 for a prompt's chunk) by the 128-column blocks a wave walks
/// (2 on the widest products, sharing x); every row's and column's sums are the same.
const Tile = struct { rows: u32, cols: u32 };
const tiles = [_]Tile{ .{ .rows = 1, .cols = 1 }, .{ .rows = 2, .cols = 1 }, .{ .rows = 8, .cols = 1 }, .{ .rows = 1, .cols = 2 }, .{ .rows = 2, .cols = 2 } };

/// The tile of an `m`-row product with `n` columns.
fn tileOf(m: usize, n: u32) usize {
    const wide = n % 256 == 0 and n / 256 >= 320;
    if (m <= 16) return if (wide) 3 else 0;
    if (m <= 32) return if (wide) 4 else 1;
    return 2;
}

/// Widths the kernels are built for, in half bits: whole bits 2 to 8 on every codebook, 1.5 to 3.5 on mul1.
fn built(halves: usize, codebook: usize) bool {
    return (halves % 2 == 0 and halves >= 4 and halves <= 16) or (codebook == 2 and (halves == 3 or halves == 5 or halves == 7));
}

pub const Fns = struct {
    rotate: Function,
    /// ciru_q8's input rotation: 128-block Hadamards, no signs, in the activation's kind.
    had: Function,
    /// By half bits, codebook (3inst, mcg, mul1) and tile.
    product: [17][3][tiles.len]?Function,
    /// A prompt's product on the dequantizing core (mul1, whole bits): by half bits, 128 x 256 or 256 x 128 blocks.
    prompt: [17][2]?Function,

    pub fn load(m: Module) Error!Fns {
        @setEvalBranchQuota(200_000);
        var f: Fns = .{ .rotate = try m.function("tf_exl3_rotate_in"), .had = try m.function("tf_had128"), .product = @splat(@splat(@splat(null))), .prompt = @splat(@splat(null)) };
        inline for (.{ 4, 6, 8, 10, 12, 14, 16 }) |h| inline for (.{ .{ 128, 256 }, .{ 256, 128 } }, 0..) |b, i| {
            f.prompt[h][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm11exl3_promptILi{d}ELi2ELi{d}ELi{d}EEEvNS0_4Exl3E", .{ h, b[0], b[1] }));
        };
        inline for (0..17) |h| inline for (0..3) |cb| {
            if (comptime built(h, cb)) inline for (tiles, 0..) |t, i| {
                f.product[h][cb][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm12exl3_productILi{d}ELi{d}ELi{d}ELi{d}EEEvNS0_4Exl3E", .{ h, cb, t.rows, t.cols }));
            };
        };
        return f;
    }
};

/// The K splits of a (K, N) product: enough column blocks for the GPU's 80 SIMDs, from the shape alone (bits never
/// depend on the call's rows); a column block's last split block adds the splits' sums in order. Each split's tiles
/// stay a multiple of the kernel's 4 k steps.
pub fn splits(k: u32, n: u32) u32 {
    const blocks = n / 128 / tiles[tileOf(1, n)].cols;
    var s: u32 = 1;
    while (blocks * s < 80 and s < 8 and ((k / 16) / (2 * s)) % 4 == 0) s *= 2;
    return s;
}

/// Device scratch a product of `m` rows takes: the rotated rows, then the split sums when K is split.
pub fn scratchBytes(m: usize, k: u32, n: u32) usize {
    const s = splits(k, n);
    return std.mem.alignForward(usize, m * k * 2, 256) + if (s > 1) m * n * 4 * s else 0;
}

/// y [m, n] (`out_kind`: 0 fp32, 1 bf16, 2 fp16) = x [m, k] (`x_kind` 1 bf16, 2 fp16) times the EXL3 matrix;
/// `scratch` holds `scratchBytes(m, k, n)`.
pub fn project(l: *const Launcher, x: u64, x_kind: c_int, w: exl3.Matrix, n: u32, k: u32, m: usize, out: u64, out_kind: c_int, scratch: u64, s: S) Error!void {
    if (m == 0 or k % 128 != 0 or n % 128 != 0 or n > w.columns or (x_kind != 1 and x_kind != 2)) return invalid("exl3 project");
    const cb: usize = @backingInt(w.codebook);
    const tile = tiles[tileOf(m, n)];
    const f = l.exl3.product[w.halves][cb][tileOf(m, n)] orelse return invalid("exl3 width");
    const sp = splits(k, n);
    const words: i64 = 4 * @as(i64, w.halves);
    const arg: Arg = .{
        .xh = scratch,
        .trellis = w.trellis,
        .stride_k = @as(i64, w.columns / 16) * words,
        .stride_nb = 8 * words,
        .svh = w.svh,
        .bias = w.bias,
        .out = out,
        .partial = scratch + std.mem.alignForward(usize, m * k * 2, 256),
        .m = @intCast(m),
        .k = @intCast(k),
        .n = @intCast(n),
        .splits = @intCast(sp),
        .out_kind = out_kind,
    };
    {
        var a: Args = .{};
        try a.add(x);
        try a.add(x_kind);
        try a.add(w.suh);
        try a.add(scratch);
        try a.add(@as(c_int, @intCast(k)));
        // a round's row tiles read x by k steps (quant/exl3.hip)
        try a.add(@as(c_int, if (tile.rows <= 2) @intCast(m) else 0));
        try l.go(l.exl3.rotate, dim(k / 128, m, 1), dim(32, 1, 1), 0, s, &a);
    }
    const rows = 16 * tile.rows;
    const blocks = (m + rows - 1) / rows;
    const cols = n / 128 / tile.cols;
    // the split blocks' arrival counts (quant/exl3.hip) cover every block of a launch
    if (sp > 1 and blocks * cols > arrivals) return invalid("exl3 arrivals");
    var a: Args = .{};
    try a.add(arg);
    try l.go(f, dim(cols, sp, blocks), dim(256, 1, 1), 0, s, &a);
}

/// `project` for a prompt's rows on the dequantizing tile (quant/exl3.hip exl3_prompt): every output one WMMA chain over
/// k, so a row's bits depend on its own x and (K, N) alone, never on m or where the prompt is cut, but differ from
/// `project`'s (decode and rounds). False, nothing launched, where no prompt tile takes the matrix (mul1 whole bits).
pub fn prompt(l: *const Launcher, x: u64, x_kind: c_int, w: exl3.Matrix, n: u32, k: u32, m: usize, out: u64, out_kind: c_int, scratch: u64, s: S) Error!bool {
    if (m == 0 or k % 128 != 0 or n % 128 != 0 or n > w.columns or (x_kind != 1 and x_kind != 2)) return invalid("exl3 prompt");
    if (w.codebook != .mul1 or w.halves % 2 != 0) return false;
    // its buffer loads take 31-bit byte offsets into the rotated x and the trellis
    const words: u64 = 4 * @as(u64, w.halves);
    if (@as(u64, m) * k * 2 >= 1 << 31 or @as(u64, k / 16) * (w.columns / 16) * words * 4 >= 1 << 31) return false;
    // 256 x 128 (half the decode a weight) but where k is near n (o_proj, out_proj: 128 x 256 measured 25% faster);
    // the same bits either way
    const tall = k < n or k >= 3 * n;
    const f = l.exl3.prompt[w.halves][@intFromBool(tall)] orelse return false;
    {
        var a: Args = .{};
        try a.add(x);
        try a.add(x_kind);
        try a.add(w.suh);
        try a.add(scratch);
        try a.add(@as(c_int, @intCast(k)));
        try a.add(@as(c_int, 0));
        try l.go(l.exl3.rotate, dim(k / 128, m, 1), dim(32, 1, 1), 0, s, &a);
    }
    const arg: Arg = .{
        .xh = scratch,
        .trellis = w.trellis,
        .stride_k = @as(i64, w.columns / 16) * @as(i64, @intCast(words)),
        .stride_nb = 8 * @as(i64, @intCast(words)),
        .svh = w.svh,
        .bias = w.bias,
        .out = out,
        .partial = 0,
        .m = @intCast(m),
        .k = @intCast(k),
        .n = @intCast(n),
        .splits = 1,
        .out_kind = out_kind,
    };
    const bm: usize = if (tall) 256 else 128;
    const bn: usize = 32768 / bm;
    var a: Args = .{};
    try a.add(arg);
    try l.go(f, dim(((n + bn - 1) / bn) * ((m + bm - 1) / bm), 1, 1), dim(256, 1, 1), 0, s, &a);
    return true;
}

/// out [rows, k] = x [rows, k] rotated by 128-block Hadamards in x's kind (`kind`: 1 fp16, 2 bf16); `out` may be `x`.
pub fn hadamard(l: *const Launcher, x: u64, kind: c_int, out: u64, k: u32, rows: usize, s: S) Error!void {
    if (k % 128 != 0 or (kind != 1 and kind != 2)) return invalid("hadamard");
    const waves: i64 = @intCast(rows * (k / 128));
    if (waves == 0) return;
    var a: Args = .{};
    try a.add(x);
    try a.add(kind);
    try a.add(out);
    try a.add(@as(c_int, @intCast(k)));
    try a.add(waves);
    try l.go(l.exl3.had, dim(@divFloor(waves + 7, 8), 1, 1), dim(256, 1, 1), 0, s, &a);
}

test "splits keep whole tile runs and depend on the shape alone" {
    for ([_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 }, .{ 5120, 1024 }, .{ 6144, 5120 }, .{ 5120, 248320 }, .{ 128, 128 } }) |kn| {
        const s = splits(kn[0], kn[1]);
        try std.testing.expect(s >= 1 and s <= 8);
        try std.testing.expectEqual(@as(u32, 0), ((kn[0] / 16) / s) % 4);
    }
}
