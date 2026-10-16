//! EXL3 projections (quant/exl3.hip): the input rotation, the row-invariant WMMA product and the split sums' finish.

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

/// The row tiles of a launch (16 rows each): 1 and 2 for a round, 8 for a prompt's chunk; the sums are the same.
const row_tiles = [_]u32{ 1, 2, 8 };

/// Widths the kernels are built for, in half bits: whole bits 2 to 8 on every codebook, 1.5 to 3.5 on mul1.
fn built(halves: usize, codebook: usize) bool {
    return (halves % 2 == 0 and halves >= 4 and halves <= 16) or (codebook == 2 and (halves == 3 or halves == 5 or halves == 7));
}

pub const Fns = struct {
    rotate: Function,
    finish: Function,
    /// By half bits, codebook (3inst, mcg, mul1) and row tiles.
    product: [17][3][row_tiles.len]?Function,

    pub fn load(m: Module) Error!Fns {
        var f: Fns = .{ .rotate = try m.function("tf_exl3_rotate_in"), .finish = try m.function("tf_exl3_finish"), .product = @splat(@splat(@splat(null))) };
        inline for (0..17) |h| inline for (0..3) |cb| {
            if (comptime built(h, cb)) inline for (row_tiles, 0..) |mt, i| {
                f.product[h][cb][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm12exl3_productILi{d}ELi{d}ELi{d}EEEvNS0_4Exl3E", .{ h, cb, mt }));
            };
        };
        return f;
    }
};

/// The K splits of a (K, N) product: enough 128-column blocks to fill the GPU, from the shape alone (bits never
/// depend on the call's rows). Each split's tiles stay a multiple of the kernel's 4 k steps.
pub fn splits(k: u32, n: u32) u32 {
    const blocks = n / 128;
    var s: u32 = 1;
    while (blocks * s < 320 and s < 16 and ((k / 16) / (2 * s)) % 4 == 0) s *= 2;
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
    if (m == 0 or k % 128 != 0 or n % 128 != 0 or (x_kind != 1 and x_kind != 2)) return invalid("exl3 project");
    const cb: usize = @backingInt(w.codebook);
    const tile: usize = if (m <= 16) 0 else if (m <= 32) 1 else 2;
    const f = l.exl3.product[w.halves][cb][tile] orelse return invalid("exl3 width");
    const sp = splits(k, n);
    const words: i64 = 4 * @as(i64, w.halves);
    const arg: Arg = .{
        .xh = scratch,
        .trellis = w.trellis,
        .stride_k = @as(i64, n / 16) * words,
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
        try l.go(l.exl3.rotate, dim(k / 128, m, 1), dim(32, 1, 1), 0, s, &a);
    }
    {
        var a: Args = .{};
        try a.add(arg);
        const rows = 16 * row_tiles[tile];
        try l.go(f, dim(n / 128, sp, (m + rows - 1) / rows), dim(256, 1, 1), 0, s, &a);
    }
    if (sp > 1) {
        var a: Args = .{};
        try a.add(arg);
        try l.go(l.exl3.finish, dim(n / 128, m, 1), dim(32, 1, 1), 0, s, &a);
    }
}

test "splits keep whole tile runs and depend on the shape alone" {
    for ([_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 }, .{ 5120, 1024 }, .{ 6144, 5120 }, .{ 5120, 248320 }, .{ 128, 128 } }) |kn| {
        const s = splits(kn[0], kn[1]);
        try std.testing.expect(s >= 1 and s <= 16);
        try std.testing.expectEqual(@as(u32, 0), ((kn[0] / 16) / s) % 4);
    }
}
