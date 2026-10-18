//! GGUF projections (quant/gguf.hip): ggml's block quantizations read in place, the row-invariant WMMA product, the
//! dequantizing prompt tile and the row gather.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const gguf = @import("core").quant.gguf;
const affine = @import("affine.zig");
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// quant/gguf.hip's Gguf, by value.
pub const Arg = extern struct {
    w: u64,
    x: u64,
    out: u64,
    sums: u64 = 0,
    m: c_int,
    n: c_int,
    k: c_int,
    row_bytes: c_int,
    out_kind: c_int,
};

/// The product's row tiles: 16, 32, 64 or 128 rows a block (more rows take more blocks, the same sums).
const tiles = [_]u32{ 1, 2, 4, 8 };

fn tileOf(m: usize) usize {
    if (m <= 16) return 0;
    if (m <= 32) return 1;
    if (m <= 64) return 2;
    return 3;
}

/// The elements one (s, mn) covers (quant/gguf.hpp kGroup).
fn group(t: gguf.Type) u32 {
    return if (t == .q6_k or t == .q3_k) 16 else 32;
}

/// Device scratch a product of `m` rows takes: the rows' x sums a group.
pub fn scratchBytes(m: usize, k: u32) usize {
    return (m + 15) / 16 * 16 * (k / 16) * 4;
}

pub const Fns = struct {
    /// By `gguf.kinds` index and row tile.
    product: [gguf.kinds.len][tiles.len]Function,
    /// The prompt tile's sums for a prompt's few rows, by index and row tile (16 to 64 rows a block).
    chain: [gguf.kinds.len][tiles.len - 1]Function,
    /// x's sums over groups of 16 and of 32.
    sums: [2]Function,
    prompt: [gguf.kinds.len]Function,
    rows: [gguf.kinds.len]Function,

    pub fn load(m: Module) Error!Fns {
        @setEvalBranchQuota(100_000);
        var f: Fns = undefined;
        inline for (gguf.kinds, 0..) |t, i| {
            const n = @intFromEnum(t);
            inline for (tiles, 0..) |mt, j| {
                f.product[i][j] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm12gguf_productILi{d}ELi{d}EEEvNS0_4GgufE", .{ n, mt }));
                if (j < tiles.len - 1) f.chain[i][j] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm10gguf_chainILi{d}ELi{d}EEEvNS0_4GgufE", .{ n, mt }));
            }
            f.prompt[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm11gguf_promptILi{d}EEEvNS0_6AffineE", .{n}));
            f.rows[i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm9gguf_rowsILi{d}EEEvPKhiPKiiPvi", .{n}));
        }
        f.sums = .{ try m.function("_ZN2tf4rocm9gguf_sumsILi16EEEvPKDF16bPfii"), try m.function("_ZN2tf4rocm9gguf_sumsILi32EEEvPKDF16bPfii") };
        return f;
    }
};

fn index(t: gguf.Type) ?usize {
    inline for (gguf.kinds, 0..) |k, i| if (k == t) return i;
    return null;
}

/// y [m, n] (`out_kind`: 0 fp32, 1 bf16, 2 fp16) = x bf16 [m, k] times the GGUF matrix: any m, a row's bits its own;
/// `scratch` holds `scratchBytes(m, k)`.
pub fn project(l: *const Launcher, x: u64, w: gguf.Matrix, n: u32, k: u32, m: usize, out: u64, out_kind: c_int, scratch: u64, s: S) Error!void {
    const t = index(w.type) orelse return invalid("gguf type");
    if (m == 0 or n == 0 or k % 256 != 0 or out_kind < 0 or out_kind > 2) return invalid("gguf project");
    const tile = tileOf(m);
    const g = group(w.type);
    {
        var a: Args = .{};
        try a.add(x);
        try a.add(scratch);
        try a.add(@as(c_int, @intCast(m)));
        try a.add(@as(c_int, @intCast(k)));
        try l.go(l.gguf.sums[@intFromBool(g == 32)], dim(cdiv(m * (k / g), 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    }
    const arg: Arg = .{ .w = w.blocks, .x = x, .out = out, .sums = scratch, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .row_bytes = @intCast(w.row_bytes), .out_kind = out_kind };
    var a: Args = .{};
    try a.add(arg);
    try l.go(l.gguf.product[t][tile], dim(cdiv(n, 16), cdiv(m, 16 * tiles[tile]), 1), dim(128, 1, 1), 0, s, &a);
}

/// Rows above which a prompt's product takes the dequantizing tile; up to them gguf_chain, the same sums.
pub const prompt_rows = 128;

/// `project` for a prompt's rows: each output one WMMA chain over k of ggml's weights rounded to BF16, on the
/// dequantizing tile above `prompt_rows` and gguf_chain up to them, so a row's bits depend on its own x and (K, N)
/// alone, never on where the prompt is cut (they differ from `project`'s, which keeps the weights exact). False,
/// nothing launched, when neither takes the call.
pub fn prompt(l: *const Launcher, x: u64, w: gguf.Matrix, n: u32, k: u32, m: usize, out: u64, out_kind: c_int, s: S) Error!bool {
    const t = index(w.type) orelse return false;
    if (m == 0 or n == 0 or out_kind < 0 or out_kind > 2 or k % 256 != 0 or x % 16 != 0) return false;
    // the tile's buffer loads take 31-bit byte offsets into x
    if (m <= prompt_rows or out_kind == 2 or @as(u64, m) * k * 2 >= 1 << 31) {
        const tile = @min(tileOf(m), 2);
        const arg: Arg = .{ .w = w.blocks, .x = x, .out = out, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .row_bytes = @intCast(w.row_bytes), .out_kind = out_kind };
        var a: Args = .{};
        try a.add(arg);
        try l.go(l.gguf.chain[t][tile], dim(cdiv(n, 64), cdiv(m, 16 * tiles[tile]), 1), dim(128, 1, 1), 0, s, &a);
        return true;
    }
    var arg: affine.Arg = .{ .x = x, .words = w.blocks, .scale = .{ .p = 0, .kind = 1 }, .bias = .{ .p = 0, .kind = 1 }, .out = 0, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .bits = 0, .group = 0, .fp16 = 0 };
    if (out_kind == 0) arg.out = out else arg.out16 = out;
    var a: Args = .{};
    try a.add(arg);
    try l.go(l.gguf.prompt[t], dim(cdiv(n, 128) * cdiv(m, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    return true;
}

/// `count` rows of the GGUF matrix (row ids[i], or i without ids), k values each of `out_kind`: fp32 is exactly
/// dequantize_row_*, the others it rounded to nearest even.
pub fn rows(l: *const Launcher, w: gguf.Matrix, ids: u64, count: usize, k: u32, out: u64, out_kind: c_int, s: S) Error!void {
    const t = index(w.type) orelse return invalid("gguf type");
    if (count == 0) return;
    if (k % 256 != 0) return invalid("gguf rows");
    var a: Args = .{};
    try a.add(w.blocks);
    try a.add(@as(c_int, @intCast(w.row_bytes)));
    try a.add(ids);
    try a.add(@as(c_int, @intCast(k)));
    try a.add(out);
    try a.add(out_kind);
    try l.go(l.gguf.rows[t], dim(count, 1, 1), dim(256, 1, 1), 0, s, &a);
}
