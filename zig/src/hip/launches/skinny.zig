//! The Loom skinny MXFP8 tile (zig/kernels/loom/skinny_gemm): fp4l_stream's bytes for 1-64 rows, weights read once.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const mxfp8_gemv = @import("mxfp8_gemv.zig");
const util = @import("util.zig");

const Error = runtime.Error;
pub const Out = mxfp8_gemv.Out;

/// Rows a launch takes at most.
pub const max_rows = 64;

/// A compiled variant: K, row fragments (rows <= 16 f) and column fragments a wave; only K reaches the bits.
pub const Variant = struct {
    k: u16,
    f: u8,
    nt: u8,

    pub fn symbol(comptime v: Variant, comptime out: Out) [:0]const u8 {
        const end = switch (out) {
            .f32 => "f32",
            .bf16 => "bf16",
            .sides => "x4",
        };
        return std.fmt.comptimePrint("skinny_mxfp8_k{d}_f{d}_t{d}_{s}", .{ v.k, v.f, v.nt, end });
    }

    fn lanes(v: Variant) u32 {
        const chunks: u32 = v.k / 32;
        var l: u32 = 1;
        while (l < 32 and chunks % (2 * l) == 0) l *= 2;
        return l;
    }

    fn rounds(v: Variant) u32 {
        return v.k / 32 / v.lanes();
    }

    /// One wave a column lane.
    pub fn threads(v: Variant) u32 {
        return 32 * v.lanes();
    }

    /// Blocks over sides of `widths` columns: T blocks a tile of 2 T nt groups of 8 columns.
    pub fn blocks(v: Variant, widths: []const u32) u32 {
        const tile = 2 * v.rounds() * v.nt;
        var b: u32 = 0;
        for (widths) |w| b += util.cdiv(util.cdiv(w, 8), tile) * v.rounds();
        return b;
    }
};

/// Where a variant serves: the checkpoint's product (n by k, all sides) and its output.
pub const Entry = struct { n: u32, out: Out, v: Variant };

/// Column fragments a wave at each row-fragment count (1..4) for K.
fn nts(k: u16) [4]u8 {
    return switch (k) {
        2560 => .{ 4, 4, 2, 2 },
        6144 => .{ 2, 2, 1, 1 },
        else => .{ 1, 1, 1, 1 },
    };
}

pub const table = sides(16480) ++ sides(13952) ++ singles(1280, 2560) ++ singles(2560, 6144) ++ singles(2560, 640);

fn sides(n: u32) [4]Entry {
    var e: [4]Entry = undefined;
    for (0..4) |i| e[i] = .{ .n = n, .out = .sides, .v = .{ .k = 2560, .f = @intCast(i + 1), .nt = nts(2560)[i] } };
    return e;
}

fn singles(n: u32, k: u16) [8]Entry {
    var e: [8]Entry = undefined;
    for (0..4) |i| {
        const v: Variant = .{ .k = k, .f = @intCast(i + 1), .nt = nts(k)[i] };
        e[2 * i] = .{ .n = n, .out = .f32, .v = v };
        e[2 * i + 1] = .{ .n = n, .out = .bf16, .v = v };
    }
    return e;
}

/// The arguments each output's kernel takes: (m, n[, c1, c2, c3]) as i32, then w, e, x and the outputs.
fn interface(out: Out) [3]u32 {
    return if (out == .sides) .{ 12, 7, 20 } else .{ 6, 4, 8 };
}

pub const Kernels = struct {
    /// Each table entry's kernel; null in a build without Loom kernels.
    f: ?[table.len]Function = null,

    pub fn load(m: ?Module, manifest: []const u8) Error!Kernels {
        const mod = m orelse return .{};
        if (try problem(manifest)) |why| {
            std.log.err("skinny_mxfp8: the Loom image's manifest: {s}", .{why});
            return error.Invalid;
        }
        var f: [table.len]Function = undefined;
        inline for (table, 0..) |e, i| f[i] = try mod.function(comptime e.v.symbol(e.out));
        return .{ .f = f };
    }

    /// The entry for `m` rows of an n by k product into `out`, or null where another tile keeps it.
    pub fn pick(k: *const Kernels, n: c_int, kk: c_int, m: c_int, out: Out) ?usize {
        if (k.f == null or m < 1 or m > max_rows) return null;
        const f: u8 = @intCast(util.cdiv(m, 16));
        for (table, 0..) |e, i| if (e.n == n and e.v.k == kk and e.out == out and e.v.f == f) return i;
        return null;
    }

    /// x (m, k) bf16 against e4m3 codes `w` [n][k] and e8m0 scales `e` [n][k / 32] into `out`: fp32, or bf16 (`half`).
    pub fn go(k: *const Kernels, r: *const runtime.Runtime, at: usize, m: c_int, n: c_int, w: u64, e: u64, x: u64, out: u64, half: bool, s: abi.Stream) Error!void {
        if (table[at].out != (if (half) Out.bf16 else Out.f32)) return error.Invalid;
        var args: hl.Args = .{};
        try args.add(m);
        try args.add(n);
        inline for (.{ w, e, x, out }) |p| try args.add(p);
        return k.launch(r, at, table[at].v.blocks(&.{@intCast(n)}), &args, s);
    }

    /// `go` into four bf16 outputs split at `cuts` (multiples of 16).
    pub fn goSides(k: *const Kernels, r: *const runtime.Runtime, at: usize, m: c_int, n: c_int, cuts: [3]c_int, w: u64, e: u64, x: u64, out: [4]u64, s: abi.Stream) Error!void {
        if (table[at].out != .sides) return error.Invalid;
        for (cuts) |c| if (@rem(c, 16) != 0) return error.Invalid;
        var args: hl.Args = .{};
        try args.add(m);
        try args.add(n);
        for (cuts) |c| try args.add(c);
        inline for (.{ w, e, x }) |p| try args.add(p);
        for (out) |p| try args.add(p);
        const widths = [4]u32{ @intCast(cuts[0]), @intCast(cuts[1] - cuts[0]), @intCast(cuts[2] - cuts[1]), @intCast(n - cuts[2]) };
        return k.launch(r, at, table[at].v.blocks(&widths), &args, s);
    }

    fn launch(k: *const Kernels, r: *const runtime.Runtime, at: usize, blocks: u32, args: *hl.Args, s: abi.Stream) Error!void {
        if (s == util.counting) return;
        const cfg: hl.Config = .{ .grid = .{ .x = blocks }, .block = .{ .x = table[at].v.threads() } };
        try hl.launch(k.f.?[at], cfg, .{ .r = r, .handle = s }, args);
    }
};

/// What in `manifest` differs from the kernels the table launches, or null when every one is there as packed.
fn problem(manifest: []const u8) Error!?[]const u8 {
    var buf: [1 << 20]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
    @setEvalBranchQuota(100_000);
    inline for (table) |e| {
        const name = comptime e.v.symbol(e.out);
        const f = for (doc.functions) |f| {
            if (std.mem.eql(u8, f.name, name)) break f;
        } else return "no " ++ name;
        const i = f.interface;
        const want = interface(e.out);
        if (i.parameter_count != want[0] or i.binding_count != want[1] or i.constant_byte_length != want[2])
            return "the arguments of " ++ name;
        if (f.execution.workgroup_size[0] != e.v.threads()) return "the workgroup size of " ++ name;
    }
    return null;
}

const Manifest = struct {
    functions: []const struct {
        name: []const u8,
        interface: struct { parameter_count: u32, binding_count: u32, constant_byte_length: u32 },
        execution: struct { workgroup_size: [3]u32 },
    },
};

test "a variant's symbol, threads and blocks follow the kernel's geometry" {
    const v: Variant = .{ .k = 2560, .f = 2, .nt = 4 };
    try std.testing.expectEqualStrings("skinny_mxfp8_k2560_f2_t4_x4", comptime v.symbol(.sides));
    try std.testing.expectEqual(@as(u32, 512), v.threads());
    try std.testing.expectEqual(@as(u32, (32 + 20 + 1 + 1) * 5), v.blocks(&.{ 10240, 6144, 48, 48 }));
    const w: Variant = .{ .k = 6144, .f = 1, .nt = 2 };
    try std.testing.expectEqual(@as(u32, 1024), w.threads());
    try std.testing.expectEqual(@as(u32, 14 * 6), w.blocks(&.{2560}));
}

test "pick takes rows 1-64 by row fragments, and only the table's shapes" {
    var f: [table.len]Function = undefined;
    const k: Kernels = .{ .f = f };
    _ = &f;
    try std.testing.expectEqual(table[k.pick(16480, 2560, 20, .sides).?].v.f, 2);
    try std.testing.expectEqual(table[k.pick(2560, 6144, 64, .bf16).?].v.f, 4);
    try std.testing.expect(k.pick(2560, 6144, 65, .bf16) == null);
    try std.testing.expect(k.pick(2048, 2560, 4, .f32) == null);
}
