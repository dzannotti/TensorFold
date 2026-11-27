//! The Loom dense bf16 prompt GEMMs (zig/kernels/loom/dense_prompt_gemm): fp4l_wmma's BF16 bytes, faster where taken.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const util = @import("util.zig");

const Error = runtime.Error;
const Arg = affine.Arg;

/// The two block tiles (rows x columns) and their work-items, in kernels.zig's Loom order.
pub const Tile = enum(u1) {
    t128x128,
    t128x256,

    pub fn cols(t: Tile) usize {
        return if (t == .t128x128) 128 else 256;
    }
    pub fn threads(t: Tile) c_uint {
        return if (t == .t128x128) 256 else 512;
    }
};

const symbols = [2][:0]const u8{ "bf16_gemm_128x128", "bf16_gemm_128x256" };

/// The products they serve: Flash Next's hyper-connection down (with and without the inject rows) and up.
const shapes = [_]struct { n: c_int, k: c_int }{ .{ .n = 324, .k = 10240 }, .{ .n = 320, .k = 10240 }, .{ .n = 10240, .k = 320 } };

pub const Kernels = struct {
    /// Both tiles; null in a build without Loom kernels.
    f: ?[2]Function = null,

    /// Both functions of their Loom images, after each manifest shows the interface the launches pack.
    pub fn load(mods: [2]?Module, manifests: [2][]const u8) Error!Kernels {
        var f: [2]Function = undefined;
        for (mods, manifests, symbols, 0..) |m, doc, name, i| {
            const mod = m orelse return .{};
            if (try problem(doc, name, Tile.threads(@fromBackingInt(@intCast(i))))) |why| {
                std.log.err("bf16_gemm: the Loom image's manifest: {s}", .{why});
                return error.Invalid;
            }
            f[i] = try mod.function(name);
        }
        return .{ .f = f };
    }

    /// The tile for the plain BF16 prompt product `a` (fp32 out, the fold FP4-Local's BF16 decoder takes), or null.
    pub fn pick(k: *const Kernels, a: Arg) ?Tile {
        if (k.f == null or a.route.items != 0 or a.out == 0 or a.out16 != 0 or a.m < 1) return null;
        if (a.words % 16 != 0 or a.x % 16 != 0 or a.group != (if (@rem(a.k, 128) == 0) @as(c_int, 128) else 64)) return null;
        for (shapes) |s| if (s.n == a.n and s.k == a.k) return choose(a.m, a.n);
        return null;
    }

    /// out = x . W^T over `a` on tile `t`.
    pub fn go(k: *const Kernels, r: *const runtime.Runtime, t: Tile, a: Arg, s: abi.Stream) Error!void {
        if (s == util.counting) return;
        var args: hl.Args = .{};
        inline for (.{ a.x, a.words, a.out }) |p| try args.add(p);
        try args.add(a.m);
        try args.add(a.n);
        try args.add(a.k);
        const grid = util.dim(util.cdiv(a.n, t.cols()), util.cdiv(a.m, 128), 1);
        try hl.launch(k.f.?[@backingInt(t)], .{ .grid = grid, .block = .{ .x = t.threads() } }, .{ .r = r, .handle = s }, &args);
    }
};

/// Narrow products (the down projection) take the wide block from 1,024 rows, where x's reads bound them; the hipcc
/// tile keeps one block of rows or fewer, where it is as fast; everything else takes the 128 x 128 block.
fn choose(m: c_int, n: c_int) ?Tile {
    if (n >= 1024) return .t128x128;
    if (m <= 128) return null;
    return if (m >= 1024) .t128x256 else .t128x128;
}

/// What in `manifest` differs from `name` as the launches pack it (3 buffers, 3 i32, `threads` work-items), or null.
fn problem(manifest: []const u8, name: []const u8, threads: u32) Error!?[]const u8 {
    var buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
    const f = for (doc.functions) |f| {
        if (std.mem.eql(u8, f.name, name)) break f;
    } else return "the kernel is missing";
    const i = f.interface;
    if (i.parameter_count != 6 or i.binding_count != 3 or i.constant_byte_length != 12) return "the kernel's arguments";
    if (f.execution.workgroup_size[0] != threads) return "the kernel's workgroup size";
    return null;
}

const Manifest = struct {
    functions: []const struct {
        name: []const u8,
        interface: struct { parameter_count: u32, binding_count: u32, constant_byte_length: u32 },
        execution: struct { workgroup_size: [3]u32 },
    },
};

test "the manifest check takes the dense bf16 interface and names what differs" {
    const fmt = "{{\"functions\":[{{\"name\":\"bf16_gemm_128x256\",\"interface\":{{\"parameter_count\":{d},\"binding_count\":3,\"constant_byte_length\":12}},\"execution\":{{\"workgroup_size\":[{d},1,1]}}}}]}}";
    try std.testing.expect(try problem(std.fmt.comptimePrint(fmt, .{ 6, 512 }), "bf16_gemm_128x256", 512) == null);
    try std.testing.expectEqualStrings("the kernel's arguments", (try problem(std.fmt.comptimePrint(fmt, .{ 5, 512 }), "bf16_gemm_128x256", 512)).?);
    try std.testing.expectEqualStrings("the kernel's workgroup size", (try problem(std.fmt.comptimePrint(fmt, .{ 6, 256 }), "bf16_gemm_128x256", 512)).?);
    try std.testing.expectEqualStrings("the kernel is missing", (try problem(std.fmt.comptimePrint(fmt, .{ 6, 512 }), "bf16_gemm_128x128", 512)).?);
}

test "the down projection takes the wide block from 1,024 rows and the hipcc tile at 128 or fewer; up takes 128 x 128" {
    try std.testing.expectEqual(@as(?Tile, .t128x256), choose(4096, 324));
    try std.testing.expectEqual(@as(?Tile, .t128x128), choose(1000, 324));
    try std.testing.expectEqual(@as(?Tile, null), choose(128, 320));
    try std.testing.expectEqual(@as(?Tile, .t128x128), choose(1, 10240));
}
