//! The Loom NVFP4 grouped-expert prompt kernels (zig/kernels/loom/nvfp4_grouped_expert): the hipcc tiles' bytes.

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

/// The shapes the kernels are compiled for (README: K4_K, K4_N, K4_DK, K4_DN): another one is not theirs.
pub const gate_up = .{ .n = 640, .k = 2560 };
pub const down = .{ .n = 2560, .k = 640 };

const symbols = [2][:0]const u8{ "nvfp4_expert_gate_up", "nvfp4_expert_down" };

pub const Kernels = struct {
    /// gate|up with SwiGLU, then down; null in a build without Loom kernels.
    f: ?[2]Function = null,

    /// Both functions of their Loom images, after each manifest shows the interface the launches pack.
    pub fn load(mods: [2]?Module, manifests: [2][]const u8) Error!Kernels {
        var f: [2]Function = undefined;
        for (mods, manifests, symbols, 0..) |m, doc, name, i| {
            const mod = m orelse return .{};
            if (try problem(doc, name)) |why| {
                std.log.err("nvfp4_experts: the Loom image's manifest: {s}", .{why});
                return error.Invalid;
            }
            f[i] = try mod.function(name);
        }
        return .{ .f = f };
    }

    /// Whether the routed gate|up stack of `a` (2 x 640 by 2560, gate and up globals, no SwiGLU limit) is theirs.
    pub fn takesGateUp(k: *const Kernels, a: Arg, limit: f32) bool {
        return k.f != null and a.route.items != 0 and a.n == 2 * gate_up.n and a.k == gate_up.k and
            a.global_layout == .expert_pair and limit == 0;
    }

    /// Whether the routed down product of `a` (2560 by 640, a global an expert, fp32 out) is theirs.
    pub fn takesDown(k: *const Kernels, a: Arg) bool {
        return k.f != null and a.route.items != 0 and a.n == down.n and a.k == down.k and
            a.global_layout == .expert and (a.out != 0) != (a.out16 != 0);
    }

    /// act = bf16(silu(gate) x up) into a.out16 (pairs, 640) over a plan's `items`; `a.m` the most rows an item holds.
    pub fn gateUpGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
        return k.go(r, 0, a, a.out16, gate_up.n, 64, items, s);
    }

    /// The fp32 down sums into a.out (pairs, 2560) over a plan's `items`.
    pub fn downGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
        return k.go(r, 1, a, if (a.out16 != 0) a.out16 else a.out, down.n, 128, items, s);
    }

    fn go(k: *const Kernels, r: *const runtime.Runtime, which: usize, a: Arg, out: u64, n: c_int, cols: usize, items: c_int, s: abi.Stream) Error!void {
        if (s == util.counting) return;
        if (a.m < 1 or items < 1 or a.route.members == 0 or a.route.x_div < 1) return error.Invalid;
        var args: hl.Args = .{};
        inline for (.{ a.x, a.words, a.scale.p, a.global, a.route.items, a.route.members, out }) |p| try args.add(p);
        try args.add(a.route.x_div);
        try args.add(if (which == 1 and a.out16 != 0) -n else n);
        try args.add(a.k);
        const grid = util.dim(util.cdiv(n, cols), util.cdiv(a.m, 128), items);
        try hl.launch(k.f.?[which], .{ .grid = grid, .block = .{ .x = 256 } }, .{ .r = r, .handle = s }, &args);
    }
};

/// What in `manifest` differs from `name` as the launches pack it (7 buffers, 3 i32, 256 work-items), or null.
fn problem(manifest: []const u8, name: []const u8) Error!?[]const u8 {
    var buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
    const f = for (doc.functions) |f| {
        if (std.mem.eql(u8, f.name, name)) break f;
    } else return "the kernel is missing";
    const i = f.interface;
    if (i.parameter_count != 10 or i.binding_count != 7 or i.constant_byte_length != 12) return "the kernel's arguments";
    if (f.execution.workgroup_size[0] != 256) return "the kernel's workgroup size";
    return null;
}

const Manifest = struct {
    functions: []const struct {
        name: []const u8,
        interface: struct { parameter_count: u32, binding_count: u32, constant_byte_length: u32 },
        execution: struct { workgroup_size: [3]u32 },
    },
};

test "the manifest check takes the grouped-expert interface and names what differs" {
    const fmt = "{{\"functions\":[{{\"name\":\"nvfp4_expert_down\",\"interface\":{{\"parameter_count\":{d},\"binding_count\":7,\"constant_byte_length\":12}},\"execution\":{{\"workgroup_size\":[256,1,1]}}}}]}}";
    try std.testing.expect(try problem(std.fmt.comptimePrint(fmt, .{10}), "nvfp4_expert_down") == null);
    try std.testing.expectEqualStrings("the kernel's arguments", (try problem(std.fmt.comptimePrint(fmt, .{9}), "nvfp4_expert_down")).?);
    try std.testing.expectEqualStrings("the kernel is missing", (try problem(std.fmt.comptimePrint(fmt, .{10}), "nvfp4_expert_gate_up")).?);
}
