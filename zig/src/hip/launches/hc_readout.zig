//! The Loom hyper-connection read-out (dense_prompt_gemm hc_*): the hipcc norm, gates, mix and BF16 tiles' bytes.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;

/// The compiled shapes: 4 streams of 2,560 (hc_norm.cxx K5_HC_DIMS); the down's columns 0..383 on the wide launch.
pub const streams = 4;
pub const dims = 2560;
const wide_cols = 384;
/// The down takes the wide launch (two column blocks, 256 + 128) from these rows, 128 x 128 blocks below.
pub const wide_rows = 1024;

pub const Kind = enum { norm, down_wide, down_narrow, up_mix };
pub const symbols = [4][:0]const u8{ "hc_norm", "hc_down_wide", "hc_down_narrow", "hc_up_mix" };
/// Parameters, buffers and constant bytes of each launch, and its work-items.
const interfaces = [4][4]u32{ .{ 9, 5, 16, 256 }, .{ 8, 4, 16, 512 }, .{ 8, 4, 16, 256 }, .{ 7, 4, 12, 512 } };

pub const Kernels = struct {
    /// norm, down (wide and narrow), up + mix; null in a build without Loom kernels.
    f: ?[4]Function = null,

    /// The functions of their Loom images, after each manifest shows the interface the launches pack.
    pub fn load(mods: [4]?Module, manifests: [4][]const u8) Error!Kernels {
        var f: [4]Function = undefined;
        for (mods, manifests, 0..) |m, doc, i| {
            const mod = m orelse return .{};
            if (try problem(doc, i)) |why| {
                std.log.err("hc_readout: the Loom image's manifest of {s}: {s}", .{ symbols[i], why });
                return error.Invalid;
            }
            f[i] = try mod.function(symbols[i]);
        }
        return .{ .f = f };
    }

    /// Whether a read-out of `s` streams of `d` dims, `low` gates and a `n_down`-row down projection is theirs.
    pub fn takes(k: *const Kernels, s: usize, d: usize, low: usize, n_down: usize) bool {
        return k.f != null and s == streams and d == dims and low % 64 == 0 and
            (n_down == low or n_down == low + streams) and n_down <= wide_cols;
    }

    /// h += bf16(branch inject) per stream when `branch` is set (in place), then x = the streams normed with `scale`.
    pub fn norm(k: *const Kernels, r: *const runtime.Runtime, h: u64, branch: ?u64, inject: ?u64, scale: u64, x: u64, rows: usize, eps: f32, s: abi.Stream) Error!void {
        var a: hl.Args = .{};
        const add: c_int = @intFromBool(branch != null);
        inline for (.{ h, branch orelse h, inject orelse h, scale, x }) |p| try a.add(p);
        inline for (.{ int(rows), @as(c_int, dims), eps, add }) |v| try a.add(v);
        return k.go(r, .norm, util.dim(rows, 1, 1), s, &a);
    }

    /// The down projection (bf16 [n][k] rows `w`) of x with the gates: act [rows][low], inject [rows][4] when n > low.
    pub fn down(k: *const Kernels, r: *const runtime.Runtime, x: u64, w: u64, act: u64, inject: ?u64, rows: usize, n: usize, low: usize, s: abi.Stream) Error!void {
        var a: hl.Args = .{};
        inline for (.{ x, w, act, inject orelse act }) |p| try a.add(p);
        inline for (.{ int(rows), int(n), @as(c_int, streams * dims), int(low) }) |v| try a.add(v);
        const wide = rows >= wide_rows;
        const grid = if (wide) util.dim(2, util.cdiv(rows, 128), 1) else util.dim(util.cdiv(n, 128), util.cdiv(rows, 128), 1);
        return k.go(r, if (wide) .down_wide else .down_narrow, grid, s, &a);
    }

    /// The up projection (bf16 [4 dims][low] rows `w`) of act, gated onto x, the streams mixed: mixed [rows][dims].
    pub fn upMix(k: *const Kernels, r: *const runtime.Runtime, act: u64, w: u64, x: u64, mixed: u64, rows: usize, low: usize, s: abi.Stream) Error!void {
        var a: hl.Args = .{};
        inline for (.{ act, w, x, mixed }) |p| try a.add(p);
        inline for (.{ int(rows), @as(c_int, dims), int(low) }) |v| try a.add(v);
        return k.go(r, .up_mix, util.dim(streams * dims / 256, util.cdiv(rows, 128), 1), s, &a);
    }

    fn go(k: *const Kernels, r: *const runtime.Runtime, which: Kind, grid: hl.Dim3, s: abi.Stream, a: *hl.Args) Error!void {
        if (s == util.counting) return;
        const i = @backingInt(which);
        try hl.launch(k.f.?[i], .{ .grid = grid, .block = .{ .x = interfaces[i][3] } }, .{ .r = r, .handle = s }, a);
    }
};

fn int(v: usize) c_int {
    return @intCast(v);
}

/// What in `manifest` differs from kernel `i` as the launches pack it, or null.
fn problem(manifest: []const u8, i: usize) Error!?[]const u8 {
    var buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
    const f = for (doc.functions) |f| {
        if (std.mem.eql(u8, f.name, symbols[i])) break f;
    } else return "the kernel is missing";
    const want = interfaces[i];
    const got = f.interface;
    if (got.parameter_count != want[0] or got.binding_count != want[1] or got.constant_byte_length != want[2])
        return "the kernel's arguments";
    if (f.execution.workgroup_size[0] != want[3]) return "the kernel's workgroup size";
    return null;
}

const Manifest = struct {
    functions: []const struct {
        name: []const u8,
        interface: struct { parameter_count: u32, binding_count: u32, constant_byte_length: u32 },
        execution: struct { workgroup_size: [3]u32 },
    },
};

test "the read-out's manifest check takes each kernel's interface and names what differs" {
    const fmt = "{{\"functions\":[{{\"name\":\"hc_up_mix\",\"interface\":{{\"parameter_count\":7,\"binding_count\":4,\"constant_byte_length\":{d}}},\"execution\":{{\"workgroup_size\":[512,1,1]}}}}]}}";
    try std.testing.expect(try problem(std.fmt.comptimePrint(fmt, .{12}), 3) == null);
    try std.testing.expectEqualStrings("the kernel's arguments", (try problem(std.fmt.comptimePrint(fmt, .{16}), 3)).?);
    try std.testing.expectEqualStrings("the kernel is missing", (try problem(std.fmt.comptimePrint(fmt, .{12}), 0)).?);
}

test "the read-out takes Flash Next's streams and down shapes" {
    const k: Kernels = .{ .f = @splat(undefined) };
    try std.testing.expect(k.takes(4, 2560, 320, 324));
    try std.testing.expect(k.takes(4, 2560, 320, 320));
    try std.testing.expect(!k.takes(4, 2048, 320, 324));
    try std.testing.expect(!k.takes(4, 2560, 320, 330));
    try std.testing.expect(!(Kernels{}).takes(4, 2560, 320, 324));
}
