//! The Loom sparse prompt attention and block indexer (zig/kernels/loom/sparse_attention): the hipcc kernels' bytes.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;
const CF = util.CF;
const C = util.C;
const CI = util.CI;
const F = util.F;
const I = util.I;
const S = util.S;
const ad = util.ad;

/// The compiled shapes (README: Geometry): 12 query heads a KV head, 256 dims, 4 parts; indexer 4 heads of 128 dims.
pub const group = 12;
pub const dims = 256;
pub const parts = 4;
pub const index_heads = 4;
pub const index_dims = 128;

const Kind = enum { attend, scores, select };
const symbols = [3][:0]const u8{ "sparse_attend", "idx_scores", "idx_select" };
/// Parameters, buffers and constant bytes of each launch, and its work-items.
const interfaces = [3][4]u32{ .{ 14, 7, 36, 128 }, .{ 10, 4, 28, 256 }, .{ 10, 6, 24, 256 } };

pub const Kernels = struct {
    /// attend, scores, select; null in a build without Loom kernels.
    f: ?[3]Function = null,

    /// The functions of their Loom images, after each manifest shows the interface the launches pack.
    pub fn load(mods: [3]?Module, manifests: [3][]const u8) Error!Kernels {
        var f: [3]Function = undefined;
        for (mods, manifests, 0..) |m, doc, i| {
            const mod = m orelse return .{};
            if (try problem(doc, i)) |why| {
                std.log.err("loom_attention: the Loom image's manifest of {s}: {s}", .{ symbols[i], why });
                return error.Invalid;
            }
            f[i] = try mod.function(symbols[i]);
        }
        return .{ .f = f };
    }

    /// Whether a layer of these heads and a bf16 cache is theirs.
    pub fn takes(k: *const Kernels, heads: usize, kv_heads: usize, head_dim: usize, ih: usize, id: usize) bool {
        return k.f != null and kv_heads > 0 and heads == group * kv_heads and head_dim == dims and ih == index_heads and
            id == index_dims;
    }

    /// Each row's block scores (tf_fn_idx_scores): rows past `top` complete blocks score their complete blocks.
    pub fn scores(k: *const Kernels, r: *const runtime.Runtime, q: CF, pooled: C, complete: CI, out: F, rows: c_int, blocks: c_int, stride: i64, top: c_int, s: S) Error!void {
        if (blocks == 0) return;
        var a: hl.Args = .{};
        inline for (.{ ad(q), ad(pooled), ad(complete), ad(out), stride, rows, @as(c_int, index_heads), @as(c_int, index_dims), blocks, top }) |v| try a.add(v);
        return k.go(r, .scores, util.dim(util.cdiv(blocks, 256), util.cdiv(rows, 16), 1), s, &a);
    }

    /// Each row's keys (tf_fn_idx_select): its `top` best blocks and its tail, else its first ends[r] keys.
    pub fn select(k: *const Kernels, r: *const runtime.Runtime, sc: CF, stride: i64, complete: CI, ends: CI, keys: I, key_stride: i64, count: I, sparse: I, rows: c_int, top: c_int, ratio: c_int, s: S) Error!void {
        var a: hl.Args = .{};
        inline for (.{ ad(sc), ad(complete), ad(ends), ad(keys), ad(count), ad(sparse), stride, key_stride, top, ratio }) |v| try a.add(v);
        return k.go(r, .select, util.dim(rows, 1, 1), s, &a);
    }

    /// Each row's heads over its listed (or first count[r]) keys: tf_fn_sparse_lanes12_bf16 and merge in one launch.
    pub fn attend(k: *const Kernels, r: *const runtime.Runtime, q: CF, kc: C, vc: C, ids: CI, id_stride: i64, count: CI, sparse: CI, out: F, rows: c_int, heads: c_int, kv_heads: c_int, cap: i64, scale: f32, s: S) Error!void {
        var a: hl.Args = .{};
        inline for (.{ ad(q), ad(kc), ad(vc), ad(ids), ad(count), ad(sparse), ad(out), id_stride, cap, heads, kv_heads, @as(c_int, dims), scale, @as(c_int, parts) }) |v| try a.add(v);
        return k.go(r, .attend, util.dim(kv_heads, rows, 1), s, &a);
    }

    fn go(k: *const Kernels, r: *const runtime.Runtime, which: Kind, grid: hl.Dim3, s: S, a: *hl.Args) Error!void {
        if (s == util.counting) return;
        const i = @backingInt(which);
        try hl.launch(k.f.?[i], .{ .grid = grid, .block = .{ .x = interfaces[i][3] } }, .{ .r = r, .handle = s }, a);
    }
};

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

test "the manifest check takes each kernel's interface and names what differs" {
    const fmt = "{{\"functions\":[{{\"name\":\"idx_select\",\"interface\":{{\"parameter_count\":10,\"binding_count\":6,\"constant_byte_length\":{d}}},\"execution\":{{\"workgroup_size\":[256,1,1]}}}}]}}";
    try std.testing.expect(try problem(std.fmt.comptimePrint(fmt, .{24}), 2) == null);
    try std.testing.expectEqualStrings("the kernel's arguments", (try problem(std.fmt.comptimePrint(fmt, .{20}), 2)).?);
    try std.testing.expectEqualStrings("the kernel is missing", (try problem(std.fmt.comptimePrint(fmt, .{24}), 0)).?);
}
