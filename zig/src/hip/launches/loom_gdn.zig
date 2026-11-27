//! The Loom prompt DeltaNet recurrence (zig/kernels/loom/gdn_prompt): tf_gdn_prompt_128's bytes at any row count.

const std = @import("std");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;

const symbol: [:0]const u8 = "gdn_recur";
/// Parameters, buffers and constant bytes of the launch, and its work-items.
const interface = [4]u32{ 10, 7, 12, 128 };
/// State rows a block: half a value head.
const block_rows = 64;

pub const Kernels = struct {
    /// gdn_recur; null in a build without Loom kernels.
    f: ?Function = null,

    /// The function of its Loom image, after the manifest shows the interface the launch packs.
    pub fn load(mod: ?Module, manifest: []const u8) Error!Kernels {
        const m = mod orelse return .{};
        if (try problem(manifest)) |why| {
            std.log.err("loom_gdn: the Loom image's manifest of {s}: {s}", .{ symbol, why });
            return error.Invalid;
        }
        return .{ .f = try m.function(symbol) };
    }

    /// Whether a recurrence of these shapes is theirs: one stream, 128 x 128 heads, no per-row states.
    pub fn takes(k: *const Kernels, batch: c_int, dk: c_int, dv: c_int, states: bool) bool {
        return k.f != null and batch == 1 and dk == 128 and dv == 128 and !states;
    }

    /// `length` rows of the delta rule from `state` (updated in place) into y, as tf_gated_delta's prompt route.
    pub fn run(k: *const Kernels, r: *const runtime.Runtime, q: util.CF, kk: util.CF, v: util.CF, gate: util.CF, beta: util.CF, state: util.F, y: util.F, length: c_int, key_heads: c_int, value_heads: c_int, s: util.S) Error!void {
        if (s == util.counting) return;
        var a: hl.Args = .{};
        inline for (.{ util.ad(q), util.ad(kk), util.ad(v), util.ad(gate), util.ad(beta), util.ad(state), util.ad(y), length, key_heads, value_heads }) |x| try a.add(x);
        const grid = util.dim(128 / block_rows, value_heads, 1);
        try hl.launch(k.f.?, .{ .grid = grid, .block = .{ .x = interface[3] } }, .{ .r = r, .handle = s }, &a);
    }
};

/// What in `manifest` differs from the kernel as the launch packs it, or null.
fn problem(manifest: []const u8) Error!?[]const u8 {
    var buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
    const f = for (doc.functions) |f| {
        if (std.mem.eql(u8, f.name, symbol)) break f;
    } else return "the kernel is missing";
    const got = f.interface;
    if (got.parameter_count != interface[0] or got.binding_count != interface[1] or got.constant_byte_length != interface[2])
        return "the kernel's arguments";
    if (f.execution.workgroup_size[0] != interface[3]) return "the kernel's workgroup size";
    return null;
}

const Manifest = struct {
    functions: []const struct {
        name: []const u8,
        interface: struct { parameter_count: u32, binding_count: u32, constant_byte_length: u32 },
        execution: struct { workgroup_size: [3]u32 },
    },
};

test "the manifest check takes gdn_recur's interface and names what differs" {
    const fmt = "{{\"functions\":[{{\"name\":\"gdn_recur\",\"interface\":{{\"parameter_count\":10,\"binding_count\":7,\"constant_byte_length\":{d}}},\"execution\":{{\"workgroup_size\":[{d},1,1]}}}}]}}";
    try std.testing.expect(try problem(std.fmt.comptimePrint(fmt, .{ 12, 128 })) == null);
    try std.testing.expectEqualStrings("the kernel's arguments", (try problem(std.fmt.comptimePrint(fmt, .{ 8, 128 }))).?);
    try std.testing.expectEqualStrings("the kernel's workgroup size", (try problem(std.fmt.comptimePrint(fmt, .{ 12, 256 }))).?);
}
