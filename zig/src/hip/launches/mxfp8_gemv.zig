//! The Loom MXFP8 decode GEMV (zig/kernels/loom/mxfp8_gemv): fp4l_stream's sum order and bytes, where it is faster.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;

/// What a launch writes: one fp32 or bf16 [m][n] output, or bf16 split at three columns into four (a group's sides).
pub const Out = enum { f32, bf16, sides };

/// A compiled variant (gen.py's spec R:K:L:CB:WAVES:G:P:sKS): only `k` and `lanes` reach the bits.
pub const Variant = struct {
    rows: u8,
    k: u16,
    lanes: u8,
    split: u8 = 1,
    cb: u8 = 1,
    waves: u8 = 4,
    /// Rounds a loop trip; 0 is every round in a straight line.
    g: u8 = 0,
    p: u8 = 1,

    fn rounds(v: Variant) u32 {
        return @as(u32, v.k) / 32 / v.lanes;
    }

    /// The kernel's symbol for `out`: gen.py's name of the spec.
    pub fn symbol(comptime v: Variant, comptime out: Out) [:0]const u8 {
        const g = if (v.g == 0) v.rounds() else v.g;
        const end = switch (out) {
            .f32 => "",
            .bf16 => "_bf16",
            .sides => "_x4",
        };
        const fmt = "mxfp8_gemv_r{d}_k{d}_l{d}_s{d}_c{d}_w{d}_g{d}_p{d}{s}";
        return std.fmt.comptimePrint(fmt, .{ v.rows, v.k, v.lanes, v.split, v.cb, v.waves, g, v.p, end });
    }

    /// Output columns a workgroup covers.
    pub fn columns(v: Variant) u32 {
        return (32 / (@as(u32, v.lanes) / v.split)) * v.cb * (v.waves / v.split) * v.p;
    }

    pub fn threads(v: Variant) u32 {
        return 32 * @as(u32, v.waves);
    }
};

/// Where a variant replaces the stream tile: the checkpoint's product (n by k, all sides), its output and row tile.
pub const Entry = struct { n: u32, out: Out, v: Variant };

/// The variants and the shapes they beat fp4l_stream at; gen.py's VARIANTS builds each, `load` refuses a missing one.
pub const table = groups(16480) ++ groups(13952) ++ singles(1280, k2560) ++ singles(2560, k6144) ++ singles(2560, k640);

/// The stitched row groups at K 2560 (qkv | z | b | a, q | k | v | index): bf16 sides.
fn groups(n: u32) [4]Entry {
    return .{
        .{ .n = n, .out = .sides, .v = .{ .rows = 1, .k = 2560, .lanes = 16 } },
        .{ .n = n, .out = .sides, .v = .{ .rows = 2, .k = 2560, .lanes = 16 } },
        .{ .n = n, .out = .sides, .v = .{ .rows = 4, .k = 2560, .lanes = 16 } },
        .{ .n = n, .out = .sides, .v = .{ .rows = 8, .k = 2560, .lanes = 16, .g = 1 } },
    };
}

/// A single product's row tiles 1, 2, 4 and 8: their K, lanes, split and rounds a trip.
const Singles = struct { k: u16, lanes: u8, g: [4]u8 };
const k2560: Singles = .{ .k = 2560, .lanes = 16, .g = .{ 1, 1, 1, 1 } };
const k6144: Singles = .{ .k = 6144, .lanes = 32, .g = .{ 2, 2, 2, 2 } };
const k640: Singles = .{ .k = 640, .lanes = 4, .g = .{ 1, 1, 1, 1 } };

/// A single product (shared expert, attention and DeltaNet out): fp32 and bf16 outputs at each row tile.
fn singles(n: u32, s: Singles) [8]Entry {
    var e: [8]Entry = undefined;
    for (0..4) |i| {
        const v: Variant = .{ .rows = @as(u8, 1) << @intCast(i), .k = s.k, .lanes = s.lanes, .split = 2, .g = s.g[i] };
        e[2 * i] = .{ .n = n, .out = .f32, .v = v };
        e[2 * i + 1] = .{ .n = n, .out = .bf16, .v = v };
    }
    return e;
}

/// The lanes a column of fp4l_stream takes at `k` (launches/fp4local.zig `lanes`): the variant must take the same.
fn streamLanes(k: u32) u32 {
    const chunks = k >> 5;
    var l: u5 = 0;
    while (l < 5 and chunks % (@as(u32, 2) << l) == 0) l += 1;
    return @as(u32, 1) << l;
}

comptime {
    for (table) |e| if (e.v.lanes != streamLanes(e.v.k)) @compileError("a Loom variant's lanes differ from the stream tile's");
}

/// The arguments each output's kernel takes: (m, n[, c1, c2, c3]) as i32, then w, e, x and the outputs.
fn interface(out: Out) [3]u32 {
    return if (out == .sides) .{ 12, 7, 20 } else .{ 6, 4, 8 };
}

pub const Kernels = struct {
    /// Each table entry's kernel; null in a build without Loom kernels.
    f: ?[table.len]Function = null,

    /// The functions of `m` (the Loom image), after its artifact manifest shows each one as the launches pack it.
    pub fn load(m: ?Module, manifest: []const u8) Error!Kernels {
        const mod = m orelse return .{};
        if (try problem(manifest)) |why| {
            std.log.err("mxfp8_gemv: the Loom image's manifest: {s}", .{why});
            return error.Invalid;
        }
        var f: [table.len]Function = undefined;
        inline for (table, 0..) |e, i| f[i] = try mod.function(comptime e.v.symbol(e.out));
        return .{ .f = f };
    }

    /// The entry for `m` rows of an n by k product into `out`, or null where the stream tile keeps it.
    pub fn pick(k: *const Kernels, n: c_int, kk: c_int, m: c_int, out: Out) ?usize {
        if (k.f == null or m < 1 or m > 8) return null;
        var best: ?usize = null;
        for (table, 0..) |e, i| {
            if (e.n != n or e.v.k != kk or e.out != out or e.v.rows < m) continue;
            if (best == null or e.v.rows < table[best.?].v.rows) best = i;
        }
        return best;
    }

    /// x (m, k) bf16 against e4m3 codes `w` [n][k] and e8m0 scales `e` [n][k / 32] into `out`: fp32, or bf16 (`half`).
    pub fn go(k: *const Kernels, r: *const runtime.Runtime, at: usize, m: c_int, n: c_int, w: u64, e: u64, x: u64, out: u64, half: bool, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(m);
        try args.add(n);
        inline for (.{ w, e, x, out }) |p| try args.add(p);
        if (table[at].out != (if (half) Out.bf16 else Out.f32)) return error.Invalid;
        return k.launch(r, at, n, &args, s);
    }

    /// `go` into four bf16 outputs: columns [0, cuts[0]) to outs[0], [cuts[0], cuts[1]) to outs[1], and so on to n.
    pub fn goSides(k: *const Kernels, r: *const runtime.Runtime, at: usize, m: c_int, n: c_int, cuts: [3]c_int, w: u64, e: u64, x: u64, out: [4]u64, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(m);
        try args.add(n);
        for (cuts) |c| try args.add(c);
        inline for (.{ w, e, x }) |p| try args.add(p);
        for (out) |p| try args.add(p);
        if (table[at].out != .sides) return error.Invalid;
        return k.launch(r, at, n, &args, s);
    }

    fn launch(k: *const Kernels, r: *const runtime.Runtime, at: usize, n: c_int, args: *hl.Args, s: abi.Stream) Error!void {
        if (s == util.counting) return;
        const v = table[at].v;
        const grid: hl.Dim3 = .{ .x = @intCast(util.cdiv(n, v.columns())) };
        const f = k.f.?[at];
        try hl.launch(f, .{ .grid = grid, .block = .{ .x = v.threads() } }, .{ .r = r, .handle = s }, args);
    }
};

/// What in `manifest` differs from the kernels the table launches, or null when every one is there as packed.
fn problem(manifest: []const u8) Error!?[]const u8 {
    var buf: [1 << 20]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    const doc = std.json.parseFromSliceLeaky(Manifest, fba.allocator(), manifest, opts) catch return "it does not parse";
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

test "a variant's symbol is gen.py's name and its grid covers its columns" {
    const v: Variant = .{ .rows = 4, .k = 6144, .lanes = 32, .split = 2, .cb = 2 };
    try std.testing.expectEqualStrings("mxfp8_gemv_r4_k6144_l32_s2_c2_w4_g6_p1_bf16", comptime v.symbol(.bf16));
    try std.testing.expectEqualStrings("mxfp8_gemv_r4_k6144_l32_s2_c2_w4_g6_p1_x4", comptime v.symbol(.sides));
    try std.testing.expectEqual(@as(u32, 2 * 2 * 2), v.columns());
    try std.testing.expectEqual(@as(u32, 16), streamLanes(2560));
    try std.testing.expectEqual(@as(u32, 32), streamLanes(6144));
    try std.testing.expectEqual(@as(u32, 4), streamLanes(640));
}

test "the manifest check takes the launches' interfaces and names what differs" {
    const a = std.testing.allocator;
    var doc: std.ArrayList(u8) = .empty;
    defer doc.deinit(a);
    for (0..2) |bad| {
        doc.clearRetainingCapacity();
        try doc.appendSlice(a, "{\"functions\":[");
        inline for (table, 0..) |e, i| {
            const want = interface(e.out);
            const params = if (bad == 1 and i == 0) want[0] - 1 else want[0];
            const one = "{{\"name\":\"{s}\",\"interface\":{{\"parameter_count\":{d},\"binding_count\":{d},\"constant_byte_length\":{d}}},\"execution\":{{\"workgroup_size\":[{d},1,1]}}}}";
            if (i > 0) try doc.append(a, ',');
            try doc.print(a, one, .{ comptime e.v.symbol(e.out), params, want[1], want[2], e.v.threads() });
        }
        try doc.appendSlice(a, "]}");
        const got = try problem(doc.items);
        if (bad == 0) try std.testing.expect(got == null) else try std.testing.expect(std.mem.startsWith(u8, got.?, "the arguments of"));
    }
    try std.testing.expectEqualStrings("it does not parse", (try problem("{")).?);
}
