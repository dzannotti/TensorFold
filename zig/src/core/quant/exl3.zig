//! EXL3 trellis weights (docs/recipes/exl3.md): int16 tiles of 16x16 codes, fp16 input and output scales; host and device side.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");
pub const format = @import("../exl3_format.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const id: types.Format = .exl3;
pub const decoder: types.Decoder = .exl3;

/// `quant_method: exl3`: every tensor carries its own width and codebook; `bpw` is the checkpoint's average.
pub const Config = struct { bpw: f64 };

/// The config's quantization object when its `quant_method` is "exl3", else null.
pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    const m = src.quantization.get("quant_method") orelse return null;
    if (m != .string or !std.mem.eql(u8, m.string, "exl3")) return null;
    const bpw: f64 = if (src.quantization.get("bpw")) |b| switch (b) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    } else 0;
    return .{ .bpw = bpw };
}

/// A projection as stored: trellis int16 [K/16, N/16, 16 bits], suh fp16 [K], svh fp16 [N], an optional fp16 bias [N].
pub const Host = struct {
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    bias: ?Tensor = null,
    width: format.Width,
    codebook: format.Codebook,

    pub fn k(h: Host) usize {
        return h.trellis.shape[0] * 16;
    }
    pub fn n(h: Host) usize {
        return h.trellis.shape[1] * 16;
    }
};

/// Whether `key` holds an EXL3 projection (its trellis).
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    return t.has(join(&buf, &.{ key, ".trellis" }));
}

/// fp16 scales under `scaled`, or the packed sign words of older checkpoints under `signed` expanded to fp16 +-1.
fn scales(a: Allocator, t: anytype, key: []const u8, scaled: []const u8, signed: []const u8, len: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const name = join(&buf, &.{ key, scaled });
    if (t.has(name)) {
        const s = try t.get(name);
        if (s.dtype != .f16 or s.numel() != len) return refuse(key, "EXL3 scales must be fp16 of the projection's width");
        return s;
    }
    const w = try t.get(join(&buf, &.{ key, signed }));
    if ((w.dtype != .i16 and w.dtype != .u16) or w.numel() * 16 != len) return refuse(key, "EXL3 sign words must be int16 [width / 16]");
    const words = try a.alloc(i16, len / 16);
    @memcpy(std.mem.sliceAsBytes(words), w.bytes);
    const out = try a.alloc(f16, len);
    format.unpackSigns(words, out);
    return .{ .dtype = .f16, .rank = 1, .shape = .{ len, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(out) };
}

/// The projection under `key`: its width from the trellis, its codebook from the marker tensor beside it.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    const trellis = try t.get(join(&buf, &.{ key, ".trellis" }));
    if ((trellis.dtype != .i16 and trellis.dtype != .u16) or trellis.rank != 3 or trellis.shape[2] % 8 != 0) return refuse(key, "EXL3 trellis must be int16 [K/16, N/16, 16 * bits]");
    const width: format.Width = .{ .halves = @intCast(trellis.shape[2] / 8) };
    const codebook: format.Codebook = if (t.has(join(&buf, &.{ key, ".mul1" }))) .mul1 else if (t.has(join(&buf, &.{ key, ".mcg" }))) .mcg else .inst3;
    if (width.halves < 3 or width.halves > 16 or (width.isHalf() and (codebook != .mul1 or width.halves > 7))) return refuse(key, "EXL3 width the kernels do not run");
    const k = trellis.shape[0] * 16;
    const n = trellis.shape[1] * 16;
    if (k % 128 != 0 or n % 128 != 0) return refuse(key, "EXL3 K and N must be multiples of 128");
    var h: Host = .{
        .trellis = trellis,
        .suh = try scales(a, t, key, ".suh", ".su", k),
        .svh = try scales(a, t, key, ".svh", ".sv", n),
        .width = width,
        .codebook = codebook,
    };
    h.trellis.dtype = .i16;
    const bias_key = join(&buf, &.{ key, ".bias" });
    if (t.has(bias_key)) {
        const b = try t.get(bias_key);
        if (b.numel() != n or !convert.isFloat(b.dtype)) return refuse(key, "EXL3 bias must be a float [N]");
        h.bias = if (b.dtype == .f16) b else try toF16(a, b);
    }
    return h;
}

fn toF16(a: Allocator, t: Tensor) !Tensor {
    const out = try a.alloc(f16, t.numel());
    for (out, 0..) |*o, i| o.* = @floatCast(convert.load(t.dtype, t.bytes, i));
    var r = t;
    r.dtype = .f16;
    r.bytes = std.mem.sliceAsBytes(out);
    return r;
}

/// Output rows N.
pub fn rows(h: Host) usize {
    return h.n();
}

pub fn bytes(h: Host) usize {
    return h.trellis.bytes.len + h.suh.bytes.len + h.svh.bytes.len + if (h.bias) |b| b.bytes.len else 0;
}

/// Both halves along K: tile rows and suh halves, each with the whole svh (whole Hadamard blocks stay whole).
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const tiles = p.trellis.shape[0];
    if ((tiles / 2) % 8 != 0 or tiles % 2 != 0) return refuse("fc", "EXL3 halves must keep whole 128-row blocks");
    var out: [2]Host = .{ p, p };
    for (&out, 0..) |*o, i| {
        o.trellis = try slice.takeRows(a, p.trellis, &.{.{ .from = i * tiles / 2, .to = (i + 1) * tiles / 2 }});
        o.suh = try slice.takeRows(a, p.suh, &.{.{ .from = i * p.k() / 2, .to = (i + 1) * p.k() / 2 }});
        if (i == 1) o.bias = null;
    }
    return out;
}

fn noCut(what: []const u8) slice.Error {
    std.log.err("{s}: EXL3 projections have no tensor-parallel cut", .{what});
    return error.UnevenSplit;
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    _ = .{ a, p, spans };
    return noCut("rows");
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    _ = .{ a, p, r };
    return noCut(what);
}

pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    _ = .{ a, p, r, part };
    return noCut("experts");
}

/// An uploaded projection: the tensors as stored.
pub const Device = struct { trellis: types.Buf, suh: types.Buf, svh: types.Buf, bias: ?types.Buf, width: format.Width, codebook: format.Codebook };

pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{
        .trellis = try u.tensor(h.trellis),
        .suh = try u.tensor(h.suh),
        .svh = try u.tensor(h.svh),
        .bias = if (h.bias) |b| try u.tensor(b) else null,
        .width = h.width,
        .codebook = h.codebook,
    };
}

/// What the kernels read: the tensors' addresses (bias 0 when absent), the width in half bits, the codebook, and the
/// stored output width the tiles are laid out for (a product may take only the first columns of it).
pub const Matrix = struct { trellis: u64, suh: u64, svh: u64, bias: u64, halves: u8, codebook: format.Codebook, columns: u32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

/// The projection's (N, K) and addresses; a stack of experts is not an EXL3 layout this backend reads.
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    if (stacked) return error.UnsupportedTables;
    return .{
        .n = @intCast(d.trellis.dim(1) * 16),
        .k = @intCast(d.trellis.dim(0) * 16),
        .matrix = .{ .trellis = d.trellis.ptr, .suh = d.suh.ptr, .svh = d.svh.ptr, .bias = if (d.bias) |b| b.ptr else 0, .halves = @intCast(d.width.halves), .codebook = d.codebook, .columns = @intCast(d.trellis.dim(1) * 16) },
    };
}

pub fn elements(h: Host) usize {
    return h.k() * h.n();
}

/// fp64 W as rows of outputs [N, K] (the format's own decoder, transposed); bias excluded.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const k = h.k();
    const n = h.n();
    if (out.len != k * n) return error.UnexpectedTensor;
    const gpa = std.heap.page_allocator;
    const words = gpa.alloc(i16, h.trellis.bytes.len / 2) catch return error.UnexpectedTensor;
    defer gpa.free(words);
    @memcpy(std.mem.sliceAsBytes(words), h.trellis.bytes);
    const suh = gpa.alloc(f16, k) catch return error.UnexpectedTensor;
    defer gpa.free(suh);
    @memcpy(std.mem.sliceAsBytes(suh), h.suh.bytes);
    const svh = gpa.alloc(f16, n) catch return error.UnexpectedTensor;
    defer gpa.free(svh);
    @memcpy(std.mem.sliceAsBytes(svh), h.svh.bytes);
    const w = gpa.alloc(f64, 3 * k * n) catch return error.UnexpectedTensor;
    defer gpa.free(w);
    format.dequantize(h.width, h.codebook, words, k / 16, n / 16, suh, svh, w[0 .. k * n], w[k * n ..]);
    for (0..n) |c| for (0..k) |r| {
        out[c * k + r] = w[r * n + c];
    };
}

/// A checkpoint as `read` sees it: tensors by name.
const Fixture = struct {
    names: []const []const u8,
    tensors: []const Tensor,

    pub fn has(f: Fixture, name: []const u8) bool {
        return f.find(name) != null;
    }

    pub fn get(f: Fixture, name: []const u8) error{MissingTensor}!Tensor {
        return f.tensors[f.find(name) orelse return error.MissingTensor];
    }

    fn find(f: Fixture, name: []const u8) ?usize {
        for (f.names, 0..) |x, i| if (std.mem.eql(u8, x, name)) return i;
        return null;
    }
};

test "a projection reads its width, codebook and sign words, and halves along K with svh whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const k = 256;
    const n = 128;
    const bits = 3;
    var trellis: [(k / 16) * (n / 16) * 16 * bits]i16 = undefined;
    for (&trellis, 0..) |*w, i| w.* = @bitCast(@as(u16, @truncate(i *% 2654435761)));
    var su: [k / 16]i16 = @splat(0);
    su[1] = 1; // element 16 is -1
    var svh: [n]f16 = @splat(0.5);
    const fx: Fixture = .{
        .names = &.{ "p.trellis", "p.su", "p.svh", "p.mcg" },
        .tensors = &.{
            .{ .dtype = .i16, .rank = 3, .shape = .{ k / 16, n / 16, 16 * bits, 1, 1 }, .bytes = std.mem.sliceAsBytes(&trellis) },
            .{ .dtype = .i16, .rank = 1, .shape = .{ k / 16, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&su) },
            .{ .dtype = .f16, .rank = 1, .shape = .{ n, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&svh) },
            .{ .dtype = .i32, .rank = 0, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &.{} },
        },
    };
    try std.testing.expect(matches(fx, "p"));
    const h = try read(a, fx, "p");
    try std.testing.expectEqual(format.Codebook.mcg, h.codebook);
    try std.testing.expectEqual(@as(u16, 2 * bits), h.width.halves);
    try std.testing.expectEqual(@as(usize, n), rows(h));
    const suh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, h.suh.bytes));
    try std.testing.expectEqual(@as(f16, -1), suh[16]);
    try std.testing.expectEqual(@as(f16, 1), suh[15]);
    // the halves' references are the whole one's input rows (W = diag(suh) H_K W_q H_N diag(svh) keeps K blocks apart)
    const whole = try a.alloc(f64, k * n);
    try reference(h, whole);
    const pair = try halves(a, h);
    for (pair, 0..) |half, i| {
        const part = try a.alloc(f64, k / 2 * n);
        try reference(half, part);
        for (0..n) |c| for (0..k / 2) |r| {
            try std.testing.expectApproxEqAbs(whole[c * k + i * k / 2 + r], part[c * k / 2 + r], 1e-12);
        };
    }
}
