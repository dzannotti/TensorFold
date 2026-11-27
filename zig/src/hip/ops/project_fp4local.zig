//! FP4-Local's products behind project.zig: a prompt span takes the matrix tile, everything else the stream tile.

const t = @import("types.zig");
const quant = @import("core").quant;
const fp4 = @import("../launches/fp4local.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const Projection = quant.Projection;
const dev = quant.fp4local_device;
const int = t.int;

/// The FP4-Local formats' common argument: the format, and the product's planes.
pub const Weights = struct { fmt: fp4.Format, words: u64, scale: u64 = 0, global: u64 = 0, layout: fp4.GlobalLayout = .none, n: u32, k: u32 };

/// A projection's FP4-Local weights, or null; `stack`: an expert over a plan; bf16 only when marked (Matrix.fold).
pub fn weightsOf(proj: Projection, stack: bool) ?Weights {
    return switch (proj.handle) {
        .nvfp4 => |h| .{
            .fmt = .nvfp4,
            .words = h.words,
            .scale = h.scales,
            .global = h.globals,
            .layout = if (stack) (if (h.per == 2) .expert_pair else .expert) else if (h.per == 1) .tensor else return null,
            .n = proj.n,
            .k = proj.k,
        },
        .mxfp8 => |h| .{ .fmt = .mxfp8, .words = h.codes, .scale = h.scales, .n = proj.n, .k = proj.k },
        .dense => |h| if (h.fold and h.dtype == .bf16) .{ .fmt = .bf16, .words = h.weight, .n = proj.n, .k = proj.k } else null,
        else => null,
    };
}

fn ltWmma(m: c_int) bool {
    const std = @import("std");
    const raw = std.c.getenv("LT_WMMA") orelse return false;
    const from = std.fmt.parseInt(c_int, std.mem.span(raw), 10) catch return false;
    return from > 0 and m >= from;
}

fn ltQmmf(m: c_int) bool {
    const std = @import("std");
    const raw = std.c.getenv("LT_QMMF") orelse return false;
    const from = std.fmt.parseInt(c_int, std.mem.span(raw), 10) catch return false;
    return from > 0 and m >= from;
}

fn argOf(w: Weights, x: Tensor, m: usize) fp4.Arg {
    return .{
        .x = x.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = 0 },
        .bias = .{ .p = 0, .kind = 0 },
        .out = 0,
        .m = int(m),
        .n = int(w.n),
        .k = int(w.k),
        .bits = fp4.Format.chunkWords(w.fmt),
        .group = if (w.fmt == .bf16) fp4.bf16Fold(int(w.k)) else 128,
        .fp16 = 0,
        .global = w.global,
        .global_layout = w.layout,
    };
}

/// x (m, k) bf16 times `w`: fp32 when `f32_out`, else bf16 rounded once from the fp32 sum.
pub fn project(o: Ops, x: Tensor, w: Weights, m: usize, f32_out: bool) Error!Tensor {
    if (m == 0 or x.kind != .bf16 or w.layout == .expert or w.layout == .expert_pair) return error.BadShape;
    const kind: t.Kind = if (f32_out) .f32 else .bf16;
    const out = try o.arena.take(m * w.n * kind.size());
    var a = argOf(w, x, m);
    if (f32_out) a.out = out else a.out16 = out;
    try run(o, w.fmt, a, 1, 0);
    return .{ .ptr = out, .kind = kind };
}

fn run(o: Ops, fmt: fp4.Format, a: fp4.Arg, items: c_int, expert_rows: c_int) Error!void {
    const k = &o.l.fp4;
    if (o.prefill) return k.promptGo(o.l.r, fmt, a, items, expert_rows, o.stream);
    if (items == 1 and try k.qmmfGo(o.l.r, fmt, a, o.stream)) return;
    if (a.route.items == 0 and ltWmma(a.m)) return k.promptGo(o.l.r, fmt, a, items, expert_rows, o.stream);
    return k.streamGo(o.l.r, fmt, a, .{}, items, o.stream);
}

/// Up to four decode products of one x that share K and format in one stream launch, bf16 outs; false otherwise.
pub fn group(o: Ops, x: Tensor, ws: []const Weights, m: usize, outs: []Tensor) Error!bool {
    if (ltWmma(int(m))) return false;
    if (o.prefill or ws.len < 2 or ws.len > 4 or m == 0 or x.kind != .bf16) return false;
    for (ws) |w| if (w.fmt != ws[0].fmt or w.k != ws[0].k or w.layout != .none) return false;
    var sides: fp4.StreamSides = .{ .count = @intCast(ws.len), .out_half = 1 };
    var widest: u32 = 0;
    for (ws, outs[0..ws.len], 0..) |w, *out, i| {
        out.* = .{ .ptr = try o.arena.take(m * w.n * 2), .kind = .bf16 };
        sides.words[i] = w.words;
        sides.scale[i] = w.scale;
        sides.out[i] = out.ptr;
        sides.n[i] = int(w.n);
        widest = @max(widest, w.n);
    }
    var a = argOf(ws[0], x, m);
    a.n = int(widest);
    if (fp4.Kernels.qmmfTakes(ws[0].fmt, int(m))) {
        if (fp4.Kernels.stitched(sides, a.k)) |st| {
            if (try o.l.fp4.qmmfSt(o.l.r, ws[0].fmt, a, st, o.stream)) return true;
        }
        for (ws, outs[0..ws.len]) |w, out| {
            var one = argOf(w, x, m);
            one.out16 = out.ptr;
            try run(o, w.fmt, one, 1, 0);
        }
        return true;
    }
    try o.l.fp4.streamGo(o.l.r, ws[0].fmt, a, sides, 1, o.stream);
    return true;
}

/// A routed stack over a plan: out (pairs, n) fp32; `rows` the most an item holds.
pub fn routed(o: Ops, x: Tensor, w: Weights, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize, expert_rows: usize) Error!u64 {
    if (w.fmt != .nvfp4 or x.kind != .bf16 or count == 0) return error.BadShape;
    const out = try o.arena.of(f32, pairs * w.n);
    var a = argOf(w, x, rows);
    a.out = out;
    a.route = .{ .items = items, .members = members, .x_div = int(x_div) };
    try run(o, .nvfp4, a, int(count), int(expert_rows));
    return out;
}

/// silu(gate) x up of a routed (gate | up) stack in one launch: the stream tile in decode, Loom's on a prompt, or null.
pub fn routedAct(o: Ops, x: Tensor, w: Weights, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize, limit: f32) Error!?Tensor {
    if (w.fmt != .nvfp4 or x.kind != .bf16 or w.n % 2 != 0 or count == 0) return null;
    var a = argOf(w, x, rows);
    a.route = .{ .items = items, .members = members, .x_div = int(x_div) };
    if (o.prefill and !o.l.fp4.experts.takesGateUp(a, limit)) return null;
    const out = try o.arena.take(pairs * (w.n / 2) * 2);
    a.out16 = out;
    if (o.prefill) {
        try o.l.fp4.experts.gateUpGo(o.l.r, a, int(count), o.stream);
    } else try o.l.fp4.streamGo(o.l.r, .nvfp4, a, .{ .pair_cols = int(w.n / 2), .limit = limit }, int(count), o.stream);
    return .{ .ptr = out, .kind = .bf16 };
}

test "a bf16 matrix takes the FP4-Local tiles only when the loader marks it; NVFP4 globals follow the call" {
    const std = @import("std");
    const plain: Projection = .{ .n = 4, .k = 320, .handle = .{ .dense = .{ .weight = 64, .dtype = .bf16 } } };
    try std.testing.expect(weightsOf(plain, false) == null);
    const folded: Projection = .{ .n = 4, .k = 320, .handle = .{ .dense = .{ .weight = 64, .dtype = .bf16, .fold = true } } };
    try std.testing.expectEqual(fp4.Format.bf16, weightsOf(folded, false).?.fmt);
    try std.testing.expectEqual(@as(c_int, 64), argOf(weightsOf(folded, false).?, .{ .ptr = 0, .kind = .bf16 }, 1).group);
    const experts: Projection = .{ .n = 1280, .k = 2560, .handle = .{ .nvfp4 = .{ .words = 1, .scales = 2, .globals = 3, .per = 2 } } };
    try std.testing.expectEqual(fp4.GlobalLayout.expert_pair, weightsOf(experts, true).?.layout);
    try std.testing.expect(weightsOf(experts, false) == null);
    const one: Projection = .{ .n = 2560, .k = 640, .handle = .{ .nvfp4 = .{ .words = 1, .scales = 2, .globals = 3, .per = 1 } } };
    try std.testing.expectEqual(fp4.GlobalLayout.tensor, weightsOf(one, false).?.layout);
    try std.testing.expectEqual(fp4.GlobalLayout.expert, weightsOf(one, true).?.layout);
}
