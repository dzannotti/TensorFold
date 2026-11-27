//! FP4-Local tensors on the host as core/quant forms: MXFP8 admitted and stitched, bf16 as stored, experts baked.
const std = @import("std");
const core = @import("core");
const mxfp8_read = @import("mxfp8_read.zig");
const nvfp4_read = @import("nvfp4_read.zig");

const quant = core.quant;
const Tensor = core.safetensors.Tensor;
const Host = quant.Host;
const fm = quant.fp4local_matrix;
const nvfp4_format = quant.nvfp4_format;
const mxfp8_format = quant.mxfp8_format;
const refuse = core.modelopt.refuse;
const Allocator = std.mem.Allocator;

/// A checkpoint (anything with `get` and `has`) under the names mxfp8_read, nvfp4_read and nvfp4_slice call.
pub fn Store(comptime T: type) type {
    return struct {
        t: T,

        pub fn get(s: @This(), name: []const u8) !Tensor {
            return s.t.get(name);
        }

        pub fn tensor(s: @This(), name: []const u8) !Tensor {
            return s.t.get(name);
        }

        pub fn hold(s: @This(), name: []const u8) !Tensor {
            return s.t.get(name);
        }

        pub fn holdAhead(s: @This(), name: []const u8) !Tensor {
            return s.t.get(name);
        }

        pub fn has(s: @This(), name: []const u8) bool {
            return s.t.has(name);
        }
    };
}

pub fn store(t: anytype) Store(@TypeOf(t)) {
    return .{ .t = t };
}

/// How `stem` is stored: NVFP4 (a weight_scale_2), MXFP8 (an e4m3 weight beside its e8m0 scales) or bf16.
pub const Kind = enum { nvfp4, mxfp8, bf16 };

pub fn kindOf(s: anytype, stem: []const u8) !Kind {
    var buf: [512]u8 = undefined;
    if (s.has(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{stem}))) return .nvfp4;
    if (s.has(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{stem}))) return .mxfp8;
    return .bf16;
}

/// `stem` as stored: NVFP4 baked, MXFP8 admitted (mxfp8_format.check), bf16 as `.dense`; any other dtype refused.
pub fn proj(a: Allocator, s: anytype, stem: []const u8) !Host {
    var buf: [512]u8 = undefined;
    switch (try kindOf(s, stem)) {
        .nvfp4 => {
            const p = try a.create(nvfp4_format.Projection);
            p.* = try nvfp4_format.bake(a, try core.nvfp4_slice.stored(s, stem), stem, null);
            return .{ .nvfp4 = fm.nvfp4.one(p) };
        },
        .mxfp8 => {
            const w = try s.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{stem}));
            const e = try s.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{stem}));
            try mxfp8_format.check(stem, w, e, null);
            return .{ .mxfp8 = fm.mxfp8.of(w.dim(0), w.dim(1), w.bytes, e.bytes) };
        },
        .bf16 => return .{ .dense = .{ .weight = try bf16Weight(s, stem) } },
    }
}

fn bf16Weight(s: anytype, stem: []const u8) !Tensor {
    var buf: [512]u8 = undefined;
    const w = try s.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{stem}));
    if (w.dtype != .bf16 or w.rank != 2) return refuse(null, error.UnexpectedTensor, "{s}.weight: {t} rank {d} is not bf16 [n, k]", .{ stem, w.dtype, w.rank });
    return w;
}

/// The output rows of `parts` under `prefix` (`{prefix}{part}`) stacked in order into one product, all of one kind.
pub fn stitched(a: Allocator, s: anytype, prefix: []const u8, parts: []const []const u8) !Host {
    var buf: [512]u8 = undefined;
    const kind = try kindOf(s, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, parts[0] }));
    for (parts[1..]) |p| if (try kindOf(s, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, p })) != kind)
        return refuse(null, error.UnexpectedTensor, "{s}{s}: stored unlike {s}{s}, so they share no product", .{ prefix, p, prefix, parts[0] });
    switch (kind) {
        .mxfp8 => {
            const m = try mxfp8_read.readStitched(a, s, prefix, parts);
            const name = try std.fmt.bufPrint(&buf, "{s}{s} and the rows under it", .{ prefix, parts[0] });
            try mxfp8_format.check(name, quant.types.view(.f8_e4m3, &.{ m.n, m.k }, m.codes), quant.types.view(.u8, &.{ m.n, m.k / mxfp8_format.block }, m.scales), null);
            return .{ .mxfp8 = fm.mxfp8.of(m.n, m.k, m.codes, m.scales) };
        },
        .bf16 => {
            var w: [8]Tensor = undefined;
            for (parts, w[0..parts.len]) |p, *t| t.* = try bf16Weight(s, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, p }));
            return .{ .dense = .{ .weight = try joinRows(a, w[0..parts.len]) } };
        },
        .nvfp4 => return refuse(null, error.UnexpectedTensor, "{s}{s}: NVFP4 projections are not stitched", .{ prefix, parts[0] }),
    }
}

/// Rows of `ts` (one dtype and width) stacked in order.
pub fn joinRows(a: Allocator, ts: []const Tensor) !Tensor {
    var out = ts[0];
    var len: usize = 0;
    var rows: usize = 0;
    for (ts) |t| {
        if (t.dtype != out.dtype or t.rank != 2 or t.shape[1] != out.shape[1]) return error.UnexpectedTensor;
        len += t.bytes.len;
        rows += t.shape[0];
    }
    const bytes = try a.alloc(u8, len);
    var at: usize = 0;
    for (ts) |t| {
        @memcpy(bytes[at..][0..t.bytes.len], t.bytes);
        at += t.bytes.len;
    }
    out.shape[0] = rows;
    out.bytes = bytes;
    return out;
}

/// The router's bf16 rows (E routed, then the shared expert's gate), and the same values widened to fp32.
pub const Router = struct { rows16: Tensor, rows32: Tensor };

pub fn router(a: Allocator, s: anytype, gate: []const u8, shared_gate: []const u8, experts: usize, hidden: usize) !Router {
    const g = try bf16Weight(s, gate);
    const sg = try bf16Weight(s, shared_gate);
    if (!g.is(.bf16, &.{ experts, hidden }) or !sg.is(.bf16, &.{ 1, hidden }))
        return refuse(null, error.UnexpectedTensor, "{s}: router rows are not bf16 [E, hidden] and [1, hidden]", .{gate});
    const rows16 = try joinRows(a, &.{ g, sg });
    const wide = try a.alloc(u8, rows16.bytes.len * 2);
    for (0..rows16.bytes.len / 2) |i| std.mem.writeInt(u32, wide[4 * i ..][0..4], @as(u32, std.mem.readInt(u16, rows16.bytes[2 * i ..][0..2], .little)) << 16, .little);
    var rows32 = rows16;
    rows32.dtype = .f32;
    rows32.bytes = wide;
    return .{ .rows16 = rows16, .rows32 = rows32 };
}

/// A stored word baked (nvfp4_format.permute: even codes to the low half), as two nibble swaps.
pub fn permute(w: u32) u32 {
    var x = w;
    var t = (x ^ (x >> 4)) & 0x00F000F0;
    x ^= t ^ (t << 4);
    t = (x ^ (x >> 8)) & 0x0000FF00;
    x ^= t ^ (t << 8);
    return x;
}

/// One expert's baked bytes in each plane: fused gate | up words and scales, down words and scales.
pub const Sizes = struct {
    up_words: usize,
    up_scales: usize,
    down_words: usize,
    down_scales: usize,

    pub fn expert(z: Sizes) usize {
        return z.up_words + z.up_scales + z.down_words + z.down_scales;
    }
};

/// A layer's routed experts as stored (gate, up and down views each), baked a chunk of experts at a time.
pub const Experts = struct {
    prefix: []const u8,
    views: []nvfp4_read.View,
    count: usize,
    width: usize,
    dims: usize,
    threads: usize,

    /// `{prefix}.experts.{e}.{gate,up,down}_proj` for every expert, shapes and globals refused as nvfp4_read does.
    pub fn read(a: Allocator, s: anytype, prefix: []const u8, count: usize, width: usize, dims: usize, threads: usize) !Experts {
        const views = try a.alloc(nvfp4_read.View, 3 * count);
        var buf: [512]u8 = undefined;
        for (0..count) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |part, j| {
            const name = try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ prefix, e, part });
            const n, const k = if (j < 2) .{ width, dims } else .{ dims, width };
            views[3 * e + j] = nvfp4_read.viewOf(s, name, n, k, false) catch |err| return refuse(null, err, "{s}: not NVFP4 [{d}, {d}] ({t})", .{ name, n, k, err });
            if (!std.math.isFinite(quant.fp4local_decode.nvfp4Global(views[3 * e + j].global)))
                return refuse(null, error.UnsupportedQuantization, "{s}: weight_scale_2 {e} overflows the epilogue", .{ name, views[3 * e + j].global });
        };
        return .{ .prefix = prefix, .views = views, .count = count, .width = width, .dims = dims, .threads = threads };
    }

    pub fn sizes(x: Experts) Sizes {
        const g = nvfp4_format.group;
        return .{
            .up_words = 2 * x.width * x.dims / 2,
            .up_scales = 2 * x.width * (x.dims / g),
            .down_words = x.dims * x.width / 2,
            .down_scales = x.dims * (x.width / g),
        };
    }

    /// Fused globals [E][2] (gate, up) and down's [E].
    pub fn globals(x: Experts, up: []f32, down: []f32) void {
        for (0..x.count) |e| {
            up[2 * e] = x.views[3 * e].global;
            up[2 * e + 1] = x.views[3 * e + 1].global;
            down[e] = x.views[3 * e + 2].global;
        }
    }

    /// Experts [lo, hi) baked into `out`, each plane's experts in turn: up words and scales, down words and scales.
    pub fn bake(x: Experts, lo: usize, hi: usize, out: []u8) !void {
        const z = x.sizes();
        if (lo >= hi or hi > x.count or out.len < (hi - lo) * z.expert()) return error.UnexpectedTensor;
        const m = hi - lo;
        const planes: Planes = .{
            .up_words = out[0 .. m * z.up_words],
            .up_scales = out[m * z.up_words ..][0 .. m * z.up_scales],
            .down_words = out[m * (z.up_words + z.up_scales) ..][0 .. m * z.down_words],
            .down_scales = out[m * (z.up_words + z.up_scales + z.down_words) ..][0 .. m * z.down_scales],
        };
        const t_n = @max(1, @min(x.threads, nvfp4_read.max_threads, m));
        const per = (m + t_n - 1) / t_n;
        var works: [nvfp4_read.max_threads]Work = undefined;
        var handles: [nvfp4_read.max_threads]?std.Thread = @splat(null);
        for (0..t_n) |t| {
            works[t] = .{ .x = &x, .planes = planes, .first = lo, .lo = lo + @min(m, t * per), .hi = lo + @min(m, (t + 1) * per) };
            handles[t] = std.Thread.spawn(.{}, Work.run, .{&works[t]}) catch blk: {
                works[t].run();
                break :blk null;
            };
        }
        for (handles[0..t_n]) |h| if (h) |th| th.join();
        for (works[0..t_n]) |w| if (w.failed) |err| return refuse(null, err, "{s}.experts.{d}: {t}", .{ x.prefix, w.at, err });
    }
};

const Planes = struct { up_words: []u8, up_scales: []u8, down_words: []u8, down_scales: []u8 };

/// Experts [lo, hi) of a chunk that starts at expert `first`, baked by one thread.
const Work = struct {
    x: *const Experts,
    planes: Planes,
    first: usize,
    lo: usize,
    hi: usize,
    at: usize = 0,
    failed: ?anyerror = null,

    fn run(w: *Work) void {
        w.pack() catch |err| {
            w.failed = err;
        };
    }

    fn pack(w: *Work) !void {
        const x = w.x;
        const z = x.sizes();
        const each = z.up_scales / 2;
        for (w.lo..w.hi) |e| {
            w.at = e;
            const i = e - w.first;
            const halves = [_]usize{ 0, z.up_words / 2 };
            for (0..2) |j| {
                const v = x.views[3 * e + j];
                try words(w.planes.up_words[i * z.up_words + halves[j] ..][0 .. z.up_words / 2], v.codes);
                const scales = w.planes.up_scales[i * z.up_scales + j * each ..][0..each];
                try scalesInto(v, x.width, x.dims, scales);
            }
            const d = x.views[3 * e + 2];
            try words(w.planes.down_words[i * z.down_words ..][0..z.down_words], d.codes);
            try scalesInto(d, x.dims, x.width, w.planes.down_scales[i * z.down_scales ..][0..z.down_scales]);
        }
    }
};

/// Stored code bytes (input 2j in byte j's low nibble) as baked little-endian words.
fn words(out: []u8, codes: []const u8) !void {
    if (out.len != codes.len or codes.len % 4 != 0) return error.UnexpectedTensor;
    var i: usize = 0;
    while (i < codes.len) : (i += 4) std.mem.writeInt(u32, out[i..][0..4], permute(std.mem.readInt(u32, codes[i..][0..4], .little)), .little);
}

/// A view's e4m3 scales [n, k / 16] (CSF decoded) into `out`; a NaN scale is refused, as nvfp4_format.bake does.
fn scalesInto(v: nvfp4_read.View, n: usize, k: usize, out: []u8) !void {
    const got = try nvfp4_read.scalesOf(v, n, k, out);
    if (got.ptr != out.ptr) @memcpy(out, got);
    for (out) |b| if (b & 0x7f == 0x7f) return error.UnsupportedQuantization;
}

test "the two-swap permute is nvfp4_format's nibble order" {
    var rng = std.Random.DefaultPrng.init(0x4f50);
    for (0..1 << 16) |_| {
        const w = rng.random().int(u32);
        try std.testing.expectEqual(nvfp4_format.permute(w), permute(w));
    }
    for (0..16) |c| try std.testing.expectEqual(nvfp4_format.permute(@intCast(c * 0x11111111)), permute(@intCast(c * 0x11111111)));
}

test {
    _ = @import("modelopt_host_layer_test.zig");
}
