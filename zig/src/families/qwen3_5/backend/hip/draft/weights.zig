//! The DFlash2 drafter's weights on the device: projections in affine-4 (quantized here, in the row kernel's tiled layout)
//! or bf16, norms in fp32.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Config = @import("config.zig").Config;


/// Affine-4 group 64 in draft.hip's tiled layout (tf_dft_q4_rows): words, then bf16 scales and biases, `n` columns.
pub const Q4 = struct { words: u64, scales: u64, biases: u64, n: usize, k: usize };

/// A product's weights: affine-4 through the drafter's row kernel, or bf16 through its bf16 one.
pub const Linear = union(enum) {
    q4: Q4,
    bf16: struct { w: u64, n: usize, k: usize },

    pub fn rows(l: Linear) usize {
        return switch (l) {
            .q4 => |p| p.n,
            .bf16 => |b| b.n,
        };
    }
};

/// MLX rows (n a multiple of 32) into the row kernel's layout: 32-column blocks, a group of a block 32 columns' 8 words
/// back to back, the tables (block, group, 32); `T` the table type.
pub fn tile(comptime T: type, n: usize, k: usize, words: []const u32, scales: []const T, biases: []const T, w_out: []u32, s_out: []T, b_out: []T) void {
    const groups = k / 64;
    for (0..n) |c| for (0..groups) |g| {
        const at = (c / 32 * groups + g) * 32 + c % 32;
        @memcpy(w_out[at * 8 ..][0..8], words[c * (k / 8) + g * 8 ..][0..8]);
        s_out[at] = scales[c * groups + g];
        b_out[at] = biases[c * groups + g];
    };
}

pub const Layer = struct {
    input_norm: u64, // fp32
    post_norm: u64,
    qkv: Linear, // q, k and v of a row side by side
    o: Linear,
    q_norm: u64, // fp32
    k_norm: u64,
    gate: Linear,
    up: Linear,
    down: Linear,
    attn_conv: Linear, // kernel_projection: hidden -> the dynamic coefficients
    attn_base: u64, // base_kernel (2 branches, 2 lags, hidden) bf16
    mlp_conv: Linear,
    mlp_base: u64,
};

pub const Weights = struct {
    fc: Linear, // the taps' concatenation -> hidden
    hidden_norm: u64,
    final_norm: u64,
    selector: Linear, // hidden_projection: hidden -> rank
    kv: Linear, // every layer's k then v: what an absorbed row writes into the rings, in one product
    k_norms: u64, // every layer's k_norm, fp32 (layers, head_dim)
    predecessor: []u16, // codebooks (vocab, rank) bf16, on the host
    successor: []u16,
    layers: []Layer,
    arena: hip.DeviceBuffer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.arena.free();
        gpa.free(w.predecessor);
        gpa.free(w.successor);
        gpa.free(w.layers);
    }
};

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

/// MLX affine-4 group 64 of one (n, k) bf16 matrix into words (n, k / 8) and scale and bias (n, k / 64), low nibble first.
fn quantize(w: []align(1) const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const groups = k / 64;
    for (0..n) |r| for (0..groups) |g| {
        const row = w[r * k + g * 64 ..][0..64];
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (row) |h| {
            lo = @min(lo, bf16(h));
            hi = @max(hi, bf16(h));
        }
        const scale = toBf16((hi - lo) / 15.0);
        const bias = toBf16(lo);
        scales[r * groups + g] = scale;
        biases[r * groups + g] = bias;
        const s = bf16(scale);
        const b = bf16(bias);
        const inv = if (s == 0) 0.0 else 1.0 / s;
        for (0..8) |j| {
            var word: u32 = 0;
            for (0..8) |i| {
                const q = std.math.clamp(@round((bf16(row[j * 8 + i]) - b) * inv), 0.0, 15.0);
                word |= @as(u32, @intFromFloat(q)) << @intCast(4 * i);
            }
            words[(r * k + g * 64) / 8 + j] = word;
        }
    };
}

/// Places the weights in one device buffer: a first pass counts (no buffer), the second quantizes and uploads.
const Loader = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    bits: u8,
    arena: ?hip.DeviceBuffer,
    at: usize = 0,

    fn take(l: *Loader, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, l.at, 256);
        l.at = at + bytes;
        return if (l.arena) |a| a.base() + at else 0;
    }

    fn put(l: *Loader, p: u64, bytes: []const u8) !void {
        if (l.arena) |a| try a.upload(p - a.base(), bytes);
    }

    /// The bf16 matrices `names`, each (n_i, k), stacked by rows into one product.
    fn linear(l: *Loader, names: []const []const u8, k: usize) !Linear {
        var n: usize = 0;
        for (names) |name| n += (try l.ck.get(name)).shape[0];
        for (names) |name| _ = try l.ck.expect(name, .bf16, &.{ (try l.ck.get(name)).shape[0], k });
        if (l.bits == 0) {
            const w = l.take(n * k * 2);
            var row: usize = 0;
            for (names) |name| {
                const t = try l.ck.get(name);
                try l.put(w + row * k * 2, t.bytes);
                row += t.shape[0];
            }
            return .{ .bf16 = .{ .w = w, .n = n, .k = k } };
        }
        const groups = k / 64;
        const words = l.take(n * k / 2);
        if (k % 64 != 0) return error.UnsupportedDraftConfig;
        const scales = l.take(n * groups * 2);
        const biases = l.take(n * groups * 2);
        if (l.arena != null) {
            var row: usize = 0;
            for (names) |name| {
                const t = try l.ck.get(name);
                const rows = t.shape[0];
                const qw = try l.gpa.alloc(u32, rows * k / 8);
                defer l.gpa.free(qw);
                const qs = try l.gpa.alloc(u16, rows * groups);
                defer l.gpa.free(qs);
                const qb = try l.gpa.alloc(u16, rows * groups);
                defer l.gpa.free(qb);
                if (rows % 32 != 0) return error.UnsupportedDraftConfig;
                quantize(std.mem.bytesAsSlice(u16, t.bytes), rows, k, qw, qs, qb);
                const tw = try l.gpa.alloc(u32, rows * k / 8);
                defer l.gpa.free(tw);
                const ts = try l.gpa.alloc(u16, rows * groups);
                defer l.gpa.free(ts);
                const tb = try l.gpa.alloc(u16, rows * groups);
                defer l.gpa.free(tb);
                tile(u16, rows, k, qw, qs, qb, tw, ts, tb);
                // a part of a stack starts on a whole block (32 rows): its tiled bytes sit where its rows would
                try l.put(words + row * k / 2, std.mem.sliceAsBytes(tw));
                try l.put(scales + row * groups * 2, std.mem.sliceAsBytes(ts));
                try l.put(biases + row * groups * 2, std.mem.sliceAsBytes(tb));
                row += rows;
            }
        }
        return .{ .q4 = .{ .words = words, .scales = scales, .biases = biases, .n = n, .k = k } };
    }

    /// The bf16 vectors `names` of `len` each, widened to fp32 back to back.
    fn norms(l: *Loader, names: []const []const u8, len: usize) !u64 {
        const p = l.take(names.len * len * 4);
        if (l.arena == null) return p;
        const wide = try l.gpa.alloc(f32, len);
        defer l.gpa.free(wide);
        for (names, 0..) |name, i| {
            const t = try l.ck.expect(name, .bf16, &.{len});
            for (wide, std.mem.bytesAsSlice(u16, t.bytes)) |*x, h| x.* = bf16(h);
            try l.put(p + i * len * 4, std.mem.sliceAsBytes(wide));
        }
        return p;
    }

    fn norm(l: *Loader, name: []const u8, len: usize) !u64 {
        return l.norms(&.{name}, len);
    }

    /// A bf16 tensor as is.
    fn raw(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.ck.expect(name, .bf16, shape);
        const p = l.take(t.bytes.len);
        try l.put(p, t.bytes);
        return p;
    }
};

fn hostBf16(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, shape: []const usize) ![]u16 {
    const t = try ck.expect(name, .bf16, shape);
    const out = try gpa.alloc(u16, t.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

/// A layer tensor's name.
fn named(buf: []u8, i: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, suffix }) catch unreachable;
}

/// Every device weight placed by `l` (counted when it has no buffer): the same order both passes.
fn place(l: *Loader, c: Config, layers: []Layer) !struct { fc: Linear, hidden_norm: u64, final_norm: u64, selector: Linear, kv: Linear, k_norms: u64 } {
    const fc = try l.linear(&.{"fc.weight"}, c.hidden * c.n_taps);
    const hidden_norm = try l.norm("hidden_norm.weight", c.hidden);
    const final_norm = try l.norm("norm.weight", c.hidden);
    const selector = try l.linear(&.{"candidate_selector.hidden_projection.weight"}, c.hidden);
    var bufs: [2 * 16][96]u8 = undefined;
    var kv_names: [2 * 16][]const u8 = undefined;
    var kn_names: [16][]const u8 = undefined;
    for (0..c.layers) |i| {
        kv_names[2 * i] = named(&bufs[2 * i], i, "self_attn.k_proj.weight");
        kv_names[2 * i + 1] = named(&bufs[2 * i + 1], i, "self_attn.v_proj.weight");
    }
    const kv = try l.linear(kv_names[0 .. 2 * c.layers], c.hidden);
    var kn_bufs: [16][96]u8 = undefined;
    for (0..c.layers) |i| kn_names[i] = named(&kn_bufs[i], i, "self_attn.k_norm.weight");
    const k_norms = try l.norms(kn_names[0..c.layers], c.head_dim);
    for (layers, 0..) |*y, i| {
        var b: [3][96]u8 = undefined;
        y.* = .{
            .input_norm = try l.norm(named(&b[0], i, "input_layernorm.weight"), c.hidden),
            .post_norm = try l.norm(named(&b[0], i, "post_attention_layernorm.weight"), c.hidden),
            .qkv = try l.linear(&.{ named(&b[0], i, "self_attn.q_proj.weight"), named(&b[1], i, "self_attn.k_proj.weight"), named(&b[2], i, "self_attn.v_proj.weight") }, c.hidden),
            .o = try l.linear(&.{named(&b[0], i, "self_attn.o_proj.weight")}, c.qWidth()),
            .q_norm = try l.norm(named(&b[0], i, "self_attn.q_norm.weight"), c.head_dim),
            .k_norm = try l.norm(named(&b[0], i, "self_attn.k_norm.weight"), c.head_dim),
            .gate = try l.linear(&.{named(&b[0], i, "mlp.gate_proj.weight")}, c.hidden),
            .up = try l.linear(&.{named(&b[0], i, "mlp.up_proj.weight")}, c.hidden),
            .down = try l.linear(&.{named(&b[0], i, "mlp.down_proj.weight")}, c.intermediate),
            .attn_conv = try l.linear(&.{named(&b[0], i, "attention_conv.kernel_projection.weight")}, c.hidden),
            .attn_base = try l.raw(named(&b[0], i, "attention_conv.base_kernel"), &.{ 2, 2, c.hidden }),
            .mlp_conv = try l.linear(&.{named(&b[0], i, "mlp_conv.kernel_projection.weight")}, c.hidden),
            .mlp_base = try l.raw(named(&b[0], i, "mlp_conv.base_kernel"), &.{ 2, 2, c.hidden }),
        };
        if (y.qkv.rows() != c.qWidth() + 2 * c.kvWidth() or y.attn_conv.rows() != c.dynamic() or y.gate.rows() != c.intermediate) return error.UnexpectedTensor;
    }
    if (fc.rows() != c.hidden or selector.rows() != c.rank or kv.rows() != 2 * c.layers * c.kvWidth()) return error.UnexpectedTensor;
    return .{ .fc = fc, .hidden_norm = hidden_norm, .final_norm = final_norm, .selector = selector, .kv = kv, .k_norms = k_norms };
}

/// The device bytes `load` takes for the checkpoint in `dir` at `bits` (4 or 0).
pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: Config, bits: u8) !usize {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .bits = bits, .arena = null };
    const layers = try gpa.alloc(Layer, c.layers);
    defer gpa.free(layers);
    _ = try place(&l, c, layers);
    return l.at;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, dir: []const u8, c: Config, bits: u8) !Weights {
    if (bits != 0 and bits != 4) return error.DrafterBitsUnsupported;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(layers);
    var count: Loader = .{ .gpa = gpa, .ck = &ck, .bits = bits, .arena = null };
    _ = try place(&count, c, layers);
    var arena = try hip.DeviceBuffer.alloc(r, count.at);
    errdefer arena.free();
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .bits = bits, .arena = arena };
    const top = try place(&l, c, layers);
    const predecessor = try hostBf16(gpa, &ck, "candidate_selector.predecessor_codebook", &.{ c.vocab, c.rank });
    errdefer gpa.free(predecessor);
    const successor = try hostBf16(gpa, &ck, "candidate_selector.successor_codebook", &.{ c.vocab, c.rank });
    if (ck.unused() != 0) return error.UnexpectedTensor;
    return .{ .fc = top.fc, .hidden_norm = top.hidden_norm, .final_norm = top.final_norm, .selector = top.selector, .kv = top.kv, .k_norms = top.k_norms, .predecessor = predecessor, .successor = successor, .layers = layers, .arena = arena };
}

test "affine-4 of a group: a flat one is exact, a ramp lands each value on its own step" {
    var flat: [64]u16 = @splat(0x3f80); // 1.0: scale 0, bias 1, every nibble 0
    var words: [8]u32 = undefined;
    var scales: [1]u16 = undefined;
    var biases: [1]u16 = undefined;
    quantize(&flat, 1, 64, &words, &scales, &biases);
    try std.testing.expectEqual(@as(u16, 0), scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), biases[0]);
    try std.testing.expectEqual(@as(u32, 0), words[0]);
    var ramp: [64]u16 = undefined;
    for (&ramp, 0..) |*x, i| x.* = toBf16(@floatFromInt(i % 16)); // 0..15: scale 1, bias 0, nibble = value
    quantize(&ramp, 1, 64, &words, &scales, &biases);
    try std.testing.expectEqual(@as(f32, 1), bf16(scales[0]));
    try std.testing.expectEqual(@as(u32, 0x76543210), words[0]);
    try std.testing.expectEqual(@as(u32, 0xfedcba98), words[1]);
}
