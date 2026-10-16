//! azampatti's Qwen3.8-Flash-Next INT4-AutoRound checkpoint (GPTQ int4 routed experts and lm_head, block-FP8 dense
//! linears and shared expert, bf16 elsewhere, an FP8 n-gram table in ple-table/) on the HIP loader: its tensors under
//! the MLX layout's names (`open`), GPTQ as MLX 4-bit words (`gptq`, exact), the rest as the loader asks (`read`).
const std = @import("std");
const core = @import("core");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const Allocator = std.mem.Allocator;
const refuse = quant.types.refuse;

/// The checkpoint's prefixes as the MLX layout spells them.
const renames = [_][2][]const u8{
    .{ "model.language_model.", "language_model.model." },
    .{ "lm_head.", "language_model.lm_head." },
    .{ "mtp.", "language_model.mtp." },
};

fn rename(a: Allocator, k: []const u8) ![]const u8 {
    for (renames) |r| if (std.mem.startsWith(u8, k, r[0])) return std.mem.concat(a, u8, &.{ r[1], k[r[0].len..] });
    return k;
}

/// `dir`'s tensors as the index places them (a name in several files is read from the index's file: the healed shared
/// expert, the AutoRound head) and the n-gram rows of ple-table/, all under the MLX layout's names.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8) !Checkpoint {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    errdefer ck.close();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const map = (doc.object.get("weight_map") orelse return error.InvalidFlashIndex).object;
    for (ck.files.items) |*f| try canon(f, map, false);
    const sub = try std.fs.path.join(a, &.{ dir, "ple-table" });
    var d = try std.Io.Dir.cwd().openDir(io, sub, .{ .iterate = true });
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    for (names.items) |n| {
        try ck.add(sub, n);
        try canon(&ck.files.items[ck.files.items.len - 1], null, true);
    }
    return ck;
}

/// A file's names renamed; those the index places in another file dropped, and with `table` all but the n-gram rows.
fn canon(f: *core.safetensors.File, map: ?std.json.ObjectMap, table: bool) !void {
    const a = f.arena.allocator();
    const base = std.fs.path.basename(f.path);
    var names: core.safetensors.Header = .empty;
    for (f.names.keys(), f.names.values()) |k, v| {
        if (table and std.mem.indexOf(u8, k, ".ngram_embedding.") == null) continue;
        if (map) |m| if (m.get(k)) |w| if (w != .string or !std.mem.eql(u8, w.string, base)) continue;
        try names.put(a, try rename(a, k), v);
    }
    f.names = names;
}

fn named(a: Allocator, key: []const u8, suffix: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ key, suffix });
}

fn words32(t: Tensor) []align(1) const u32 {
    return @as([*]align(1) const u32, @ptrCast(t.bytes.ptr))[0 .. t.bytes.len / 4];
}

fn halfs(t: Tensor) []align(1) const u16 {
    return @as([*]align(1) const u16, @ptrCast(t.bytes.ptr))[0 .. t.bytes.len / 2];
}

/// GPTQ `key`'s shape: inputs K and outputs N, checked against AutoGPTQ v1's symmetric 4-bit groups of 128.
const Gptq = struct { qweight: Tensor, scales: Tensor, k: usize, n: usize };

fn gptqOf(a: Allocator, t: anytype, key: []const u8) !Gptq {
    const qw = try t.get(try named(a, key, ".qweight"));
    const sc = try t.get(try named(a, key, ".scales"));
    const qz = try t.get(try named(a, key, ".qzeros"));
    if (qw.dtype != .i32 or qw.rank != 2 or sc.dtype != .f16 or sc.rank != 2 or qz.dtype != .i32 or qz.rank != 2) return refuse(key, "not GPTQ int4 (qweight i32, scales fp16, qzeros i32)");
    const k = qw.shape[0] * 8;
    const n = qw.shape[1];
    if (k % 128 != 0 or sc.shape[0] != k / 128 or sc.shape[1] != n or qz.shape[0] != k / 128 or qz.shape[1] * 8 != n) return refuse(key, "GPTQ tensors do not fit 4 bits in groups of 128");
    // v1 stores the zero point less one: 7 everywhere is the symmetric 8
    for (words32(qz)) |z| if (z != 0x77777777) return refuse(key, "GPTQ zero points are not all 8 (asymmetric)");
    return .{ .qweight = qw, .scales = sc, .k = k, .n = n };
}

/// One GPTQ matrix into MLX words [N, K/8] (a word's eight codes keep their order) and fp16 tables [N, K/128]: scale s
/// and bias -8 s (exact in fp16), so the kernels' s q + b is s (q - 8), the GPTQ weight, exactly in fp32.
fn gptqInto(g: Gptq, words: []u32, scales: []u16, biases: []u16) void {
    const kw = g.k / 8;
    const groups = g.k / 128;
    const qw = words32(g.qweight);
    const sc = halfs(g.scales);
    for (0..kw) |j| for (0..g.n) |c| {
        words[c * kw + j] = qw[j * g.n + c];
    };
    for (0..groups) |i| for (0..g.n) |c| {
        const s: f16 = @bitCast(sc[i * g.n + c]);
        scales[c * groups + i] = @bitCast(s);
        biases[c * groups + i] = @bitCast(s * -8.0);
    };
}

fn mlxHost(words: []u32, scales: []u16, biases: []u16, lead: ?usize, n: usize, k: usize, bits: u8, group: u16, tables: quant.DType) quant.Host {
    const w: Tensor = if (lead) |e| .{ .dtype = .i32, .rank = 3, .shape = .{ e, n, k * bits / 32, 1, 1 }, .bytes = std.mem.sliceAsBytes(words) } else .{ .dtype = .i32, .rank = 2, .shape = .{ n, k * bits / 32, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(words) };
    var s: Tensor = w;
    s.dtype = tables;
    s.shape[s.rank - 1] = k / group;
    var b = s;
    s.bytes = std.mem.sliceAsBytes(scales);
    b.bytes = std.mem.sliceAsBytes(biases);
    return .{ .mlx = .{ .words = w, .scales = s, .biases = b, .bits = bits, .group = group } };
}

/// GPTQ `key` as an MLX 4-bit projection of group 128 with fp16 tables (gptqInto).
pub fn gptq(a: Allocator, t: anytype, key: []const u8) !quant.Host {
    const g = try gptqOf(a, t, key);
    const words = try a.alloc(u32, g.n * g.k / 8);
    const scales = try a.alloc(u16, g.n * g.k / 128);
    const biases = try a.alloc(u16, g.n * g.k / 128);
    gptqInto(g, words, scales, biases);
    return mlxHost(words, scales, biases, null, g.n, g.k, 4, 128, .f16);
}

/// The routed experts `prefix` experts.{e}.{gate,up,down}_proj, GPTQ, stacked as the MLX kernels read them: gate then
/// up rows an expert (E, 2 W, K/8), down (E, D, W/8). The shared expert is not in the stack (count E).
pub fn gptqExperts(a: Allocator, t: anytype, prefix: []const u8, count: usize) !quant.experts.Experts {
    const first = try gptqOf(a, t, try std.fmt.allocPrint(a, "{s}experts.0.gate_proj", .{prefix}));
    const down0 = try gptqOf(a, t, try std.fmt.allocPrint(a, "{s}experts.0.down_proj", .{prefix}));
    const w, const d = .{ first.n, first.k };
    if (down0.k != w or down0.n != d) return refuse(prefix, "expert down does not fit gate and up");
    const fw = try a.alloc(u32, count * 2 * w * d / 8);
    const fs = try a.alloc(u16, count * 2 * w * d / 128);
    const fb = try a.alloc(u16, fs.len);
    const dw = try a.alloc(u32, count * d * w / 8);
    const ds = try a.alloc(u16, count * d * w / 128);
    const db = try a.alloc(u16, ds.len);
    for (0..count) |e| {
        inline for (.{ "gate_proj", "up_proj" }, 0..) |name, h| {
            const g = try gptqOf(a, t, try std.fmt.allocPrint(a, "{s}experts.{d}.{s}", .{ prefix, e, name }));
            if (g.n != w or g.k != d) return refuse(prefix, "experts differ in shape");
            const row = (2 * e + h) * w;
            gptqInto(g, fw[row * d / 8 ..][0 .. w * d / 8], fs[row * d / 128 ..][0 .. w * d / 128], fb[row * d / 128 ..][0 .. w * d / 128]);
        }
        const g = try gptqOf(a, t, try std.fmt.allocPrint(a, "{s}experts.{d}.down_proj", .{ prefix, e }));
        if (g.n != d or g.k != w) return refuse(prefix, "experts differ in shape");
        gptqInto(g, dw[e * d * w / 8 ..][0 .. d * w / 8], ds[e * d * w / 128 ..][0 .. d * w / 128], db[e * d * w / 128 ..][0 .. d * w / 128]);
    }
    return .{
        .fused = mlxHost(fw, fs, fb, count, 2 * w, d, 4, 128, .f16),
        .gated = true,
        .down = mlxHost(dw, ds, db, count, d, w, 4, 128, .f16),
        .width = w,
        .dims = d,
        .count = count,
    };
}

/// `key`'s weight as fp32 rows [N, K]: a float tensor widened, or block-FP8 codes times their 128 x 128 scale_inv.
pub fn floats(a: Allocator, t: anytype, key: []const u8) !struct { v: []f32, n: usize, k: usize } {
    const w = try t.get(try named(a, key, ".weight"));
    if (w.rank != 2) return refuse(key, "weight is not a matrix");
    const n, const k = .{ w.shape[0], w.shape[1] };
    const v = try a.alloc(f32, n * k);
    if (w.dtype == .f8_e4m3) {
        const s = try t.get(try named(a, key, ".weight_scale_inv"));
        const kb = (k + 127) / 128;
        if (s.rank != 2 or s.shape[0] != (n + 127) / 128 or s.shape[1] != kb or !quant.convert.tableDtype(s.dtype)) return refuse(key, "scale_inv does not fit 128 x 128 blocks");
        var lut: [256]f32 = undefined;
        for (&lut, 0..) |*x, i| x.* = @floatCast(quant.fp8.e4m3(@intCast(i)));
        for (0..n) |r| for (0..k) |c| {
            v[r * k + c] = lut[w.bytes[r * k + c]] * quant.convert.load(s.dtype, s.bytes, (r / 128) * kb + c / 128);
        };
    } else {
        if (!quant.convert.isFloat(w.dtype) or w.dtype == .f64) return refuse(key, "weight is neither float nor block FP8");
        for (v, 0..) |*x, i| x.* = quant.convert.load(w.dtype, w.bytes, i);
    }
    return .{ .v = v, .n = n, .k = k };
}

/// Rows [n, k] as MLX affine words of `bits` (4 or 8) in groups of `group`, bf16 tables: a group's min as the bias and
/// (max - min) / (2^bits - 1) as the scale, both rounded to bf16, the codes rounded against the rounded pair.
pub fn quantize(a: Allocator, v: []const f32, n: usize, k: usize, bits: u8, group: u16) !quant.Host {
    if ((bits != 4 and bits != 8) or k % group != 0 or k * bits % 32 != 0) return error.UnsupportedQuantization;
    const groups = k / group;
    const words = try a.alloc(u32, n * k * bits / 32);
    @memset(words, 0);
    const scales = try a.alloc(u16, n * groups);
    const biases = try a.alloc(u16, n * groups);
    const top: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    for (0..n) |r| for (0..groups) |g| {
        const x = v[r * k + g * group ..][0..group];
        var lo = x[0];
        var hi = x[0];
        for (x) |e| {
            lo = @min(lo, e);
            hi = @max(hi, e);
        }
        const bias = quant.convert.bf16(lo);
        var scale = quant.convert.bf16((hi - lo) / top);
        if (scale & 0x7fff == 0) scale = 0x3f80;
        scales[r * groups + g] = scale;
        biases[r * groups + g] = bias;
        const s = quant.convert.load(.bf16, std.mem.asBytes(&scale), 0);
        const b = quant.convert.load(.bf16, std.mem.asBytes(&bias), 0);
        for (x, 0..) |e, i| {
            const q: u32 = @intFromFloat(std.math.clamp(@round((e - b) / s), 0, top));
            const at = (r * k + g * group + i) * bits;
            words[at / 32] |= q << @intCast(at % 32);
        }
    };
    return mlxHost(words, scales, biases, null, n, k, bits, group, .bf16);
}

/// The draft-only and float projections' width: 8-bit codes in groups of 64.
pub const requant_bits = 8;
pub const requant_group = 64;

/// `key` as the loader reads it: GPTQ as MLX 4-bit (exact); the trunk's block FP8 and bf16 as stored for the coded
/// decoders (exact); the MTP layer's float matrices requantized to 8-bit affine (they only draft).
pub fn read(a: Allocator, t: anytype, key: []const u8) !quant.Host {
    if (t.has(try named(a, key, ".qweight"))) return gptq(a, t, key);
    if (!std.mem.startsWith(u8, key, "language_model.mtp.")) return coded(a, t, key);
    const f = try floats(a, t, key);
    return quantize(a, f.v, f.n, f.k, requant_bits, requant_group);
}

/// A block-FP8 or bf16 `key` for kernels/hip/quant/coded.hpp: the weight's bytes as words (four e4m3 or two bf16 a
/// word), and for FP8 each row's fp32 scale a 128 k split into its low (scales) and high (biases) halves.
pub fn coded(a: Allocator, t: anytype, key: []const u8) !quant.Host {
    const w = try t.get(try named(a, key, ".weight"));
    if (w.rank != 2) return refuse(key, "weight is not a matrix");
    const n, const k = .{ w.shape[0], w.shape[1] };
    if (k % 32 != 0) return refuse(key, "K is not a multiple of 32");
    if (w.dtype == .f8_e4m3) {
        const s = try t.get(try named(a, key, ".weight_scale_inv"));
        const kb = (k + 127) / 128;
        if (k % 128 != 0 or s.rank != 2 or s.shape[0] != (n + 127) / 128 or s.shape[1] != kb or s.dtype != .f32) return refuse(key, "scale_inv is not fp32 over 128 x 128 blocks");
        const lo = try a.alloc(u16, n * kb);
        const hi = try a.alloc(u16, n * kb);
        for (0..n) |r| for (0..kb) |g| {
            const bits: u32 = @bitCast(quant.convert.load(.f32, s.bytes, (r / 128) * kb + g));
            lo[r * kb + g] = @truncate(bits);
            hi[r * kb + g] = @truncate(bits >> 16);
        };
        const words: Tensor = .{ .dtype = .i32, .rank = 2, .shape = .{ n, k / 4, 1, 1, 1 }, .bytes = w.bytes };
        const table: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ n, kb, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(lo) };
        var high = table;
        high.bytes = std.mem.sliceAsBytes(hi);
        return .{ .mlx = .{ .words = words, .scales = table, .biases = high, .bits = 8, .group = 128, .code = .fp8 } };
    }
    if (w.dtype != .bf16) return refuse(key, "weight is neither block FP8 nor bf16");
    // no terms: a one-entry table stands in for both
    const none: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &[_]u8{ 0, 0 } };
    const words: Tensor = .{ .dtype = .i32, .rank = 2, .shape = .{ n, k / 2, 1, 1, 1 }, .bytes = w.bytes };
    return .{ .mlx = .{ .words = words, .scales = none, .biases = none, .bits = 16, .group = 128, .code = .bf16 } };
}

/// Float experts `prefix` experts.{e} (the MTP layer's) with the shared expert last, requantized and stacked as
/// core/quant/experts.zig stacks the MLX checkpoint's: gate then up rows an expert, count E + 1.
pub fn floatExperts(a: Allocator, t: anytype, prefix: []const u8, count: usize) !quant.experts.Experts {
    var fused: std.ArrayList(u8) = .empty;
    var fused_s: std.ArrayList(u8) = .empty;
    var fused_b: std.ArrayList(u8) = .empty;
    var down: std.ArrayList(u8) = .empty;
    var down_s: std.ArrayList(u8) = .empty;
    var down_b: std.ArrayList(u8) = .empty;
    var w: usize = 0;
    var d: usize = 0;
    for (0..count + 1) |e| {
        const stem = if (e < count) try std.fmt.allocPrint(a, "{s}experts.{d}.", .{ prefix, e }) else try std.fmt.allocPrint(a, "{s}shared_expert.", .{prefix});
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, h| {
            const f = try floats(a, t, try named(a, stem, name));
            if (h == 0) w, d = .{ f.n, f.k };
            if ((h < 2 and (f.n != w or f.k != d)) or (h == 2 and (f.n != d or f.k != w))) return refuse(stem, "experts differ in shape");
            const q = (try quantize(a, f.v, f.n, f.k, requant_bits, requant_group)).mlx;
            const lists = if (h < 2) .{ &fused, &fused_s, &fused_b } else .{ &down, &down_s, &down_b };
            try lists[0].appendSlice(a, q.words.bytes);
            try lists[1].appendSlice(a, q.scales.bytes);
            try lists[2].appendSlice(a, q.biases.bytes);
        }
    }
    const n = count + 1;
    const as32 = struct {
        fn of(b: []u8) []u32 {
            return @alignCast(std.mem.bytesAsSlice(u32, b));
        }
        fn of16(b: []u8) []u16 {
            return @alignCast(std.mem.bytesAsSlice(u16, b));
        }
    };
    return .{
        .fused = mlxHost(as32.of(fused.items), as32.of16(fused_s.items), as32.of16(fused_b.items), n, 2 * w, d, requant_bits, requant_group, .bf16),
        .gated = true,
        .down = mlxHost(as32.of(down.items), as32.of16(down_s.items), as32.of16(down_b.items), n, d, w, requant_bits, requant_group, .bf16),
        .width = w,
        .dims = d,
        .count = n,
    };
}

/// Whether `prefix`'s routed experts are GPTQ (the trunk's) rather than float (the MTP layer's).
pub fn gptqRouted(a: Allocator, t: anytype, prefix: []const u8) !bool {
    return t.has(try std.fmt.allocPrint(a, "{s}experts.0.gate_proj.qweight", .{prefix}));
}

const Fake = struct {
    map: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn has(f: *const Fake, key: []const u8) bool {
        return f.map.contains(key);
    }

    pub fn get(f: *const Fake, key: []const u8) !Tensor {
        return f.map.get(key) orelse error.MissingTensor;
    }
};

test "GPTQ int4 as MLX words: every weight s (q - 8), in float64, through the MLX reference" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const k = 256;
    const n = 16;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const qw = try a.alloc(u32, k / 8 * n);
    for (qw) |*x| x.* = rnd.int(u32);
    const sc = try a.alloc(u16, k / 128 * n);
    for (sc) |*x| x.* = @bitCast(@as(f16, @floatCast(rnd.float(f32) * 0.01 + 1e-4)));
    const qz = try a.alloc(u32, k / 128 * n / 8);
    @memset(qz, 0x77777777);
    var f: Fake = .{};
    try f.map.put(a, "p.qweight", .{ .dtype = .i32, .rank = 2, .shape = .{ k / 8, n, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(qw) });
    try f.map.put(a, "p.scales", .{ .dtype = .f16, .rank = 2, .shape = .{ k / 128, n, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(sc) });
    try f.map.put(a, "p.qzeros", .{ .dtype = .i32, .rank = 2, .shape = .{ k / 128, n / 8, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(qz) });
    const h = (try gptq(a, &f, "p")).mlx;
    const out = try a.alloc(f64, n * k);
    try quant.mlx.reference(h, out);
    for (0..n) |c| for (0..k) |i| {
        const q: f64 = @floatFromInt((qw[(i / 8) * n + c] >> @intCast(4 * (i % 8))) & 15);
        const s: f64 = @floatCast(@as(f16, @bitCast(sc[(i / 128) * n + c])));
        try std.testing.expectEqual(s * (q - 8), out[c * k + i]);
    };
    qz[0] = 0x77777778;
    try std.testing.expectError(error.UnexpectedTensor, gptq(a, &f, "p"));
}

test "requantized rows stay within half a step of their group's range" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const n = 3;
    const k = 128;
    const v = try a.alloc(f32, n * k);
    for (v, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.05;
    const h = (try quantize(a, v, n, k, 8, 64)).mlx;
    const out = try a.alloc(f64, n * k);
    try quant.mlx.reference(h, out);
    for (v, out) |x, y| try std.testing.expect(@abs(x - y) < 0.1 / 255.0 * 1.1);
}
