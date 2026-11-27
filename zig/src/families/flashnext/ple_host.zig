//! The n-gram table's host side: the row hash, shard and part names, the table scale and the config's spec.

const std = @import("std");
const core = @import("core");

const golden: u64 = 0x9E3779B97F4A7C15;
const mix1: u64 = 0xBF58476D1CE4E5B9;
const mix2: u64 = 0x94D049BB133111EB;
const prime_step: u64 = 10007;
pub const max_n: usize = 8;
pub const max_heads: usize = 64;
pub const max_shards: usize = 128;
pub const layer: usize = 1;

const Tensor = core.safetensors.Tensor;

pub const Hash = struct {
    n: usize,
    per: usize,
    heads: usize,
    eos: i64,
    dh: usize,
    shards: usize,
    sizes: [max_heads]i64 = undefined,
    offsets: [max_heads]i64 = undefined,
    mult: [max_n]i64 = undefined,
    starts: [max_shards + 1]i64 = undefined,

    pub fn init(args: struct {
        vocab: u64,
        ngram: usize,
        per: usize,
        base: u64,
        divisor: u64,
        shards: usize,
        seed: u64,
        eos: i64,
        ple_dim: usize,
        ple_index: usize,
    }) !Hash {
        if (args.ngram < 2 or args.ngram > max_n or args.per == 0 or args.shards == 0 or args.shards > max_shards) return error.UnexpectedTensor;
        if (args.base < 2 or args.divisor == 0) return error.UnexpectedTensor;
        const heads = (args.ngram - 1) * args.per;
        if (heads > max_heads or args.ple_dim == 0 or args.ple_dim % heads != 0) return error.UnexpectedTensor;
        var h = Hash{ .n = args.ngram, .per = args.per, .heads = heads, .eos = args.eos, .dh = args.ple_dim / heads, .shards = args.shards };
        var total: i64 = 0;
        for (0..heads) |head| {
            const size: i64 = @intCast(nthPrimeAfter(args.base - 1, args.ple_index * heads + head + 1));
            h.sizes[head] = size;
            h.offsets[head] = total;
            total += size;
        }
        const divisor: i64 = @intCast(args.divisor);
        const shard_n: i64 = @intCast(args.shards);
        const padded = divCeil(total, divisor) * divisor;
        const base: i64 = @divFloor(padded, shard_n);
        const extra: usize = @intCast(@mod(padded, shard_n));
        var acc: i64 = 0;
        for (0..args.shards) |i| {
            h.starts[i] = acc;
            acc += base + @as(i64, @intFromBool(i < extra));
        }
        h.starts[args.shards] = acc;
        const half: u64 = @max(1, (((1 << 63) - 1) / @max(args.vocab, 1)) / 2);
        const seed = args.seed +% prime_step *% @as(u64, @intCast(args.ple_index));
        for (0..args.ngram) |i| {
            const mixed = splitmix64(seed +% golden *% @as(u64, @intCast(i + 1)));
            h.mult[i] = @intCast(2 * (mixed % half) + 1);
        }
        return h;
    }

    pub fn locate(h: Hash, row: i64) !struct { shard: usize, local: usize } {
        if (row < 0 or row >= h.starts[h.shards]) return error.UnexpectedTensor;
        var lo: usize = 0;
        var hi = h.shards;
        while (lo + 1 < hi) {
            const mid = (lo + hi) / 2;
            if (h.starts[mid] <= row) lo = mid else hi = mid;
        }
        return .{ .shard = lo, .local = @intCast(row - h.starts[lo]) };
    }
};

fn splitmix64(value: u64) u64 {
    var v = value +% golden;
    v = (v ^ (v >> 30)) *% mix1;
    v = (v ^ (v >> 27)) *% mix2;
    return v ^ (v >> 31);
}

fn isPrime(value: u64) bool {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    var d: u64 = 3;
    while (d <= value / d) : (d += 2) if (value % d == 0) return false;
    return true;
}

fn nthPrimeAfter(start: u64, count: usize) u64 {
    var prime = start;
    for (0..count) |_| {
        prime += 1;
        while (!isPrime(prime)) prime += 1;
    }
    return prime;
}

fn divCeil(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// Row ids of one new token after `history` (n - 1 ids). A fresh decode window is n - 1 copies of eos.
pub fn lastIds(h: Hash, history: []const i64, token: i64, out: []i64) !void {
    if (history.len != h.n - 1 or out.len != h.heads) return error.UnexpectedTensor;
    var seq: [max_n]i64 = undefined;
    for (history, 0..) |v, i| seq[i] = v;
    seq[history.len] = token;
    const width = h.n;
    var eos_at: [max_n]i64 = undefined;
    for (0..width) |i| eos_at[i] = if (seq[i] == h.eos) @intCast(i) else -1;
    var acc: [max_n]i64 = undefined;
    acc[0] = eos_at[0];
    for (1..width) |i| acc[i] = @max(acc[i - 1], eos_at[i]);
    var before: [max_n]i64 = undefined;
    before[0] = -1;
    for (1..width) |i| before[i] = acc[i - 1];
    var shifted: [max_n]i64 = undefined;
    const pos: i64 = @intCast(width - 1);
    const in_segment = pos - (before[width - 1] + 1);
    for (0..h.n) |shift| {
        const source = pos - @as(i64, @intCast(shift));
        const taken = seq[@intCast(@max(source, 0))];
        const keep = in_segment >= @as(i64, @intCast(shift)) and source >= 0;
        shifted[shift] = if (keep) taken else h.eos;
    }
    var at: usize = 0;
    var ngram: usize = 2;
    while (ngram <= h.n) : (ngram += 1) {
        var mixed = shifted[0] *% h.mult[0];
        for (1..ngram) |p| mixed = mixed ^ (shifted[p] *% h.mult[p]);
        for (0..h.per) |_| {
            out[at] = @mod(mixed, h.sizes[at]) + h.offsets[at];
            at += 1;
        }
    }
}

pub const toBf16 = core.bf16_round.bits;

pub fn sumGroups(row: []const u16, out: []f32) void {
    const groups = row.len / 32;
    for (0..groups) |g| {
        var buf: [32]f32 = undefined;
        for (0..32) |i| buf[i] = @bitCast(@as(u32, row[g * 32 + i]) << 16);
        var n: usize = 32;
        while (n > 1) {
            n /= 2;
            for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        out[g] = buf[0];
    }
}

/// One mapped shard (core.safetensors.File) with a typed lookup.
pub const Open = struct {
    file: core.safetensors.File,
    io: std.Io,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Open {
        return .{ .file = try core.safetensors.File.open(gpa, io, path), .io = io };
    }

    pub fn close(self: *Open, gpa: std.mem.Allocator) void {
        _ = gpa;
        self.file.close(self.io);
    }

    pub fn tensor(self: *const Open, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const t = self.file.get(name) orelse return error.MissingTensor;
        if (t.dtype != dtype) return error.UnexpectedTensor;
        return t;
    }
};

pub fn readI64(t: Tensor, out: []i64) !void {
    if (t.dtype != .i64 or t.rank != 1 or t.dim(0) != out.len or t.bytes.len != out.len * 8) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = std.mem.readInt(i64, t.bytes[8 * i ..][0..8], .little);
}

pub fn weightName(buf: []u8, shard: usize, nested: bool) ![]const u8 {
    return if (nested)
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.weight", .{ layer, shard })
    else
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ layer, shard });
}

pub fn partName(buf: []u8, shard: usize, nested: bool, part: []const u8) ![]const u8 {
    return partNameFor(buf, shard, nested, part, false);
}

pub fn partNameFor(buf: []u8, shard: usize, nested: bool, part: []const u8, modelopt: bool) ![]const u8 {
    const prefix = if (modelopt) "model.language_model" else "language_model.model";
    return if (nested)
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.{s}", .{ prefix, layer, shard, part })
    else
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ prefix, layer, shard, part });
}

pub fn findOpen(opened: []Open, path: []const u8) ?*Open {
    for (opened) |*o| if (std.mem.eql(u8, o.file.path, path)) return o;
    return null;
}

pub fn copyRow(file: *const Open, shard: usize, nested: bool, local: usize, dh: usize, words: []u8, scales: []u8, biases: []u8) !void {
    var name: [180]u8 = undefined;
    const w = try file.tensor(try partName(&name, shard, nested, "weight"), .u32);
    const s = try file.tensor(try partName(&name, shard, nested, "scales"), .bf16);
    const b = try file.tensor(try partName(&name, shard, nested, "biases"), .bf16);
    const words_w = dh / 8;
    const groups = dh / 32;
    if (w.dim(1) != words_w or s.dim(1) != groups or b.dim(1) != groups or local >= w.dim(0)) return error.UnexpectedTensor;
    const wb = words_w * 4;
    const gb = groups * 2;
    @memcpy(words, w.bytes[local * wb ..][0..wb]);
    @memcpy(scales, s.bytes[local * gb ..][0..gb]);
    @memcpy(biases, b.bytes[local * gb ..][0..gb]);
}

pub fn tableScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap) !f32 {
    return readScale(gpa, io, dir, wm, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale", false);
}

pub fn readScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap, name: []const u8, required: bool) !f32 {
    const file = wm.get(name) orelse return if (required) error.MissingTensor else 1;
    const path = try std.fs.path.join(gpa, &.{ dir, file.string });
    defer gpa.free(path);
    var opened = try Open.load(gpa, io, path);
    defer opened.close(gpa);
    const t = opened.file.get(name) orelse return error.MissingTensor;
    if (t.numel() != 1) return error.UnexpectedTensor;
    return switch (t.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little)),
        .f64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, t.bytes[0..8], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[0..2], .little)) << 16),
        else => error.UnexpectedTensor,
    };
}

fn jsonU64(v: std.json.Value) !u64 {
    if (v != .integer or v.integer < 0) return error.UnexpectedTensor;
    return @intCast(v.integer);
}

fn optU64(o: std.json.ObjectMap, name: []const u8, default: u64) !u64 {
    const v = o.get(name) orelse return default;
    if (v == .null) return default;
    return jsonU64(v);
}

const NgramSpec = struct {
    vocab: u64,
    ngram: usize,
    per: usize,
    base: u64,
    divisor: u64,
    shards: usize,
    seed: u64,
    eos: i64,
    ple_dim: usize,
    ple_index: usize,
};

/// `ple_layer_ids` value 2 is decoder layer 1. The eos id is the text config's, which the hash mixes in.
pub fn readSpec(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !NgramSpec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const text = if (root.get("text_config")) |v| if (v == .object and v.object.count() > 0) v.object else root else root;
    const layers = try jsonU64(text.get("num_hidden_layers") orelse return error.MissingTensor);
    if (layers <= layer) return error.UnexpectedTensor;
    const hidden = try jsonU64(text.get("hidden_size") orelse return error.MissingTensor);
    const ids = text.get("ple_layer_ids") orelse return error.MissingTensor;
    if (ids != .array) return error.UnexpectedTensor;
    var seen: [512]bool = @splat(false);
    var ple_index: usize = 0;
    var found = false;
    for (ids.array.items) |item| {
        const id = try jsonU64(item);
        if (id == 0 or id > layers or id > seen.len) return error.UnexpectedTensor;
        if (id == layer + 1) found = true;
        if (!seen[id - 1] and id - 1 < layer) ple_index += 1;
        seen[id - 1] = true;
    }
    if (!found) return error.UnexpectedTensor;
    const eos: i64 = if (text.get("eos_token_id")) |v| blk: {
        if (v == .null) break :blk 0;
        if (v == .array) {
            if (v.array.items.len == 0) return error.UnexpectedTensor;
            break :blk @intCast(try jsonU64(v.array.items[0]));
        }
        break :blk @intCast(try jsonU64(v));
    } else 0;
    return .{
        .vocab = try jsonU64(text.get("vocab_size") orelse return error.MissingTensor),
        .ngram = @intCast(try optU64(text, "ngram_size", 3)),
        .per = @intCast(try optU64(text, "heads_per_ngram", 8)),
        .base = try optU64(text, "ngram_vocab_size_base", 20_000_000),
        .divisor = try optU64(text, "make_ngram_vocab_size_divisible_by", 128),
        .shards = @intCast(try optU64(text, "split_ngram_parts", 128)),
        .seed = try optU64(text, "seed", 1234),
        .eos = eos,
        .ple_dim = @intCast(try optU64(text, "ple_embed_dim", hidden)),
        .ple_index = ple_index,
    };
}

/// The text config's eos id, which a fresh n-gram window repeats.
pub fn eosOf(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !i64 {
    const cfg = try readSpec(gpa, io, model_dir);
    return cfg.eos;
}

/// A table's row format: MLX affine words, NVFP4 codes with E4M3 group scales, or FP8 rows with one table scale.
pub const Format = enum { mlx, nvfp4, fp8 };

test "a fresh window hashes token 7 into sixteen rows" {
    const h = try Hash.init(.{
        .vocab = 248320,
        .ngram = 3,
        .per = 8,
        .base = 20_000_000,
        .divisor = 128,
        .shards = 128,
        .seed = 1234,
        .eos = 248044,
        .ple_dim = 2560,
        .ple_index = 0,
    });
    try std.testing.expectEqual(@as(i64, 23703573157769), h.mult[0]);
    const history = [_]i64{ 248044, 248044 };
    var ids: [16]i64 = undefined;
    try lastIds(h, &history, 7, &ids);
    const want = [_]i64{ 2927653, 34980843, 54748278, 66612378, 97814964, 109013870, 126560393, 151352333, 167888935, 182235580, 215170017, 237467519, 247510681, 278779700, 296141806, 304994522 };
    try std.testing.expectEqualSlices(i64, &want, &ids);
    const loc = try h.locate(ids[0]);
    try std.testing.expectEqual(@as(usize, 1), loc.shard);
    try std.testing.expectEqual(@as(usize, 427641), loc.local);
}
