//! The n-gram tables in host memory (the checkpoint's mapped shards): a window's rows gathered for the device.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");

/// One shard's packed rows and their group tables, as mapped.
const Shard = struct { words: []const u8, scales: []const u8, biases: []const u8 };

pub const Tables = struct {
    shards: []Shard,
    /// The first row of each shard, and one past the last.
    starts: []usize,
    dims: usize,
    bits: u8,
    group: usize,
    hasher: ngram.Hasher,
    /// An FP8 table (INT4-AutoRound's): each code's bf16 value, e4m3 times the table's scale rounded once; a row is
    /// gathered as its bf16 values (the device reads them as the embedding), with no group tables.
    fp8: ?*const [256]u16 = null,

    /// Bytes of a row's packed words (an FP8 table's bf16 row), and of its scales (as of its biases).
    pub fn wordBytes(t: Tables) usize {
        return if (t.fp8 != null) t.dims * 2 else t.dims * t.bits / 8;
    }

    pub fn tableBytes(t: Tables) usize {
        return if (t.fp8 != null) 0 else t.dims / t.group * 2;
    }

    /// The FP8 tables of the PLE layer `layer` (shards `shard_{i}`, one bf16 weight_scale), from the mapped checkpoint.
    pub fn initFp8(a: std.mem.Allocator, ck: *core.checkpoint.Checkpoint, c: *const config.Config, layer: usize) !Tables {
        const heads = try c.ngramHeadCount();
        const dims = c.ple_dim / heads;
        const shards = try a.alloc(Shard, c.ngram_shards);
        const starts = try a.alloc(usize, c.ngram_shards + 1);
        starts[0] = 0;
        var lb: [128]u8 = undefined;
        const stem = try std.fmt.allocPrint(a, "{s}.ple.ple_embedding.ngram_embedding", .{try tensors.layer(&lb, layer)});
        for (shards, 0..) |*s, i| {
            const w = try ck.get(try std.fmt.allocPrint(a, "{s}.shard_{d}.weight", .{ stem, i }));
            if (w.dtype != .f8_e4m3 or w.rank != 2 or w.shape[1] != dims) return error.UnexpectedNgramTable;
            starts[i + 1] = starts[i] + w.shape[0];
            s.* = .{ .words = w.bytes, .scales = &.{}, .biases = &.{} };
            // a row is 160 bytes anywhere in 49 GiB: no read-around (128 KiB a fault), each page read alone
            advise(w.bytes, std.os.linux.MADV.RANDOM);
        }
        const ref = try config.pleRef(c);
        var total: usize = 0;
        for (ref.sizes) |x| total += @intCast(x);
        if (starts[c.ngram_shards] != try std.math.divCeil(usize, total, c.ngram_divisor) * c.ngram_divisor) return error.UnexpectedNgramTable;
        const sc = try ck.get(try std.fmt.allocPrint(a, "{s}.weight_scale", .{stem}));
        if (sc.numel() != 1 or !core.quant.convert.isFloat(sc.dtype) or sc.dtype == .f64) return error.UnexpectedNgramTable;
        const scale = core.quant.convert.load(sc.dtype, sc.bytes, 0);
        const lut = try a.create([256]u16);
        for (lut, 0..) |*v, code| {
            const x: f32 = @floatCast(core.quant.fp8.e4m3(@intCast(code)));
            v.* = if (std.math.isNan(x)) 0x7fc0 else core.quant.convert.bf16(x * scale);
        }
        return .{ .shards = shards, .starts = starts, .dims = dims, .bits = 8, .group = dims, .hasher = try ngram.Hasher.init(c), .fp8 = lut };
    }

    /// The tables of the PLE layer `layer` from the mapped checkpoint; the shards' rows as the reference splits them.
    pub fn init(a: std.mem.Allocator, ck: *core.checkpoint.Checkpoint, c: *const config.Config, layer: usize, spelling: []const u8) !Tables {
        const heads = try c.ngramHeadCount();
        const ref = try config.pleRef(c);
        var total: usize = 0;
        for (ref.sizes) |s| total += @intCast(s);
        const rows = try std.math.divCeil(usize, total, c.ngram_divisor) * c.ngram_divisor;
        const base = rows / c.ngram_shards;
        const extra = rows % c.ngram_shards;
        const shards = try a.alloc(Shard, c.ngram_shards);
        const starts = try a.alloc(usize, c.ngram_shards + 1);
        starts[0] = 0;
        var spec: ?@import("../spec.zig").Spec = null;
        for (shards, 0..) |*s, i| {
            var lb: [128]u8 = undefined;
            var nb: [256]u8 = undefined;
            const stem = try a.dupe(u8, try tensors.ple.table(&nb, try tensors.layer(&lb, layer), spelling, i));
            const n = base + @intFromBool(i < extra);
            starts[i + 1] = starts[i] + n;
            const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
            const sc = try ck.get(try std.fmt.allocPrint(a, "{s}.scales", .{stem}));
            const bi = try ck.get(try std.fmt.allocPrint(a, "{s}.biases", .{stem}));
            if (spec == null) spec = (try c.declared(stem, core.quant.keys.canonical)) orelse return error.UnsupportedQuantization;
            if (w.shape[0] != n or sc.dtype != .bf16 or bi.dtype != .bf16) return error.UnexpectedNgramTable;
            s.* = .{ .words = w.bytes, .scales = sc.bytes, .biases = bi.bytes };
        }
        const dims = c.ple_dim / heads;
        if (shards[0].words.len != (starts[1] - starts[0]) * dims * spec.?.bits / 8) return error.UnexpectedNgramTable;
        return .{ .shards = shards, .starts = starts, .dims = dims, .bits = spec.?.bits, .group = spec.?.group, .hasher = try ngram.Hasher.init(c) };
    }

    /// Each id's packed words, scales and biases copied into the three staging slices, in id order.
    pub fn gather(t: *const Tables, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        const wb = t.wordBytes();
        const tb = t.tableBytes();
        if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb) return error.StagingTooSmall;
        // an FP8 table's rows are mostly on disk: every page asked for first, so the reads go out together
        if (t.fp8 != null) for (ids) |id| if (t.locate(id)) |at| advise(t.shards[at.shard].words[at.row * t.dims ..][0..t.dims], std.os.linux.MADV.WILLNEED);
        for (ids, 0..) |id, i| {
            const at = t.locate(id) orelse return error.NgramRowOutOfRange;
            const row = at.row;
            const sh = t.shards[at.shard];
            if (t.fp8) |lut| {
                const codes = sh.words[row * t.dims ..][0..t.dims];
                for (codes, 0..) |code, j| std.mem.writeInt(u16, words[i * wb + 2 * j ..][0..2], lut[code], .little);
                continue;
            }
            @memcpy(words[i * wb ..][0..wb], sh.words[row * wb ..][0..wb]);
            @memcpy(scales[i * tb ..][0..tb], sh.scales[row * tb ..][0..tb]);
            @memcpy(biases[i * tb ..][0..tb], sh.biases[row * tb ..][0..tb]);
        }
    }

    /// The shard holding global row `id` and the row within it, or null past the last.
    fn locate(t: *const Tables, id: u32) ?struct { shard: usize, row: usize } {
        const s = std.sort.upperBound(usize, t.starts, @as(usize, id), struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        if (s >= t.shards.len) return null;
        return .{ .shard = s, .row = id - t.starts[s] };
    }
};

/// madvise over the pages `bytes` spans (a hint: a refusal changes nothing but speed).
fn advise(bytes: []const u8, advice: u32) void {
    const from = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), std.heap.page_size_min);
    _ = std.os.linux.madvise(@ptrFromInt(from), @intFromPtr(bytes.ptr) + bytes.len - from, advice);
}
