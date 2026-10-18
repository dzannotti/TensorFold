//! The DFlash2 (or DSpark) drafter's config.json, as the HIP drafter's kernels take it.
const std = @import("std");

/// The target's widths a drafter must match.
pub const Target = struct { hidden: usize, vocab: usize, layers: usize };

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    window: usize,
    group: usize,
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [8]u32,
    n_taps: usize,
    /// DSpark (openbmb's Qwen3DSparkModel; vLLM qwen3_dspark.py, llama.cpp dflash.cpp): DFlash without the dynamic convs,
    /// a vanilla Markov head (markov_w1[prev] . markov_w2[token], rank `rank`) in the selector's place, and every block
    /// row a draft (sample_from_anchor: row i predicts position pos0 + i + 1, its Markov predecessor the token at pos0 + i).
    dspark: bool = false,
    /// DSpark's Markov bias over the whole vocabulary, as vLLM's _sample_sequential and llama.cpp's
    /// build_dspark_markov_head do when the config has no dspark_draft_topk (vLLM dspark/speculator.py 78-80, 157-159);
    /// false: on each row's `topk` head logits only (vLLM's _sample_sequential_topk; TF_DRAFT_TOPK forces it).
    full_vocab: bool = false,

    /// Drafts one hold yields: every row with DSpark, the rows after the pending one with DFlash2.
    pub fn drafts(c: Config) usize {
        return if (c.dspark) c.block else c.block - 1;
    }
    /// The block row drafting depth `i` (0 the first draft).
    pub fn draftRow(c: Config, i: usize) usize {
        return if (c.dspark) i else i + 1;
    }

    pub fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    /// The dynamic convolutions' coefficients a row: 2 branches, 2 lags, a group's each.
    pub fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// A ring's rows: the window and a tile more, a whole number of 16-row tiles.
    pub fn ringRows(c: Config) usize {
        return std.mem.alignForward(usize, c.window + 16, 16);
    }
    /// A stream's rings: every layer's keys and values.
    pub fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: Target) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, target);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: Target) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = o.get("dflash_config") orelse return error.BadDraftConfig;
        if (d != .object) return error.BadDraftConfig;
        const rope = o.get("rope_parameters") orelse return error.BadDraftConfig;
        if (rope != .object) return error.BadDraftConfig;
        if (isDspark(o, d.object)) return parseDspark(o, d.object, rope.object, target);
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(d.object, "block_size"),
            .mask = @intCast(try whole(d.object, "mask_token_id")),
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = try int(d.object, "selector_rank"),
            .topk = try int(d.object, "selector_top_k"),
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num(rope.object, "rope_theta")),
            .taps = undefined,
            .n_taps = 0,
        };
        // 16-position blocks, as the Python engine drafted: deeper drafts keep landing on code (TF_DRAFT_BLOCK overrides)
        c.block = @max(c.block, @min(16, 64 / @max(c.heads / @max(c.kv_heads, 1), 1)));
        if (std.c.getenv("TF_DRAFT_BLOCK")) |v| c.block = std.fmt.parseInt(usize, std.mem.span(v), 10) catch c.block;
        try readTaps(&c, d.object, target);
        // what the kernels fix: 128-wide heads, two-lag convolutions, a block's queries of a key head in one 64-query tile
        if (c.head_dim != 128 or try int(d.object, "conv_kernel_size") != 2) return error.UnsupportedDraftConfig;
        if (c.hidden % c.group != 0) return error.UnsupportedDraftConfig;
        return c.check(target);
    }

    fn check(c: Config, target: Target) !Config {
        if (c.heads % c.kv_heads != 0 or c.block < 2 or c.block > 16 or c.block * (c.heads / c.kv_heads) > 64) return error.UnsupportedDraftConfig;
        if (c.topk > 64 or c.mask >= c.vocab or c.hidden != target.hidden or c.vocab != target.vocab or c.hidden % 64 != 0) return error.UnsupportedDraftConfig;
        if (c.intermediate % 64 != 0 or c.rank % 64 != 0 or c.layers > 16 or c.window < c.block) return error.UnsupportedDraftConfig;
        return c;
    }

    fn isDspark(o: std.json.ObjectMap, d: std.json.ObjectMap) bool {
        if (d.get("projector_type")) |v| if (v == .string and std.mem.eql(u8, v.string, "dspark")) return true;
        return o.get("markov_rank") != null or d.get("markov_rank") != null;
    }

    /// DSpark's config (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark): full attention, no convs, a vanilla Markov head.
    fn parseDspark(o: std.json.ObjectMap, d: std.json.ObjectMap, rope: std.json.ObjectMap, target: Target) !Config {
        if (d.get("markov_head_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "vanilla")) return error.UnsupportedDraftConfig;
        if (o.get("sample_from_anchor")) |v| if (v != .bool or !v.bool) return error.UnsupportedDraftConfig;
        if (o.get("sliding_window")) |v| if (v != .null) return error.UnsupportedDraftConfig;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(o, "block_size"),
            .mask = @intCast(try whole(d, "mask_token_id")),
            // full attention in the references; the rings keep the last `window` positions (the target's tap ring
            // keeps 2048 rows past a round, so a prompt's earlier rows never reach the drafter)
            .window = 2048,
            .group = 1,
            .rank = if (o.get("markov_rank") != null) try int(o, "markov_rank") else try int(d, "markov_rank"),
            // vLLM's dspark_draft_topk path when the config sets it (the Markov bias on each row's top-k head logits only)
            .topk = 16,
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num(rope, "rope_theta")),
            .taps = undefined,
            .n_taps = 0,
            .dspark = true,
            .full_vocab = true,
        };
        if (o.get("dspark_draft_topk")) |v| if (v != .null) {
            c.topk = try int(o, "dspark_draft_topk");
            c.full_vocab = false;
        };
        if (rope.get("rope_dim")) |v| if (v != .integer or v.integer != c.head_dim) return error.UnsupportedDraftConfig;
        // the drafts a hold yields: the block (vLLM and llama.cpp draft num_speculative_tokens / n_max rows)
        if (std.c.getenv("TF_DRAFT_BLOCK")) |v| c.block = std.fmt.parseInt(usize, std.mem.span(v), 10) catch c.block;
        if (std.c.getenv("TF_DRAFT_WINDOW")) |v| c.window = std.fmt.parseInt(usize, std.mem.span(v), 10) catch c.window;
        if (std.c.getenv("TF_DRAFT_TOPK")) |v| {
            c.topk = std.fmt.parseInt(usize, std.mem.span(v), 10) catch c.topk;
            c.full_vocab = false;
        }
        // the whole-vocabulary kernels take rank 256 (draft.hip markov_step); another rank keeps the top-k walk
        if (c.rank != 256) c.full_vocab = false;
        try readTaps(&c, d, target);
        if (c.head_dim != 128 and c.head_dim != 256) return error.UnsupportedDraftConfig;
        return c.check(target);
    }
};

fn readTaps(c: *Config, d: std.json.ObjectMap, target: Target) !void {
    const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > c.taps.len) return error.BadDraftConfig;
    for (taps.array.items, 0..) |t, i| {
        if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
        c.taps[i] = @intCast(t.integer);
    }
    c.n_taps = taps.array.items.len;
}

/// A size, count or divisor: positive, so a zero can't reach a remainder or an empty allocation.
fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

/// A token id: any whole number.
fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

test "z-lab's DFlash2 config parses; a zero divisor is refused; a wider block than one tile takes is unsupported" {
    const gpa = std.testing.allocator;
    const target: Target = .{ .hidden = 5120, .vocab = 248320, .layers = 64 };
    const text =
        \\{"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,"selector_rank":256,
        \\ "selector_top_k":16,"target_layer_ids":[5,19,33,47,61]},"head_dim":128,"hidden_size":5120,"intermediate_size":17408,
        \\ "num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\ "rope_parameters":{"rope_theta":10000000},"sliding_window":2048,"vocab_size":248320}
    ;
    const c = try Config.parse(gpa, text, target);
    try std.testing.expectEqual(@as(usize, 256), c.rank);
    try std.testing.expectEqual(@as(usize, 5), c.n_taps);
    try std.testing.expectEqual(@as(usize, 2064), c.ringRows());
    try std.testing.expectEqual(@as(f32, 1e7), c.theta);
    for ([_][2][]const u8{ .{ "\"conv_group_size\":16", "\"conv_group_size\":0" }, .{ "\"num_key_value_heads\":8", "\"num_key_value_heads\":0" } }) |r| {
        const bad = try std.mem.replaceOwned(u8, gpa, text, r[0], r[1]);
        defer gpa.free(bad);
        try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, bad, target));
    }
    const wide = try std.mem.replaceOwned(u8, gpa, text, "\"block_size\":8", "\"block_size\":32");
    defer gpa.free(wide);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(gpa, wide, target));
    try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, text, .{ .hidden = 5120, .vocab = 248320, .layers = 61 }));
}

test "openbmb's DSpark config parses: 256-wide heads, every row a draft, the Markov rank" {
    const gpa = std.testing.allocator;
    const target: Target = .{ .hidden = 2048, .vocab = 248144, .layers = 40 };
    const text =
        \\{"block_size":15,"dflash_config":{"attention_mode":"gqa","markov_head_type":"vanilla","markov_rank":256,
        \\ "mask_token_id":248076,"projector_type":"dspark","target_layer_ids":[3,11,19,27,35]},"head_dim":256,"hidden_size":2048,
        \\ "intermediate_size":6144,"num_attention_heads":8,"num_hidden_layers":5,"num_key_value_heads":2,"rms_norm_eps":1e-06,
        \\ "rope_parameters":{"rope_theta":10000000,"rope_type":"default","rope_dim":256},"sliding_window":null,"vocab_size":248144,
        \\ "markov_rank":256}
    ;
    const c = try Config.parse(gpa, text, target);
    try std.testing.expect(c.dspark);
    try std.testing.expect(c.full_vocab); // no dspark_draft_topk: the bias over the whole vocabulary
    try std.testing.expectEqual(@as(usize, 256), c.head_dim);
    try std.testing.expectEqual(@as(usize, 256), c.rank);
    try std.testing.expectEqual(@as(usize, 15), c.drafts());
    try std.testing.expectEqual(@as(usize, 0), c.draftRow(0));
    try std.testing.expectEqual(@as(u32, 35), c.taps[4]);
    const gated = try std.mem.replaceOwned(u8, gpa, text, "\"vanilla\"", "\"gated\"");
    defer gpa.free(gated);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(gpa, gated, target));
}
