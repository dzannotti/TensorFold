//! FP4-Local (ModelOpt mixed precision) admitted for HIP before load: its config by field, every index tensor by name.
const std = @import("std");
const core = @import("core");
const Why = core.modelopt.Why;
const config = @import("../config.zig");
const mc = @import("../modelopt_config.zig");
const names = @import("../names_modelopt.zig");
const index = @import("../index.zig");
const tensors = @import("tensors.zig");
const Config = config.Config;

/// The layout the HIP kernels serve, PR 1's `Served` from the config; `linear` holds the layer roles it points at.
pub fn served(c: *const Config, linear: *[config.max_layers]bool) ?mc.Served {
    var ple_layer: ?usize = null;
    for (0..c.layers) |i| {
        linear[i] = c.kinds[i] == .linear_attention;
        if (c.ple[i]) ple_layer = i;
    }
    return .{
        .linear = linear[0..c.layers],
        .eps = c.eps,
        .ple_layer = ple_layer orelse return null,
        .rope_theta = c.rope_theta,
        .partial_rotary = @as(f64, @floatFromInt(c.rotary_dim)) / @as(f64, @floatFromInt(c.head_dim)),
        .index_budget = c.index_budget,
        .ple_dim = c.ple_dim,
    };
}

/// The first config field outside the FP4-Local layout the HIP path reads, or null when every one matches.
pub fn mismatch(c: *const Config) ?[]const u8 {
    if (c.format != .modelopt) return "quantization_config.quant_method";
    var linear: [config.max_layers]bool = undefined;
    const s = served(c, &linear) orelse return "ple_layer_ids";
    if (mc.quantMismatch(c.quant)) |field| return field;
    if (mc.semantics(c.text, s)) |field| return field;
    if (mc.layersMismatch(c.quant, s.linear)) |field| return field;
    if (mc.given(c.parsed.value.object, "tie_word_embeddings")) |v| if (!mc.holds(v, .{ .flag = false })) return "tie_word_embeddings";
    return shapeMismatch(c);
}

/// Every baked matrix's inputs are whole prefill folds; the on-disk n-gram rows are whole NVFP4 groups.
fn shapeMismatch(c: *const Config) ?[]const u8 {
    const heads = c.ngramHeadCount() catch return "heads_per_ngram";
    const fold = core.quant.fp4local_decode.fold;
    const blocks = [_]struct { []const u8, usize, usize }{
        .{ "hidden_size", c.hidden, fold },
        .{ "linear_value_head_dim", c.value_heads * c.value_dim, fold },
        .{ "head_dim", c.heads * c.head_dim, fold },
        .{ "shared_expert_intermediate_size", c.shared_width, fold },
        .{ "moe_intermediate_size", c.expert_width, fold },
        .{ "ple_embed_dim", c.ple_dim / heads, core.quant.nvfp4_format.group },
    };
    for (blocks) |b| if (b[1] % b[2] != 0) return b[0];
    return null;
}

/// A module's stored recipe, as quantized_layers declares it (bf16 when unlisted).
pub const Want = enum { bf16, mxfp8, nvfp4_csf, nvfp4, w4a16 };

/// What the index walk took and set aside; `empty_modules` are quantized_layers entries with no tensor in the index.
pub const Inventory = struct { taken: usize = 0, vision: usize = 0, input_scales: usize = 0, empty_modules: usize = 0 };

/// Index tensors no HIP reader takes, each with its reason.
pub const ignored = [_]struct { match: enum { prefix, suffix }, text: []const u8, reason: []const u8 }{
    .{ .match = .prefix, .text = "model.visual.", .reason = "the vision tower: Flash Next on HIP serves text" },
    .{ .match = .suffix, .text = ".input_scale", .reason = "an NVFP4 activation scale: activations stay bf16" },
};

const Walk = struct {
    a: std.mem.Allocator,
    map: std.json.ObjectMap,
    entries: std.json.ObjectMap,
    taken: std.StringHashMapUnmanaged(void) = .empty,
    declared: std.StringHashMapUnmanaged(void) = .empty,
    why: *Why,

    /// The stored name of `requested` (an MLX spelling or the stored one), marked taken.
    fn take(w: *Walk, requested: []const u8) ![]const u8 {
        var buf: [512]u8 = undefined;
        const found = names.resolve(w.map, requested, &buf) catch {
            w.why.set("the index lacks {s}", .{requested});
            return error.MissingFlashTensor;
        };
        const stored = w.map.getKey(found).?;
        try w.taken.put(w.a, stored, {});
        return stored;
    }

    fn part(w: *Walk, stem: []const u8, suffix: []const u8) !void {
        var buf: [512]u8 = undefined;
        _ = try w.take(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    /// quantized_layers' entry for the stored module `module` must be `want`'s.
    fn recipe(w: *Walk, module: []const u8, want: Want) !void {
        const entry = w.entries.get(module);
        if (entry != null) try w.declared.put(w.a, w.entries.getKey(module).?, {});
        if (holds(entry, want)) return;
        w.why.set("quantization_config.quantized_layers {s} is not {s}, the format FP4-Local's HIP reader takes", .{ module, @tagName(want) });
        return error.UnsupportedQuantization;
    }

    /// A projection's tensors under its MLX-spelled stem, its recipe checked.
    fn matrix(w: *Walk, stem: []const u8, want: Want) !void {
        var buf: [512]u8 = undefined;
        const weight = try w.take(try std.fmt.bufPrint(&buf, "{s}.weight", .{stem}));
        try w.recipe(weight[0 .. weight.len - ".weight".len], want);
        try w.planes(stem, want);
    }

    fn planes(w: *Walk, stem: []const u8, want: Want) !void {
        switch (want) {
            .bf16 => {},
            .mxfp8, .nvfp4, .w4a16 => try w.part(stem, "weight_scale"),
            .nvfp4_csf => for ([_][]const u8{ "weight_scale.nvfp4_csf_fixed", "weight_scale.nvfp4_csf_exceptions" }) |s| try w.part(stem, s),
        }
        if (want == .nvfp4 or want == .w4a16 or want == .nvfp4_csf) try w.part(stem, "weight_scale_2");
    }

    fn proj(w: *Walk, stem: []const u8, suffix: []const u8, want: Want) !void {
        var buf: [512]u8 = undefined;
        try w.matrix(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }), want);
    }

    fn hc(w: *Walk, stem: []const u8, inject: bool) !void {
        try w.part(stem, tensors.hc.norm);
        try w.proj(stem, tensors.hc.down, .bf16);
        try w.proj(stem, tensors.hc.up, .bf16);
        if (inject) try w.proj(stem, tensors.hc.inject, .bf16);
    }

    /// Every expert's gate, up and down projections, one recipe for the layer's `mlp.experts` module.
    fn experts(w: *Walk, c: *const Config, stem: []const u8, want: Want) !void {
        var buf: [512]u8 = undefined;
        for (0..c.experts) |e| for (tensors.moe.expert_parts) |p| {
            const name = try tensors.moe.expert(&buf, stem, e, p);
            const weight = try w.take(try std.fmt.bufPrint(buf[name.len..], "{s}.weight", .{name}));
            if (e == 0 and std.mem.eql(u8, p, tensors.moe.expert_parts[0])) {
                try w.recipe(weight[0 .. std.mem.lastIndexOf(u8, weight, ".experts.").? + ".experts".len], want);
            }
            try w.planes(name, want);
        };
    }

    fn layer(w: *Walk, c: *const Config, stem: []const u8, kind: config.Kind, l: Layout) !void {
        var buf: [512]u8 = undefined;
        try w.hc(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, tensors.hc.attn }), true);
        try w.hc(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, tensors.hc.mlp }), true);
        try w.proj(stem, tensors.moe.gate, .bf16);
        try w.proj(stem, tensors.moe.shared_gate, .bf16);
        for (tensors.moe.shared) |s| try w.proj(stem, s, l.dense);
        try w.experts(c, stem, l.routed);
        if (kind == .linear_attention) {
            for (tensors.linear.inputs) |s| try w.proj(stem, s, l.dense);
            try w.proj(stem, tensors.linear.out, l.dense);
            for ([_][]const u8{ tensors.linear.a_log, tensors.linear.dt_bias, tensors.linear.conv, tensors.linear.norm }) |s| try w.part(stem, s);
        } else {
            for (tensors.sparse.inputs) |s| try w.proj(stem, s, l.dense);
            try w.proj(stem, tensors.sparse.o, l.dense);
            for ([_][]const u8{ tensors.sparse.q_norm, tensors.sparse.k_norm, tensors.sparse.iq_norm, tensors.sparse.ik_norm }) |s| try w.part(stem, s);
        }
    }

    /// The n-gram block: vectors, bf16 key and value projections, and the NVFP4 table shards (read from disk).
    fn ple(w: *Walk, c: *const Config, stem: []const u8) !void {
        for (tensors.ple.vectors) |s| try w.part(stem, s);
        try w.proj(stem, tensors.ple.key, .bf16);
        try w.proj(stem, tensors.ple.value, .bf16);
        var buf: [512]u8 = undefined;
        var table: []const u8 = undefined;
        for (0..c.ngram_shards) |i| {
            const shard = try tensors.ple.table(&buf, stem, "shard_", i);
            try w.part(shard, "weight");
            try w.part(shard, "weight_scale");
            table = shard[0 .. std.mem.lastIndexOfScalar(u8, shard, '.').?];
        }
        const global = try w.take(try std.fmt.bufPrint(buf[table.len..], "{s}.weight_scale_2", .{table}));
        try w.recipe(global[0 .. global.len - ".weight_scale_2".len], .nvfp4);
    }
};

const Layout = struct { routed: Want, dense: Want };
const main_layers: Layout = .{ .routed = .nvfp4_csf, .dense = .mxfp8 };
const mtp_layers: Layout = .{ .routed = .w4a16, .dense = .bf16 };

fn holds(entry: ?std.json.Value, want: Want) bool {
    const v = entry orelse return want == .bf16;
    if (v != .object) return false;
    const o = v.object;
    const declared: struct { []const u8, i64 } = switch (want) {
        .bf16 => return false,
        .mxfp8 => .{ "MXFP8", 32 },
        .nvfp4, .nvfp4_csf => .{ "NVFP4", 16 },
        .w4a16 => .{ "W4A16_NVFP4", 16 },
    };
    if (!mc.holds(o.get("quant_algo") orelse return false, .{ .word = declared[0] }) or mc.int(o, "group_size") != declared[1]) return false;
    for (o.keys()) |key| if (!std.mem.eql(u8, key, "quant_algo") and !std.mem.eql(u8, key, "group_size") and !std.mem.eql(u8, key, "weight_scale_encoding")) return false;
    const encoding = mc.given(o, "weight_scale_encoding");
    if (want == .nvfp4_csf) return encoding != null and mc.holds(encoding.?, .{ .word = "csf" });
    return encoding == null;
}

fn holdsTensors(map: std.json.ObjectMap, module: []const u8) bool {
    for (map.keys()) |name| {
        if (name.len > module.len and name[module.len] == '.' and std.mem.startsWith(u8, name, module)) return true;
    }
    return false;
}

fn ignoredReason(name: []const u8) ?usize {
    for (ignored, 0..) |rule, i| {
        const hit = switch (rule.match) {
            .prefix => std.mem.startsWith(u8, name, rule.text),
            .suffix => std.mem.endsWith(u8, name, rule.text),
        };
        if (hit) return i;
    }
    return null;
}

/// Every tensor of `index_json` taken by a FP4-Local HIP reader or ignored by a rule; anything else is refused by name.
pub fn admitIndex(gpa: std.mem.Allocator, index_json: []const u8, c: *const Config, why: *Why) !Inventory {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, index_json, .{});
    const map = if (root == .object) mc.object(root.object, "weight_map") orelse return error.InvalidFlashIndex else return error.InvalidFlashIndex;
    var files = map.iterator();
    while (files.next()) |e| if (e.value_ptr.* != .string or !index.shardName(e.value_ptr.string)) return error.UnsafeFlashShard;
    const entries = mc.object(c.quant, "quantized_layers") orelse return error.UnsupportedQuantization;
    var w: Walk = .{ .a = a, .map = map, .entries = entries, .why = why };
    try w.matrix(tensors.embed, .bf16);
    try w.matrix(tensors.lm_head, .bf16);
    try w.hc(tensors.mixer, false);
    var stem: [128]u8 = undefined;
    for (0..c.layers) |i| {
        const l = try tensors.layer(&stem, i);
        try w.layer(c, l, try c.kind(i), main_layers);
        if (c.ple[i]) try w.ple(c, l);
    }
    if (c.mtp.layers > 0) {
        const m = index.mtpRoot(map);
        for ([_][]const u8{ tensors.mtp.fc_embedding, tensors.mtp.fc_hidden }) |s| try w.proj(m, s, .bf16);
        for ([_][]const u8{ tensors.mtp.norm_embedding, tensors.mtp.norm_hidden }) |s| try w.part(m, s);
        try w.hc(try std.fmt.bufPrint(&stem, "{s}.{s}", .{ m, tensors.mtp.mixer }), false);
        for (c.mtp.kinds[0..c.mtp.layers], 0..) |kind, i| {
            try w.layer(c, try std.fmt.bufPrint(&stem, "{s}.layers.{d}", .{ m, i }), kind, mtp_layers);
        }
    }
    var inv: Inventory = .{ .taken = w.taken.count() };
    for (entries.keys()) |module| if (!std.mem.startsWith(u8, module, ignored[0].text) and !w.declared.contains(module)) {
        // b9ce18a6 also declares mtp.layers.48.mlp.experts, a module with no tensor in its index
        if (!holdsTensors(map, module)) {
            inv.empty_modules += 1;
            continue;
        }
        why.set("quantization_config.quantized_layers {s} names a module FP4-Local's HIP reader reads as bf16 or not at all", .{module});
        return error.UnsupportedQuantization;
    };
    for (map.keys()) |name| if (!w.taken.contains(name)) {
        const rule = ignoredReason(name) orelse {
            why.set("{s} is in the index, and no FP4-Local HIP reader takes it", .{name});
            return error.UnexpectedFlashTensor;
        };
        if (rule == 0) inv.vision += 1 else inv.input_scales += 1;
    };
    return inv;
}

test {
    _ = @import("modelopt_admit_test.zig");
}
