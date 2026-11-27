//! The ModelOpt declarations, semantic settings and RoPE refused by name against the served Flash Next layout.

const std = @import("std");
const mc = @import("modelopt_config.zig");

const layers_n = 48;
const linear = blk: {
    var out: [layers_n]bool = undefined;
    for (&out, 0..) |*l, i| l.* = i % 4 != 3;
    break :blk out;
};
const served: mc.Served = .{ .linear = &linear, .eps = 1e-6, .ple_layer = 1, .rope_theta = 10_000_000, .partial_rotary = 0.25, .index_budget = 2048, .ple_dim = 2560 };

/// The served config's semantic text fields as the checked-in checkpoint has them: a test's base to change one by one.
pub fn servedText(gpa: std.mem.Allocator, s: mc.Served, text: *std.json.ObjectMap, rope_params: *std.json.ObjectMap, mtp: *std.json.ObjectMap, kinds: *std.json.Array, ple: *std.json.Array) !void {
    try text.put(gpa, "rms_norm_eps", .{ .float = 1e-06 });
    try text.put(gpa, "output_gate_type", .{ .string = "sigmoid" });
    try text.put(gpa, "hidden_act", .{ .string = "silu" });
    try text.put(gpa, "attention_bias", .{ .bool = false });
    try text.put(gpa, "partial_rotary_factor", .{ .float = 0.25 });
    try text.put(gpa, "indexer_budget", .{ .integer = 2048 });
    try text.put(gpa, "ple_conv_kernel_size", .{ .integer = 4 });
    try text.put(gpa, "ngram_size", .{ .integer = 3 });
    for (s.linear) |is_linear| try kinds.append(.{ .string = if (is_linear) "linear_attention" else "full_attention" });
    try text.put(gpa, "layer_types", .{ .array = kinds.* });
    try ple.append(.{ .integer = 2 });
    try text.put(gpa, "ple_layer_ids", .{ .array = ple.* });
    try rope_params.put(gpa, "rope_theta", .{ .integer = 10_000_000 });
    try rope_params.put(gpa, "partial_rotary_factor", .{ .float = 0.25 });
    try rope_params.put(gpa, "type", .{ .string = "default" });
    try text.put(gpa, "rope_parameters", .{ .object = rope_params.* });
    try mtp.put(gpa, "rope_theta", .{ .integer = 10_000_000 });
    try text.put(gpa, "mtp", .{ .object = mtp.* });
}

const Text = struct {
    text: std.json.ObjectMap = .empty,
    rope_params: std.json.ObjectMap = .empty,
    mtp: std.json.ObjectMap = .empty,
    kinds: std.json.Array,
    ple: std.json.Array,

    fn init(gpa: std.mem.Allocator) !Text {
        var t: Text = .{ .kinds = .init(gpa), .ple = .init(gpa) };
        try servedText(gpa, served, &t.text, &t.rope_params, &t.mtp, &t.kinds, &t.ple);
        return t;
    }
};

fn modelopt(text: std.json.ObjectMap, quant: std.json.ObjectMap) ?[]const u8 {
    return mc.quantMismatch(quant) orelse mc.semantics(text, served) orelse mc.layersMismatch(quant, served.linear);
}

test "a changed epsilon, RoPE, gate, activation or layer role is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const Change = struct { field: []const u8, value: std.json.Value, refused: []const u8 };
    const changes = [_]Change{
        .{ .field = "rms_norm_eps", .value = .{ .float = 1e-5 }, .refused = "rms_norm_eps" },
        .{ .field = "output_gate_type", .value = .{ .string = "silu" }, .refused = "output_gate_type" },
        .{ .field = "hidden_act", .value = .{ .string = "gelu" }, .refused = "hidden_act" },
        .{ .field = "attention_bias", .value = .{ .bool = true }, .refused = "attention_bias" },
        .{ .field = "attn_output_gate", .value = .{ .bool = false }, .refused = "attn_output_gate" },
        .{ .field = "norm_topk_prob", .value = .{ .bool = false }, .refused = "norm_topk_prob" },
        .{ .field = "rope_scaling", .value = .{ .string = "yarn" }, .refused = "rope_scaling" },
        .{ .field = "indexer_budget", .value = .{ .integer = 4096 }, .refused = "indexer_budget" },
        .{ .field = "ngram_size", .value = .{ .integer = 4 }, .refused = "ngram_size" },
        .{ .field = "ple_layer_ids", .value = .{ .array = .init(gpa) }, .refused = "ple_layer_ids" },
        .{ .field = "layer_types", .value = .{ .array = .init(gpa) }, .refused = "layer_types" },
    };
    for (changes) |c| {
        var t = try Text.init(gpa);
        try std.testing.expectEqual(@as(?[]const u8, null), mc.semantics(t.text, served));
        try t.text.put(gpa, c.field, c.value);
        try std.testing.expectEqualStrings(c.refused, mc.semantics(t.text, served) orelse "accepted");
    }
}

test "each layer's label is exactly the role the kernels run: a sliding or misspelt one is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const full = 3; // the first full-attention layer
    try std.testing.expect(!linear[full] and linear[0]);
    const Case = struct { layer: usize, label: []const u8 };
    const cases = [_]Case{
        .{ .layer = full, .label = "sliding_attention" },
        .{ .layer = full, .label = "attention" },
        .{ .layer = full, .label = "Full_Attention" },
        .{ .layer = layers_n - 1, .label = "sliding_attention" },
        .{ .layer = 0, .label = "full_attention" },
        .{ .layer = 0, .label = "sliding_attention" },
    };
    for (cases) |c| {
        var t = try Text.init(gpa);
        try std.testing.expectEqual(@as(?[]const u8, null), mc.semantics(t.text, served));
        t.kinds.items[c.layer] = .{ .string = c.label };
        try t.text.put(gpa, "layer_types", .{ .array = t.kinds });
        try std.testing.expectEqualStrings("layer_types", mc.semantics(t.text, served) orelse "accepted");
    }
}

test "RoPE's base and type inside rope_parameters, and the MTP layer's base, are checked too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const Case = struct { in_mtp: bool, field: []const u8, value: std.json.Value, refused: []const u8 };
    const cases = [_]Case{
        .{ .in_mtp = false, .field = "rope_theta", .value = .{ .integer = 1_000_000 }, .refused = "rope_theta" },
        .{ .in_mtp = false, .field = "partial_rotary_factor", .value = .{ .float = 0.5 }, .refused = "partial_rotary_factor" },
        .{ .in_mtp = false, .field = "type", .value = .{ .string = "yarn" }, .refused = "rope_parameters.rope_type" },
        .{ .in_mtp = false, .field = "factor", .value = .{ .float = 4 }, .refused = "rope_parameters.factor" },
        .{ .in_mtp = true, .field = "rope_theta", .value = .{ .integer = 5_000_000 }, .refused = "mtp.rope_theta" },
        .{ .in_mtp = true, .field = "layer_types", .value = .{ .array = .init(gpa) }, .refused = "mtp.layer_types" },
    };
    for (cases) |c| {
        var t = try Text.init(gpa);
        if (c.in_mtp) {
            try t.mtp.put(gpa, c.field, c.value);
            try t.text.put(gpa, "mtp", .{ .object = t.mtp });
        } else {
            try t.rope_params.put(gpa, c.field, c.value);
            try t.text.put(gpa, "rope_parameters", .{ .object = t.rope_params });
        }
        try std.testing.expectEqualStrings(c.refused, mc.semantics(t.text, served) orelse "accepted");
    }
}

test "mixed precision requires every routed, shared and attention declaration and refuses altered formats" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try Text.init(a);
    const text = t.text;
    var quant: std.json.ObjectMap = .empty;
    try quant.put(a, "quant_method", .{ .string = "modelopt" });
    try quant.put(a, "quant_algo", .{ .string = "MIXED_PRECISION" });
    var entries: std.json.ObjectMap = .empty;
    for (0..layers_n) |layer| {
        const routed = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.experts", .{layer});
        var nv: std.json.ObjectMap = .empty;
        try nv.put(a, "quant_algo", .{ .string = "NVFP4" });
        try nv.put(a, "group_size", .{ .integer = 16 });
        try nv.put(a, "weight_scale_encoding", .{ .string = "csf" });
        try entries.put(a, routed, .{ .object = nv });
        const prefix = if (linear[layer]) "linear_attn" else "self_attn";
        const parts: []const []const u8 = if (linear[layer]) &.{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj" } else &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj" };
        for (parts) |part| try addMxfp8(a, &entries, layer, prefix, part);
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |part| try addMxfp8(a, &entries, layer, "mlp.shared_expert", part);
    }
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expect(modelopt(text, quant) == null);
    const cases = [_]struct { name: []const u8, field: []const u8, value: std.json.Value, refused: []const u8 }{
        .{ .name = "model.language_model.layers.47.mlp.experts", .field = "group_size", .value = .{ .integer = 32 }, .refused = "quantization_config.routed_experts" },
        .{ .name = "model.language_model.layers.0.mlp.experts", .field = "weight_scale_encoding", .value = .{ .string = "other" }, .refused = "quantization_config.weight_scale_encoding" },
        .{ .name = "model.language_model.layers.17.mlp.shared_expert.down_proj", .field = "quant_algo", .value = .{ .string = "NVFP4" }, .refused = "quantization_config.shared_expert" },
        .{ .name = "model.language_model.layers.0.linear_attn.in_proj_b", .field = "group_size", .value = .{ .integer = 16 }, .refused = "quantization_config.attention_projection" },
        .{ .name = "model.language_model.layers.47.self_attn.indexer.index_qk_proj", .field = "quant_algo", .value = .{ .string = "FP8" }, .refused = "quantization_config.attention_projection" },
    };
    for (cases) |c| {
        const entry = entries.getPtr(c.name).?;
        const original = entry.object.get(c.field).?;
        try entry.object.put(a, c.field, c.value);
        try std.testing.expectEqualStrings(c.refused, modelopt(text, quant) orelse "accepted");
        try entry.object.put(a, c.field, original);
    }
    try quant.put(a, "kv_cache_quant_algo", .{ .string = "FP8" });
    try std.testing.expectEqualStrings("quantization_config.kv_cache_quant_algo", modelopt(text, quant).?);
    try quant.put(a, "kv_cache_quant_algo", .null);
    try std.testing.expect(modelopt(text, quant) == null);
    // a projection loaded alone may be BF16 (unlisted); a stitched group may not mix MXFP8 and BF16 parts
    _ = entries.swapRemove("model.language_model.layers.47.self_attn.o_proj");
    _ = entries.swapRemove("model.language_model.layers.3.mlp.shared_expert.up_proj");
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expect(modelopt(text, quant) == null);
    _ = entries.swapRemove("model.language_model.layers.47.self_attn.k_proj");
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expectEqualStrings("quantization_config.attention_projection", modelopt(text, quant).?);
    try addMxfp8(a, &entries, 47, "self_attn", "k_proj");
    _ = entries.swapRemove("model.language_model.layers.0.linear_attn.in_proj_a");
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expectEqualStrings("quantization_config.attention_projection", modelopt(text, quant).?);
}

test "NVIDIA's NVFP4 layout passes: routed experts NVFP4 with plain scales, every other projection BF16 (unlisted)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try Text.init(a);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(a, "quant_method", .{ .string = "modelopt" });
    try quant.put(a, "quant_algo", .{ .string = "MIXED_PRECISION" });
    var entries: std.json.ObjectMap = .empty;
    for (0..layers_n) |layer| {
        var nv: std.json.ObjectMap = .empty;
        try nv.put(a, "quant_algo", .{ .string = "NVFP4" });
        try nv.put(a, "group_size", .{ .integer = 16 });
        try entries.put(a, try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.experts", .{layer}), .{ .object = nv });
    }
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expect(modelopt(t.text, quant) == null);
    // one shared projection declared in another format is still refused
    var nv: std.json.ObjectMap = .empty;
    try nv.put(a, "quant_algo", .{ .string = "NVFP4" });
    try nv.put(a, "group_size", .{ .integer = 16 });
    try entries.put(a, "model.language_model.layers.5.mlp.shared_expert.gate_proj", .{ .object = nv });
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expectEqualStrings("quantization_config.shared_expert", modelopt(t.text, quant).?);
}

fn addMxfp8(a: std.mem.Allocator, entries: *std.json.ObjectMap, layer: usize, prefix: []const u8, part: []const u8) !void {
    var p: std.json.ObjectMap = .empty;
    try p.put(a, "quant_algo", .{ .string = "MXFP8" });
    try p.put(a, "group_size", .{ .integer = 32 });
    const name = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.{s}.{s}", .{ layer, prefix, part });
    try entries.put(a, name, .{ .object = p });
}
