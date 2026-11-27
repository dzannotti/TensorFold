//! A ModelOpt config.json against a served layout: the quantization declarations, semantic settings and RoPE.

const std = @import("std");

/// What a backend's kernels fix: each layer's role (linear attention or full), eps, the PLE layer, RoPE and widths.
pub const Served = struct {
    linear: []const bool,
    eps: f32,
    ple_layer: usize,
    rope_theta: f64,
    partial_rotary: f64,
    index_budget: usize,
    ple_dim: usize,
};

/// Settings the kernels fix, absent meaning the value they assume: a whole number, a float, a word or a flag.
pub const Fixed = struct { name: []const u8, value: union(enum) { int: i64, float: f64, word: []const u8, flag: bool } };

/// The text config's semantic settings the evaluation hard-codes (norms, gates, activations, layer roles, n-grams).
fn fixed(s: Served) [13]Fixed {
    return .{
        .{ .name = "output_gate_type", .value = .{ .word = "sigmoid" } },
        .{ .name = "hidden_act", .value = .{ .word = "silu" } },
        .{ .name = "attention_bias", .value = .{ .flag = false } },
        .{ .name = "attn_output_gate", .value = .{ .flag = true } },
        .{ .name = "norm_topk_prob", .value = .{ .flag = true } },
        .{ .name = "mamba_ssm_dtype", .value = .{ .word = "float32" } },
        .{ .name = "tie_word_embeddings", .value = .{ .flag = false } },
        .{ .name = "mtp_use_dedicated_embeddings", .value = .{ .flag = false } },
        .{ .name = "indexer_budget", .value = .{ .int = @intCast(s.index_budget) } },
        .{ .name = "indexer_kv_heads", .value = .{ .int = 1 } },
        .{ .name = "ple_embed_dim", .value = .{ .int = @intCast(s.ple_dim) } },
        .{ .name = "ple_conv_kernel_size", .value = .{ .int = 4 } },
        .{ .name = "ngram_size", .value = .{ .int = 3 } },
    };
}

pub fn int(o: std.json.ObjectMap, name: []const u8) ?i64 {
    const v = o.get(name) orelse return null;
    return if (v == .integer) v.integer else null;
}

pub fn float(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

pub fn object(o: std.json.ObjectMap, name: []const u8) ?std.json.ObjectMap {
    const v = o.get(name) orelse return null;
    return if (v == .object) v.object else null;
}

/// A value present and not null.
pub fn given(o: std.json.ObjectMap, name: []const u8) ?std.json.Value {
    const v = o.get(name) orelse return null;
    return if (v == .null) null else v;
}

pub fn holds(v: std.json.Value, want: @FieldType(Fixed, "value")) bool {
    return switch (want) {
        .int => |i| v == .integer and v.integer == i,
        .float => |f| float(v) == f,
        .word => |w| v == .string and std.mem.eql(u8, v.string, w),
        .flag => |b| v == .bool and v.bool == b,
    };
}

fn projection(q: std.json.ObjectMap, name: []const u8, algo: []const u8, group: i64) bool {
    const p = object(q, name) orelse return false;
    return holds(p.get("quant_algo") orelse return false, .{ .word = algo }) and int(p, "group_size") == group;
}

/// The quantization_config's method, algorithm and KV cache: the first that is not ModelOpt mixed precision.
pub fn quantMismatch(quant: std.json.ObjectMap) ?[]const u8 {
    if (!holds(quant.get("quant_method") orelse return "quantization_config.quant_method", .{ .word = "modelopt" })) return "quantization_config.quant_method";
    if (!holds(quant.get("quant_algo") orelse return "quantization_config.quant_algo", .{ .word = "MIXED_PRECISION" })) return "quantization_config.quant_algo";
    if (given(quant, "kv_cache_quant_algo") != null) return "quantization_config.kv_cache_quant_algo";
    return null;
}

/// Each layer's routed, shared and attention declarations: the first one outside the served formats.
pub fn layersMismatch(quant: std.json.ObjectMap, linear: []const bool) ?[]const u8 {
    const entries = object(quant, "quantized_layers") orelse return "quantization_config.quantized_layers";
    var buf: [256]u8 = undefined;
    for (linear, 0..) |is_linear, layer| {
        const routed = std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts", .{layer}) catch unreachable;
        if (!projection(entries, routed, "NVFP4", 16)) return "quantization_config.routed_experts";
        const p = object(entries, routed).?;
        if (given(p, "weight_scale_encoding")) |encoding| if (!holds(encoding, .{ .word = "csf" })) return "quantization_config.weight_scale_encoding";
        // each projection MXFP8/32 (FP4-Local) or unlisted (BF16, NVIDIA's NVFP4); a stitched group one or the other
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |part| {
            if (declared(entries, layer, "mlp.shared_expert", &.{part}) == null) return "quantization_config.shared_expert";
        }
        const prefix = if (is_linear) "linear_attn" else "self_attn";
        const stitched: []const []const u8 = if (is_linear) &.{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" } else &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" };
        if (declared(entries, layer, prefix, stitched) == null) return "quantization_config.attention_projection";
        if (declared(entries, layer, prefix, &.{if (is_linear) "out_proj" else "o_proj"}) == null) return "quantization_config.attention_projection";
    }
    return null;
}

/// One loaded projection's parts: all MXFP8/32, or none listed (BF16); null when mixed or another format.
pub fn declared(entries: std.json.ObjectMap, layer: usize, prefix: []const u8, parts: []const []const u8) ?enum { mxfp8, bf16 } {
    var listed: usize = 0;
    var buf: [256]u8 = undefined;
    for (parts) |part| {
        const name = std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.{s}.{s}", .{ layer, prefix, part }) catch unreachable;
        if (entries.get(name) == null) continue;
        if (!projection(entries, name, "MXFP8", 32)) return null;
        listed += 1;
    }
    return if (listed == 0) .bf16 else if (listed == parts.len) .mxfp8 else null;
}

/// The first semantic setting the kernels would ignore: eps, RoPE, gates, activations, layer roles, the head's layer.
pub fn semantics(text: std.json.ObjectMap, s: Served) ?[]const u8 {
    const eps = float(given(text, "rms_norm_eps") orelse return "rms_norm_eps") orelse return "rms_norm_eps";
    if (@as(f32, @floatCast(eps)) != s.eps) return "rms_norm_eps"; // the kernels take eps as fp32
    for (fixed(s)) |f| if (given(text, f.name)) |v| if (!holds(v, f.value)) return f.name;
    if (rope(text, s)) |field| return field;
    if (given(text, "layer_types")) |v| {
        if (v != .array or v.array.items.len != s.linear.len) return "layer_types";
        for (v.array.items, s.linear) |kind, is_linear| {
            const want = if (is_linear) "linear_attention" else "full_attention";
            if (!holds(kind, .{ .word = want })) return "layer_types";
        }
    }
    const ple = given(text, "ple_layer_ids") orelse return "ple_layer_ids";
    if (ple != .array or ple.array.items.len != 1 or !holds(ple.array.items[0], .{ .int = @intCast(s.ple_layer + 1) })) return "ple_layer_ids";
    if (object(text, "mtp")) |m| {
        if (given(m, "rope_theta")) |v| if (float(v) != s.rope_theta) return "mtp.rope_theta";
        if (given(m, "num_hidden_layers")) |v| if (!holds(v, .{ .int = 1 })) return "mtp.num_hidden_layers";
        if (given(m, "layer_types")) |v| {
            if (v != .array or v.array.items.len != 1 or !holds(v.array.items[0], .{ .word = "full_attention" })) return "mtp.layer_types";
        }
    }
    return null;
}

/// RoPE as the kernels apply it: the served base on the served share of a head, no scaling (text positions only).
fn rope(text: std.json.ObjectMap, s: Served) ?[]const u8 {
    if (given(text, "rope_scaling") != null) return "rope_scaling";
    const params = object(text, "rope_parameters");
    const theta = (if (params) |p| given(p, "rope_theta") else null) orelse given(text, "rope_theta");
    if (theta) |v| if (float(v) != s.rope_theta) return "rope_theta";
    const partial = (if (params) |p| given(p, "partial_rotary_factor") else null) orelse given(text, "partial_rotary_factor");
    if (partial) |v| if (float(v) != s.partial_rotary) return "partial_rotary_factor";
    const p = params orelse return null;
    for ([_][]const u8{ "rope_type", "type" }) |name| if (given(p, name)) |v| if (!holds(v, .{ .word = "default" })) return "rope_parameters.rope_type";
    if (given(p, "factor")) |v| if (float(v) != 1) return "rope_parameters.factor";
    return null;
}

test {
    _ = @import("modelopt_config_test.zig");
}
