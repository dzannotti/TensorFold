//! A qwen35 GGUF's tensor names as the MLX conversions name them, so the checkpoint reads either. The names are
//! llama.cpp's (gguf-py/gguf/constants.py and tensor_mapping.py, src/models/qwen35.cpp), the transforms its
//! conversion applied are conversion/qwen.py's (Qwen3NextModel and _LinearAttentionVReorderBase.modify_tensors):
//!   - norms but linear_attn.norm hold 1 + w (as the MLX conversions bake them);
//!   - ssm_a is -exp(A_log), named linear_attn.ssm_a here (the checkpoint takes the log back);
//!   - every linear_attn tensor keeps its value heads in llama.cpp's tiled order (value head j * key_heads + g reads key
//!     head g), which the model runs as is (Spec.tiled_heads).
//! The MTP block (blk.<n_layer>, nextn_predict_layers of them) is the MLX `mtp.` head.

const std = @import("std");

const prefix = "language_model.model.";

/// GGUF suffixes of a layer and the MLX ones (both without the trailing ".weight" of a projection).
const layer_names = [_][2][]const u8{
    .{ "attn_norm.weight", "input_layernorm.weight" },
    .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
    .{ "attn_qkv", "linear_attn.in_proj_qkv" },
    .{ "attn_gate", "linear_attn.in_proj_z" },
    .{ "ssm_alpha", "linear_attn.in_proj_a" },
    .{ "ssm_beta", "linear_attn.in_proj_b" },
    .{ "ssm_out", "linear_attn.out_proj" },
    .{ "ssm_conv1d", "linear_attn.conv1d" },
    .{ "ssm_dt.bias", "linear_attn.dt_bias" },
    .{ "ssm_a", "linear_attn.ssm_a" },
    .{ "ssm_norm.weight", "linear_attn.norm.weight" },
    .{ "attn_q", "self_attn.q_proj" },
    .{ "attn_k", "self_attn.k_proj" },
    .{ "attn_v", "self_attn.v_proj" },
    .{ "attn_output", "self_attn.o_proj" },
    .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
    .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
    .{ "ffn_gate", "mlp.gate_proj" },
    .{ "ffn_up", "mlp.up_proj" },
    .{ "ffn_down", "mlp.down_proj" },
};

/// The MTP block's own tensors (conversion/qwen.py _QwenMtpMixin's remapper, reversed).
const nextn_names = [_][2][]const u8{
    .{ "nextn.eh_proj", "mtp.fc" },
    .{ "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight" },
    .{ "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight" },
    .{ "nextn.shared_head_norm.weight", "mtp.norm.weight" },
};

const top_names = [_][2][]const u8{
    .{ "token_embd", prefix ++ "embed_tokens" },
    .{ "output_norm.weight", prefix ++ "norm.weight" },
    .{ "output", "language_model.lm_head" },
};

/// The MLX name of GGUF tensor `name` (a projection's without ".weight": the caller adds its suffix), or null for a
/// tensor the model does not read. `layers` is the trunk's (block_count less the nextn layers).
pub fn mlx(a: std.mem.Allocator, name: []const u8, layers: u64) !?[]const u8 {
    const base = if (std.mem.endsWith(u8, name, ".weight") and !isNorm(name)) name[0 .. name.len - ".weight".len] else name;
    for (top_names) |p| if (std.mem.eql(u8, base, p[0])) return p[1];
    if (!std.mem.startsWith(u8, base, "blk.")) return null;
    const dot = std.mem.indexOfScalarPos(u8, base, 4, '.') orelse return null;
    const index = std.fmt.parseInt(u64, base[4..dot], 10) catch return null;
    const rest = base[dot + 1 ..];
    if (index >= layers) {
        if (index != layers) return null; // a second MTP block: the head reads one
        for (nextn_names) |p| if (std.mem.eql(u8, rest, p[0])) return p[1];
        for (layer_names) |p| if (std.mem.eql(u8, rest, p[0])) return try std.fmt.allocPrint(a, "mtp.layers.0.{s}", .{p[1]});
        return null;
    }
    for (layer_names) |p| if (std.mem.eql(u8, rest, p[0])) return try std.fmt.allocPrint(a, prefix ++ "layers.{d}.{s}", .{ index, p[1] });
    return null;
}

fn isNorm(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "norm.weight");
}

test "trunk, top and MTP names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("language_model.model.layers.3.self_attn.q_proj", (try mlx(a, "blk.3.attn_q.weight", 64)).?);
    try std.testing.expectEqualStrings("language_model.model.layers.0.linear_attn.dt_bias", (try mlx(a, "blk.0.ssm_dt.bias", 64)).?);
    try std.testing.expectEqualStrings("language_model.model.layers.0.input_layernorm.weight", (try mlx(a, "blk.0.attn_norm.weight", 64)).?);
    try std.testing.expectEqualStrings("language_model.model.embed_tokens", (try mlx(a, "token_embd.weight", 64)).?);
    try std.testing.expectEqualStrings("language_model.model.norm.weight", (try mlx(a, "output_norm.weight", 64)).?);
    try std.testing.expectEqualStrings("mtp.fc", (try mlx(a, "blk.64.nextn.eh_proj.weight", 64)).?);
    try std.testing.expectEqualStrings("mtp.layers.0.mlp.down_proj", (try mlx(a, "blk.64.ffn_down.weight", 64)).?);
    try std.testing.expectEqualStrings("mtp.norm.weight", (try mlx(a, "blk.64.nextn.shared_head_norm.weight", 64)).?);
    try std.testing.expect((try mlx(a, "blk.65.ffn_down.weight", 64)) == null);
    try std.testing.expect((try mlx(a, "rope_freqs.weight", 64)) == null);
}
