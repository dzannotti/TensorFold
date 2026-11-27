//! The checkpoint's config.json against what the CUDA path is built for: another shape or setting is refused at load.

const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const widths = @import("cuda_widths.zig");
const mc = @import("modelopt_config.zig");

/// One config field and the value the kernels and buffers assume.
const Want = struct { name: []const u8, value: i64 };

/// The text config's shapes the CUDA path's constants hold.
const shapes = [_]Want{
    .{ .name = "hidden_size", .value = wt.dims },
    .{ .name = "num_hidden_layers", .value = wt.layers_n },
    .{ .name = "full_attention_interval", .value = 4 },
    .{ .name = "vocab_size", .value = wt.vocab },
    .{ .name = "num_attention_heads", .value = wt.q_heads },
    .{ .name = "num_key_value_heads", .value = wt.kv_heads },
    .{ .name = "head_dim", .value = wt.head_dim },
    .{ .name = "num_experts", .value = wt.experts_n },
    .{ .name = "moe_intermediate_size", .value = wt.moe_w },
    .{ .name = "hc_count", .value = wt.streams },
    .{ .name = "hc_lowrank", .value = wt.low },
    .{ .name = "linear_num_key_heads", .value = gdn.nk },
    .{ .name = "linear_num_value_heads", .value = gdn.nv },
    .{ .name = "linear_key_head_dim", .value = gdn.dk },
    .{ .name = "linear_value_head_dim", .value = gdn.dv },
    .{ .name = "linear_conv_kernel_dim", .value = gdn.taps },
    .{ .name = "indexer_n_heads", .value = wt.index_heads },
    .{ .name = "indexer_head_dim", .value = wt.index_dim },
    .{ .name = "indexer_compress_ratio", .value = wt.index_ratio },
    .{ .name = "mtp_num_hidden_layers", .value = 1 },
};

/// RoPE's base: Scratch.inv's 10,000,000 (the Python engine's default when the config names none).
pub const rope_theta: f64 = 10_000_000;
/// The rotated share of a head: 64 of 256 features, `wt.half` pairs.
pub const partial_rotary: f64 = @as(f64, @floatFromInt(2 * wt.half)) / @as(f64, @floatFromInt(wt.head_dim));

const linear_layers = blk: {
    var out: [wt.layers_n]bool = undefined;
    for (&out, 0..) |*l, i| l.* = wt.linearLayer(i);
    break :blk out;
};

/// The layout the CUDA kernels and buffers fix, as modelopt_config.zig checks a config against it.
const served_layout: mc.Served = .{
    .linear = &linear_layers,
    .eps = wt.eps,
    .ple_layer = wt.ple_layer,
    .rope_theta = rope_theta,
    .partial_rotary = partial_rotary,
    .index_budget = wt.index_top * wt.index_ratio,
    .ple_dim = wt.dims,
};

const int = mc.int;
const object = mc.object;
const given = mc.given;
const holds = mc.holds;

/// The first config value that differs from what the CUDA path serves, or null when every one matches.
pub fn mismatch(text: std.json.ObjectMap, quant: ?std.json.ObjectMap) ?[]const u8 {
    const q = quant orelse return "quantization";
    if (int(q, "bits") != 4) return "quantization.bits";
    if (int(q, "group_size") != 32) return "quantization.group_size";
    if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return "quantization.mode";
    if (geometry(text)) |field| return field;
    // the MLX path's experts and captured kernels route top-10 with a 640-wide shared expert only
    return if (widthsOf(text).widths.isDefault()) null else "num_experts_per_tok";
}

/// The routing and shared widths `text` names: a supported shape, or the field whose value has none.
pub fn widthsOf(text: std.json.ObjectMap) widths.Selected {
    return widths.select(int(text, "num_experts_per_tok"), int(text, "shared_expert_intermediate_size"), int(text, learned_key) orelse 0);
}

fn geometry(text: std.json.ObjectMap) ?[]const u8 {
    // a folder Living Weights taught widens the shared expert by its pocket (a whole number of groups, 128 at most)
    const learned = int(text, learned_key) orelse 0;
    if (!widths.learnedOk(learned)) return learned_key;
    for (shapes) |w| if (int(text, w.name) != w.value) return w.name;
    switch (widthsOf(text)) {
        .field => |field| return field,
        .widths => {},
    }
    return mc.semantics(text, served_layout);
}

pub const Mode = enum { affine, modelopt, tfq3 };

fn tfq3Mismatch(text: std.json.ObjectMap, quant: ?std.json.ObjectMap) ?[]const u8 {
    if (quant) |q| {
        const bits = int(q, "bits") orelse return "quantization.bits";
        if (bits != 4 and bits != 8) return "quantization.bits";
        if (int(q, "group_size") != 32) return "quantization.group_size";
        if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return "quantization.mode";
    }
    return geometry(text);
}

fn modeloptMismatch(text: std.json.ObjectMap, quant: std.json.ObjectMap) ?[]const u8 {
    return mc.quantMismatch(quant) orelse geometry(text) orelse mc.layersMismatch(quant, served_layout.linear);
}

/// The neurons a Living Weights pocket adds to each shared expert: cuda_widths.zig owns the limits and the name.
const learned_key = widths.learned_key;

/// Reads `dir`/config.json; error.UnsupportedCheckpoint names the field in a log line when one differs.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    _ = try read(gpa, io, dir);
}

pub fn mode(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Mode {
    return (try read(gpa, io, dir)).mode;
}

/// What a checked config.json selects: the quantization's mode and the routing and shared widths.
pub const Checked = struct { mode: Mode, widths: widths.Widths };

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Checked {
    const parsed = try parse(gpa, io, dir);
    defer parsed.deinit();
    const root = parsed.value.object;
    const text = textOf(root);
    const quant = if (root.get("quantization")) |v| (if (v == .object) v.object else null) else null;
    const is_tfq3 = @import("cuda_tfq3.zig").isTfq3(gpa, io, dir);
    const selected: Mode = if (is_tfq3) .tfq3 else if (quant != null) .affine else .modelopt;
    var field = switch (selected) {
        .tfq3 => tfq3Mismatch(text, quant),
        .affine => mismatch(text, quant),
        .modelopt => if (object(root, "quantization_config")) |q| modeloptMismatch(text, q) else "quantization",
    };
    if (field == null) if (given(root, "tie_word_embeddings")) |v| if (!holds(v, .{ .flag = false })) {
        field = "tie_word_embeddings";
    };
    if (field == null) return .{ .mode = selected, .widths = widthsOf(text).widths };
    std.log.err("flashnext: config.json's {s} differs from the supported Flash Next CUDA layout", .{field.?});
    return error.UnsupportedCheckpoint;
}

/// The routing and shared widths `dir`/config.json selects; a width outside the supported list is refused by name.
pub fn widthsIn(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !widths.Widths {
    const parsed = try parse(gpa, io, dir);
    defer parsed.deinit();
    switch (widthsOf(textOf(parsed.value.object))) {
        .widths => |w| return w,
        .field => |field| {
            std.log.err("flashnext: config.json's {s} differs from the supported Flash Next CUDA layout", .{field});
            return error.UnsupportedCheckpoint;
        },
    }
}

fn parse(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    if (parsed.value != .object) {
        parsed.deinit();
        return error.UnsupportedCheckpoint;
    }
    return parsed;
}

fn textOf(root: std.json.ObjectMap) std.json.ObjectMap {
    return if (root.get("text_config")) |v| (if (v == .object) v.object else root) else root;
}

/// The served config's text fields, as the checked-in checkpoint has them (a test's base to change one at a time).
fn served(gpa: std.mem.Allocator, text: *std.json.ObjectMap, rope_params: *std.json.ObjectMap, mtp: *std.json.ObjectMap, kinds: *std.json.Array, ple: *std.json.Array) !void {
    for (shapes) |w| try text.put(gpa, w.name, .{ .integer = w.value });
    try text.put(gpa, "num_experts_per_tok", .{ .integer = @intCast(widths.default.top_k) });
    try text.put(gpa, "shared_expert_intermediate_size", .{ .integer = @intCast(widths.default.shared) });
    try @import("modelopt_config_test.zig").servedText(gpa, served_layout, text, rope_params, mtp, kinds, ple);
}

test "the served shapes pass and a 6-bit checkpoint or another width is named" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var text: std.json.ObjectMap = .empty;
    var rope_params: std.json.ObjectMap = .empty;
    var mtp: std.json.ObjectMap = .empty;
    var kinds: std.json.Array = .init(gpa);
    var ple: std.json.Array = .init(gpa);
    try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try quant.put(gpa, "group_size", .{ .integer = 32 });
    try std.testing.expectEqual(@as(?[]const u8, null), mismatch(text, quant));
    try quant.put(gpa, "bits", .{ .integer = 6 });
    try std.testing.expectEqualStrings("quantization.bits", mismatch(text, quant).?);
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try text.put(gpa, "hidden_size", .{ .integer = 4096 });
    try std.testing.expectEqualStrings("hidden_size", mismatch(text, quant).?);
    try std.testing.expectEqualStrings("quantization", mismatch(text, null).?);
}

test "a top-5 config with a 1280-wide shared expert selects that shape, and the MLX path refuses it by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var text: std.json.ObjectMap = .empty;
    var rope_params: std.json.ObjectMap = .empty;
    var mtp: std.json.ObjectMap = .empty;
    var kinds: std.json.Array = .init(gpa);
    var ple: std.json.Array = .init(gpa);
    try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
    try std.testing.expectEqual(widths.default, widthsOf(text).widths);
    try text.put(gpa, "num_experts_per_tok", .{ .integer = 5 });
    try text.put(gpa, "shared_expert_intermediate_size", .{ .integer = 1280 });
    try std.testing.expectEqual(@as(?[]const u8, null), geometry(text));
    try std.testing.expectEqual(widths.Widths{ .top_k = 5, .shared = 1280 }, widthsOf(text).widths);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try quant.put(gpa, "group_size", .{ .integer = 32 });
    try std.testing.expectEqualStrings("num_experts_per_tok", mismatch(text, quant).?);
    try text.put(gpa, "shared_expert_intermediate_size", .{ .integer = 640 });
    try std.testing.expectEqualStrings("shared_expert_intermediate_size", geometry(text).?);
    try text.put(gpa, "num_experts_per_tok", .{ .integer = 8 });
    try std.testing.expectEqualStrings("num_experts_per_tok", geometry(text).?);
}

test "a taught folder's wider shared expert passes with its pocket's count, and not without it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var text: std.json.ObjectMap = .empty;
    var rope_params: std.json.ObjectMap = .empty;
    var mtp: std.json.ObjectMap = .empty;
    var kinds: std.json.Array = .init(gpa);
    var ple: std.json.Array = .init(gpa);
    try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try quant.put(gpa, "group_size", .{ .integer = 32 });
    try text.put(gpa, "shared_expert_intermediate_size", .{ .integer = wt.moe_w + 64 });
    try std.testing.expectEqualStrings("shared_expert_intermediate_size", mismatch(text, quant).?);
    try text.put(gpa, learned_key, .{ .integer = 64 });
    try std.testing.expectEqual(@as(?[]const u8, null), mismatch(text, quant));
    try text.put(gpa, learned_key, .{ .integer = 40 });
    try std.testing.expectEqualStrings(learned_key, mismatch(text, quant).?);
}

test "the standalone width read takes a taught pocket count and its selection refuses the signed maximum" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    // a folder taught 64 neurons: 704 shared rows, read from config.json as the standalone engine reads it
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const taught =
        \\{"text_config": {"num_experts_per_tok": 10, "shared_expert_intermediate_size": 704, "tensorfold_learned_neurons": 64}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = taught });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    try std.testing.expectEqual(widths.default, try widthsIn(gpa, io, dir));
    // the same fields with the signed maximum: the selection names the field rather than summing it into the width
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa,
        \\{"num_experts_per_tok": 10, "shared_expert_intermediate_size": 640, "tensorfold_learned_neurons": 9223372036854775807}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(learned_key, widthsOf(parsed.value.object).field);
}

test "the CUDA ModelOpt check on the served text: accepted, then refused at the header, the shapes and a layer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.json.ObjectMap = .empty;
    var rope_params: std.json.ObjectMap = .empty;
    var mtp: std.json.ObjectMap = .empty;
    var kinds: std.json.Array = .init(a);
    var ple: std.json.Array = .init(a);
    try served(a, &text, &rope_params, &mtp, &kinds, &ple);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(a, "quant_method", .{ .string = "modelopt" });
    try quant.put(a, "quant_algo", .{ .string = "MIXED_PRECISION" });
    var entries: std.json.ObjectMap = .empty;
    var nv: std.json.ObjectMap = .empty;
    try nv.put(a, "quant_algo", .{ .string = "NVFP4" });
    try nv.put(a, "group_size", .{ .integer = 16 });
    for (0..wt.layers_n) |layer| {
        try entries.put(a, try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.experts", .{layer}), .{ .object = nv });
    }
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expect(modeloptMismatch(text, quant) == null);
    try quant.put(a, "quant_algo", .{ .string = "NVFP4" });
    try std.testing.expectEqualStrings("quantization_config.quant_algo", modeloptMismatch(text, quant).?);
    try quant.put(a, "quant_algo", .{ .string = "MIXED_PRECISION" });
    try text.put(a, "hidden_size", .{ .integer = 4096 });
    try std.testing.expectEqualStrings("hidden_size", modeloptMismatch(text, quant).?);
    try text.put(a, "hidden_size", .{ .integer = wt.dims });
    try std.testing.expect(modeloptMismatch(text, quant) == null);
    var wide: std.json.ObjectMap = .empty;
    try wide.put(a, "quant_algo", .{ .string = "NVFP4" });
    try wide.put(a, "group_size", .{ .integer = 32 });
    try entries.put(a, "model.language_model.layers.47.mlp.experts", .{ .object = wide });
    try quant.put(a, "quantized_layers", .{ .object = entries });
    try std.testing.expectEqualStrings("quantization_config.routed_experts", modeloptMismatch(text, quant).?);
}

test {
    _ = @import("cuda_widths.zig");
    _ = @import("modelopt_config.zig");
}
