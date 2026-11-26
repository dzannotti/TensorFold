//! FP4-Local on HIP: the config, every index tensor and the admitted bytes, synthetic and on the real checkpoint.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const plan = @import("plan.zig");
const admit = @import("modelopt_admit.zig");

/// A synthetic FP4-Local: the 6-bit fixture's text with `layers`, `experts` and `shards`, ModelOpt's recipes.
const Shape = struct { layers: usize = 48, experts: usize = 512, shards: usize = 128 };

fn put(a: std.mem.Allocator, o: *std.json.ObjectMap, key: []const u8, algo: []const u8, group: i64, csf: bool) !void {
    var e: std.json.ObjectMap = .empty;
    try e.put(a, "quant_algo", .{ .string = algo });
    try e.put(a, "group_size", .{ .integer = group });
    if (csf) try e.put(a, "weight_scale_encoding", .{ .string = "csf" });
    try o.put(a, key, .{ .object = e });
}

fn linear(s: Shape, i: usize) bool {
    _ = s;
    return i % 4 != 3;
}

fn configText(a: std.mem.Allocator, s: Shape) ![]u8 {
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("../fixtures/config.json"), .{});
    _ = root.object.orderedRemove("quantization");
    const text = &root.object.getPtr("text_config").?.object;
    try text.put(a, "num_hidden_layers", .{ .integer = @intCast(s.layers) });
    var kinds = std.json.Array.init(a);
    for (0..s.layers) |i| try kinds.append(.{ .string = if (linear(s, i)) "linear_attention" else "full_attention" });
    try text.put(a, "layer_types", .{ .array = kinds });
    try text.put(a, "num_experts", .{ .integer = @intCast(s.experts) });
    if (s.experts < 10) try text.put(a, "num_experts_per_tok", .{ .integer = 2 });
    try text.put(a, "split_ngram_parts", .{ .integer = @intCast(s.shards) });
    var layers: std.json.ObjectMap = .empty;
    var buf: [160]u8 = undefined;
    for (0..s.layers) |i| {
        try put(a, &layers, try a.dupe(u8, try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts", .{i})), "NVFP4", 16, true);
        const parts: []const []const u8 = if (linear(s, i))
            &.{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj" }
        else
            &.{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.indexer.index_qk_proj" };
        for (parts) |p| try put(a, &layers, try a.dupe(u8, try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.{s}", .{ i, p })), "MXFP8", 32, false);
    }
    try put(a, &layers, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding", "NVFP4", 16, false);
    try put(a, &layers, "mtp.layers.0.mlp.experts", "W4A16_NVFP4", 16, false);
    try put(a, &layers, try a.dupe(u8, try std.fmt.bufPrint(&buf, "mtp.layers.{d}.mlp.experts", .{s.layers})), "W4A16_NVFP4", 16, false);
    try put(a, &layers, "model.visual.blocks.0.attn.qkv", "MXFP8", 32, false);
    var q: std.json.ObjectMap = .empty;
    try q.put(a, "quant_method", .{ .string = "modelopt" });
    try q.put(a, "quant_algo", .{ .string = "MIXED_PRECISION" });
    try q.put(a, "kv_cache_quant_algo", .null);
    try q.put(a, "quantized_layers", .{ .object = layers });
    try root.object.put(a, "quantization_config", .{ .object = q });
    return std.json.Stringify.valueAlloc(a, root, .{});
}

/// The checkpoint's tensor names as the published FP4-Local headers spell them, for `s`.
fn indexText(a: std.mem.Allocator, s: Shape) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"weight_map\": {\"model.visual.blocks.0.attn.qkv.weight\": \"tensors/model-00001-of-00002.safetensors\"");
    const hc = [_][]const u8{ "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight" };
    const top = [_][]const u8{ "lm_head.weight", "model.language_model.embed_tokens.weight" };
    for (top) |n| try w.print(",\"{s}\":\"x.safetensors\"", .{n});
    for (hc) |n| try w.print(",\"model.language_model.hyper_connection_mixer.{s}\":\"x.safetensors\"", .{n});
    for (0..s.layers) |i| try layerNames(w, s, "model.language_model.layers", i, false);
    for ([_][]const u8{ "fc_embedding.weight", "fc_hidden.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight" }) |n| try w.print(",\"mtp.{s}\":\"x.safetensors\"", .{n});
    for (hc) |n| try w.print(",\"mtp.hyper_connection_mixer.{s}\":\"x.safetensors\"", .{n});
    try layerNames(w, s, "mtp.layers", 0, true);
    try w.writeAll("}}");
    return out.written();
}

fn layerNames(w: *std.Io.Writer, s: Shape, root: []const u8, i: usize, mtp: bool) !void {
    const scaled = [_][]const u8{ ".weight", ".weight_scale" };
    const plain = [_][]const u8{".weight"};
    const dense: []const []const u8 = if (mtp) &plain else &scaled;
    for ([_][]const u8{ "attn_hyper_connection", "mlp_hyper_connection" }) |h|
        for ([_][]const u8{ "block_inject_weight.weight", "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight" }) |n|
            try w.print(",\"{s}.{d}.{s}.{s}\":\"x.safetensors\"", .{ root, i, h, n });
    for ([_][]const u8{ "mlp.gate.weight", "mlp.shared_expert_gate.weight" }) |n| try w.print(",\"{s}.{d}.{s}\":\"x.safetensors\"", .{ root, i, n });
    for ([_][]const u8{ "down_proj", "gate_proj", "up_proj" }) |p| for (dense) |x| try w.print(",\"{s}.{d}.mlp.shared_expert.{s}{s}\":\"x.safetensors\"", .{ root, i, p, x });
    const expert: []const []const u8 = if (mtp) &.{ "weight", "weight_scale", "weight_scale_2" } else &.{ "input_scale", "weight", "weight_scale.nvfp4_csf_exceptions", "weight_scale.nvfp4_csf_fixed", "weight_scale_2" };
    for (0..s.experts) |e| for ([_][]const u8{ "down_proj", "gate_proj", "up_proj" }) |p| for (expert) |x|
        try w.print(",\"{s}.{d}.mlp.experts.{d}.{s}.{s}\":\"x.safetensors\"", .{ root, i, e, p, x });
    if (mtp or !linear(s, i)) {
        for ([_][]const u8{ "indexer.index_qk_proj", "k_proj", "o_proj", "q_proj", "v_proj" }) |p| for (dense) |x| try w.print(",\"{s}.{d}.self_attn.{s}{s}\":\"x.safetensors\"", .{ root, i, p, x });
        for ([_][]const u8{ "indexer.k_layernorm.weight", "indexer.q_layernorm.weight", "k_norm.weight", "q_norm.weight" }) |n| try w.print(",\"{s}.{d}.self_attn.{s}\":\"x.safetensors\"", .{ root, i, n });
    } else {
        for ([_][]const u8{ "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj" }) |p| for (dense) |x| try w.print(",\"{s}.{d}.linear_attn.{s}{s}\":\"x.safetensors\"", .{ root, i, p, x });
        for ([_][]const u8{ "A_log", "conv1d.weight", "dt_bias", "norm.weight" }) |n| try w.print(",\"{s}.{d}.linear_attn.{s}\":\"x.safetensors\"", .{ root, i, n });
    }
    if (mtp or i != 1) return;
    for ([_][]const u8{ "conv1d.weight", "key_proj.weight", "norm_conv.weight", "norm_key.weight", "norm_query.weight", "value_proj.weight", "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes", "ple_embedding.ngram_embedding.weight_scale_2" }) |n|
        try w.print(",\"{s}.{d}.ple.{s}\":\"x.safetensors\"", .{ root, i, n });
    for (0..s.shards) |k| for ([_][]const u8{ "weight", "weight_scale" }) |x| try w.print(",\"{s}.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.{s}\":\"x.safetensors\"", .{ root, i, k, x });
}

fn parsed(a: std.mem.Allocator, s: Shape) !config.Config {
    return config.parseAny(std.testing.allocator, try configText(a, s));
}

test "FP4-Local config: parseAny reads ModelOpt, parse still refuses it, the MLX fixture reads alike through both" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const text = try configText(arena.allocator(), .{});
    try std.testing.expectError(error.UnsupportedAffineMode, config.parse(std.testing.allocator, text));
    var c = try config.parseAny(std.testing.allocator, text);
    defer c.deinit();
    try std.testing.expect(c.format == .modelopt);
    try std.testing.expectError(error.NotAffineCheckpoint, c.affineSpec());
    try std.testing.expectEqual(@as(?[]const u8, null), admit.mismatch(&c));
    var mlx = try config.parseAny(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer mlx.deinit();
    var plain = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer plain.deinit();
    try std.testing.expectEqual(plain.format.mlx, mlx.format.mlx);
    var linear_layers: [config.max_layers]bool = undefined;
    const s = admit.served(&c, &linear_layers).?;
    try std.testing.expectEqual(@as(usize, 1), s.ple_layer);
    try std.testing.expectEqual(@as(f64, 0.25), s.partial_rotary);
    try std.testing.expect(s.linear[0] and !s.linear[3] and s.linear.len == 48);
}

fn expectMismatch(field: []const u8, from: []const u8, to: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.mem.replaceOwned(u8, a, try configText(a, .{}), from, to);
    var c = try config.parseAny(std.testing.allocator, text);
    defer c.deinit();
    const got = admit.mismatch(&c) orelse return error.TestExpectedMismatch;
    try std.testing.expectEqualStrings(field, got);
}

test "FP4-Local config: a wrong recipe, encoding, KV format, setting or shape is refused by its field" {
    const experts = "\"model.language_model.layers.5.mlp.experts\":{\"quant_algo\":\"NVFP4\"";
    try expectMismatch("quantization_config.routed_experts", experts, "\"model.language_model.layers.5.mlp.experts\":{\"quant_algo\":\"MXFP8\"");
    try expectMismatch("quantization_config.weight_scale_encoding", "\"csf\"", "\"csf0_v1\"");
    try expectMismatch("quantization_config.kv_cache_quant_algo", "\"kv_cache_quant_algo\":null", "\"kv_cache_quant_algo\":\"FP8\"");
    try expectMismatch("quantization_config.quant_algo", "\"MIXED_PRECISION\"", "\"NVFP4\"");
    try expectMismatch("hidden_act", "\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"");
    try expectMismatch("moe_intermediate_size", "\"moe_intermediate_size\":640", "\"moe_intermediate_size\":648");
    try expectMismatch("shared_expert_intermediate_size", "\"shared_expert_intermediate_size\":640", "\"shared_expert_intermediate_size\":704");
}

const gib: f64 = 1 << 30;

test "FP4-Local admission: every device byte of the published headers, CSF scales decoded, the n-gram table apart" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var c = try parsed(arena.allocator(), .{});
    defer c.deinit();
    const one = try plan.weightsOf(&c, plan.fp4_local, 1);
    // b9ce18a6's headers less the vision tower and input scales, each CSF plane pair as n x k / 16 scale bytes
    try std.testing.expectEqual(@as(u64, 76_595_701_272), one.device(0));
    try std.testing.expectEqual(@as(u64, 28_800_138_244), one.host);
    for ([_]u32{ 2, 4 }) |world| {
        const w = try plan.weightsOf(&c, plan.fp4_local, world);
        try std.testing.expectEqual(one.host, w.host);
        try std.testing.expect(w.total(world) >= one.total(1));
    }
    // one routed projection: codes, decoded e4m3 scales and its weight_scale_2
    try std.testing.expectEqual(@as(u64, 640 * 2560 / 2 + 640 * 160 + 4), plan.Format.bytes(.nvfp4, 640, 2560));
    try std.testing.expectEqual(@as(u64, 640 * 2560 + 640 * 80), plan.Format.bytes(.mxfp8, 640, 2560));
}

test "FP4-Local index: every tensor is taken or ignored by its reason; a missing, extra or misdeclared one is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s: Shape = .{ .layers = 4, .experts = 4, .shards = 2 };
    var c = try parsed(a, s);
    defer c.deinit();
    const idx = try indexText(a, s);
    var why: core.modelopt.Why = .{};
    const inv = try admit.admitIndex(std.testing.allocator, idx, &c, &why);
    try std.testing.expectEqual(@as(usize, 1), inv.vision);
    try std.testing.expectEqual(@as(usize, 4 * 4 * 3), inv.input_scales);
    try std.testing.expectEqual(@as(usize, 1), inv.empty_modules);
    const missing = try std.mem.replaceOwned(u8, a, idx, ",\"model.language_model.layers.2.mlp.experts.3.up_proj.weight_scale_2\":\"x.safetensors\"", "");
    try std.testing.expectError(error.MissingFlashTensor, admit.admitIndex(std.testing.allocator, missing, &c, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "layers.2.mlp.experts.3.up_proj.weight_scale_2") != null);
    const extra = try std.mem.replaceOwned(u8, a, idx, "}}", ",\"model.language_model.norm.weight\":\"x.safetensors\"}}");
    try std.testing.expectError(error.UnexpectedFlashTensor, admit.admitIndex(std.testing.allocator, extra, &c, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "model.language_model.norm.weight") != null);
    const unsafe = try std.mem.replaceOwned(u8, a, idx, "\"tensors/model-00001-of-00002.safetensors\"", "\"../x.safetensors\"");
    try std.testing.expectError(error.UnsafeFlashShard, admit.admitIndex(std.testing.allocator, unsafe, &c, &why));
    const open = "\"quantized_layers\":{";
    const recipes = [_]struct { []const u8, []const u8, []const u8 }{
        .{ "\"mtp.layers.0.mlp.experts\":{\"quant_algo\":\"W4A16_NVFP4\"", "\"mtp.layers.0.mlp.experts\":{\"quant_algo\":\"NVFP4\"", "mtp.layers.0.mlp.experts" },
        .{ open, open ++ "\"model.language_model.layers.0.mlp.gate\":{\"quant_algo\":\"MXFP8\",\"group_size\":32},", "layers.0.mlp.gate" },
        .{ open, open ++ "\"model.language_model.layers.0.linear_attn.norm\":{\"quant_algo\":\"MXFP8\",\"group_size\":32},", "layers.0.linear_attn.norm" },
    };
    for (recipes) |r| {
        var bad = try config.parseAny(std.testing.allocator, try std.mem.replaceOwned(u8, a, try configText(a, s), r[0], r[1]));
        defer bad.deinit();
        try std.testing.expectError(error.UnsupportedQuantization, admit.admitIndex(std.testing.allocator, idx, &bad, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), r[2]) != null);
    }
}

/// The real checkpoint's device and host bytes from its safetensors headers alone (no tensor data is read).
const Headers = struct { routed: u64 = 0, mtp_routed: u64 = 0, mxfp8: u64 = 0, plain: u64 = 0, host: u64 = 0, ignored: u64 = 0 };

fn headerBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, map: std.json.ObjectMap) !Headers {
    var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer shards.deinit(gpa);
    for (map.values()) |v| try shards.put(gpa, v.string, {});
    var h: Headers = .{};
    for (shards.keys()) |shard| {
        const path = try std.fs.path.join(gpa, &.{ dir, shard });
        defer gpa.free(path);
        var md = try core.safetensors.Metadata.read(gpa, io, path);
        defer md.deinit();
        var it = md.names.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            const t = e.value_ptr.*;
            const n: u64 = t.end - t.begin;
            if (std.mem.startsWith(u8, name, "model.visual.") or std.mem.endsWith(u8, name, ".input_scale")) {
                h.ignored += n;
            } else if (std.mem.indexOf(u8, name, ".ngram_embedding.") != null) {
                h.host += n;
            } else if (std.mem.indexOf(u8, name, ".weight_scale.nvfp4_csf_") != null) {} else if (std.mem.startsWith(u8, name, "model.") and std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
                h.routed += n;
                if (std.mem.endsWith(u8, name, ".weight")) h.routed += t.dim(0) * (t.dim(1) * 2 / 16);
            } else if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
                h.mtp_routed += n;
            } else if (t.dtype == .f8_e4m3 or (t.dtype == .u8 and std.mem.endsWith(u8, name, ".weight_scale"))) {
                h.mxfp8 += n;
            } else h.plain += n;
        }
    }
    return h;
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

test "FP4-Local on the real checkpoint (TF_FLASHNEXT_FP4LOCAL_DIR): config, every index tensor, admitted bytes" {
    const dir = @import("sys").getenv("TF_FLASHNEXT_FP4LOCAL_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var c = try config.Config.readAny(gpa, io, dir);
    defer c.deinit();
    if (admit.mismatch(&c)) |field| {
        std.debug.print("FP4-Local config.json: {s} refused\n", .{field});
        return error.TestUnexpectedResult;
    }
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    var why: core.modelopt.Why = .{};
    const inv = admit.admitIndex(gpa, text, &c, &why) catch |err| {
        std.debug.print("FP4-Local index: {s}\n", .{why.text()});
        return err;
    };
    const idx = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer idx.deinit();
    const map = idx.value.object.get("weight_map").?.object;
    try std.testing.expectEqual(map.count(), inv.taken + inv.vision + inv.input_scales);
    std.debug.print("FP4-Local index: {d} tensors taken, {d} ignored ({s}), {d} ignored ({s}); {d} declared modules hold no tensor\n", .{ inv.taken, inv.vision, admit.ignored[0].reason, inv.input_scales, admit.ignored[1].reason, inv.empty_modules });
    const h = try headerBytes(gpa, io, dir, map);
    const w = try plan.weightsOf(&c, plan.fp4_local, 1);
    std.debug.print("FP4-Local headers: NVFP4 experts {d:.2} GiB, MTP experts {d:.2}, MXFP8 {d:.2}, bf16/f32 {d:.2}; " ++
        "device {d:.2} GiB (replicated {d:.2}, split {d:.2}, shared {d:.2}), n-gram table {d:.2} GiB on disk, {d:.2} GiB ignored\n", .{ gibs(h.routed), gibs(h.mtp_routed), gibs(h.mxfp8), gibs(h.plain), gibs(w.device(0)), gibs(w.replicated), gibs(w.split), gibs(w.shared), gibs(h.host), gibs(h.ignored) });
    try std.testing.expectEqual(h.routed + h.mtp_routed + h.mxfp8 + h.plain, w.device(0));
    try std.testing.expectEqual(h.host, w.host);
}
