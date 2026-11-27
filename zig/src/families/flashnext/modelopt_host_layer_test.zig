//! One real FP4-Local layer on the host path: baked experts equal nvfp4_format.bake and unpack to the stored codes.
const std = @import("std");
const core = @import("core");
const mh = @import("modelopt_host.zig");
const nvfp4_read = @import("nvfp4_read.zig");

const quant = core.quant;
const nvfp4_format = quant.nvfp4_format;

/// Experts a chunk of the test bakes at once (the loader's pinned staging takes a like count).
const chunk = 64;

fn expertsRoundTrip(a: std.mem.Allocator, s: anytype, prefix: []const u8, count: usize, width: usize, dims: usize) !void {
    const x = try mh.Experts.read(a, s, prefix, count, width, dims, try nvfp4_read.configured());
    const z = x.sizes();
    const out = try a.alloc(u8, chunk * z.expert());
    var scratch: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer scratch.deinit();
    var lo: usize = 0;
    while (lo < count) : (lo += chunk) {
        const hi = @min(count, lo + chunk);
        const m = hi - lo;
        try x.bake(lo, hi, out);
        const up_w = out[0 .. m * z.up_words];
        const up_s = out[m * z.up_words ..][0 .. m * z.up_scales];
        const dn_w = out[m * (z.up_words + z.up_scales) ..][0 .. m * z.down_words];
        const dn_s = out[m * (z.up_words + z.up_scales + z.down_words) ..][0 .. m * z.down_scales];
        for (lo..hi) |e| {
            _ = scratch.reset(.retain_capacity);
            const b = scratch.allocator();
            const i = e - lo;
            var buf: [512]u8 = undefined;
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |part, j| {
                const name = try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ prefix, e, part });
                const want = try nvfp4_format.bake(b, try core.nvfp4_slice.stored(s, name), name, null);
                const words, const scales = switch (j) {
                    0, 1 => .{ up_w[i * z.up_words + j * z.up_words / 2 ..][0 .. z.up_words / 2], up_s[i * z.up_scales + j * z.up_scales / 2 ..][0 .. z.up_scales / 2] },
                    else => .{ dn_w[i * z.down_words ..][0..z.down_words], dn_s[i * z.down_scales ..][0..z.down_scales] },
                };
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.words), words);
                try std.testing.expectEqualSlices(u8, want.scales, scales);
                var wb: [512]u8 = undefined;
                const stored = (try s.get(try std.fmt.bufPrint(&wb, "{s}.weight", .{name}))).bytes;
                const back = try b.alloc(u8, words.len);
                for (0..words.len / 4) |w| std.mem.writeInt(u32, back[4 * w ..][0..4], nvfp4_format.unpermute(std.mem.readInt(u32, words[4 * w ..][0..4], .little)), .little);
                try std.testing.expectEqualSlices(u8, stored, back);
                try std.testing.expectEqual(want.global, x.views[3 * e + j].global);
            }
        }
    }
}

fn stitchedRoundTrip(a: std.mem.Allocator, s: anytype, prefix: []const u8, parts: []const []const u8) !usize {
    const h = try mh.stitched(a, s, prefix, parts);
    var buf: [512]u8 = undefined;
    var row: usize = 0;
    for (parts) |p| {
        const w = try s.get(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, p }));
        switch (h) {
            .mxfp8 => |m| {
                const k = m.codes.shape[1];
                const e = try s.get(try std.fmt.bufPrint(&buf, "{s}{s}.weight_scale", .{ prefix, p }));
                try std.testing.expectEqualSlices(u8, w.bytes, m.codes.bytes[row * k ..][0..w.bytes.len]);
                try std.testing.expectEqualSlices(u8, e.bytes, m.scales.bytes[row * (k / 32) ..][0..e.bytes.len]);
            },
            .dense => |d| try std.testing.expectEqualSlices(u8, w.bytes, d.weight.bytes[row * d.weight.shape[1] * 2 ..][0..w.bytes.len]),
            else => return error.TestUnexpectedResult,
        }
        row += w.shape[0];
    }
    try std.testing.expectEqual(row, h.rows());
    return h.bytes();
}

test "FP4-Local one layer on the host (TF_FLASHNEXT_FP4LOCAL_DIR): experts, stitched MXFP8 and bf16 round-trip" {
    const dir = @import("sys").getenv("TF_FLASHNEXT_FP4LOCAL_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, std.testing.io, dir);
    defer ck.close();
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = mh.store(&ck);
    const t0 = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    // layer 3 holds sparse attention, layer 0 a DeltaNet; the MTP head's experts are W4A16 with plain scales
    try expertsRoundTrip(a, s, "model.language_model.layers.3.mlp", 512, 640, 2560);
    try expertsRoundTrip(a, s, "mtp.layers.0.mlp", 512, 640, 2560);
    const experts_ns = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - t0;
    var mx: usize = 0;
    mx += try stitchedRoundTrip(a, s, "model.language_model.layers.3.self_attn.", &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" });
    mx += try stitchedRoundTrip(a, s, "model.language_model.layers.0.linear_attn.", &.{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" });
    mx += try stitchedRoundTrip(a, s, "model.language_model.layers.3.mlp.shared_expert.", &.{ "gate_proj", "up_proj" });
    const dense = try stitchedRoundTrip(a, s, "model.language_model.layers.3.attn_hyper_connection.", &.{ "input_mix_weight_down", "block_inject_weight" });
    _ = try stitchedRoundTrip(a, s, "mtp.layers.0.self_attn.", &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" });
    const out = try mh.proj(a, s, "model.language_model.layers.3.self_attn.o_proj");
    try std.testing.expect(out == .mxfp8);
    const r = try mh.router(a, s, "model.language_model.layers.3.mlp.gate", "model.language_model.layers.3.mlp.shared_expert_gate", 512, 2560);
    try std.testing.expectEqual(@as(usize, 513), r.rows16.shape[0]);
    for (0..r.rows16.bytes.len / 2) |i| {
        const h: u32 = std.mem.readInt(u16, r.rows16.bytes[2 * i ..][0..2], .little);
        try std.testing.expectEqual(h << 16, std.mem.readInt(u32, r.rows32.bytes[4 * i ..][0..4], .little));
    }
    std.debug.print("FP4-Local host layer: 2 x 512 experts baked and checked in {d:.1} s; stitched MXFP8 {d:.1} MiB, bf16 hc {d:.1} MiB\n", .{ @as(f64, @floatFromInt(experts_ns)) / 1e9, @as(f64, @floatFromInt(mx)) / (1 << 20), @as(f64, @floatFromInt(dense)) / (1 << 20) });
}
