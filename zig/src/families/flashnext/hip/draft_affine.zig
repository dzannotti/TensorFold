//! The draft head's bf16 rows as 4-bit affine groups of 32 (TF_FLASHNEXT_DRAFT_HEAD=affine4): drafts only.
const std = @import("std");
const core = @import("core");
const row = @import("../cuda_draft_affine.zig").row;

const quant = core.quant;
const view = quant.types.view;

/// How the draft rows are held: TF_FLASHNEXT_DRAFT_HEAD as on CUDA, unset or blank affine4, bf16 the rows as stored.
pub const Mode = enum { bf16, affine4 };

pub fn mode() !Mode {
    return @fromBackingInt(@intCast(try @import("sys").environment.read(u1, "TF_FLASHNEXT_DRAFT_HEAD", @backingInt(Mode.affine4))));
}

/// `h`'s bf16 rows [n, k] as MLX affine words, bf16 scales and biases; the verify head keeps its own rows.
pub fn affine4(a: std.mem.Allocator, h: quant.Host) !quant.Host {
    const w = switch (h) {
        .dense => |d| d.weight,
        else => return error.UnexpectedTensor,
    };
    if (w.dtype != .bf16 or w.rank != 2 or w.shape[1] % 32 != 0) return error.UnexpectedTensor;
    const n = w.shape[0];
    const k = w.shape[1];
    const words = try a.alloc(u8, n * k / 2);
    const scales = try a.alloc(u8, n * k / 16);
    const biases = try a.alloc(u8, n * k / 16);
    for (0..n) |i| try row(w.bytes[i * k * 2 ..][0 .. k * 2], words[i * k / 2 ..][0 .. k / 2], scales[i * k / 16 ..][0 .. k / 16], biases[i * k / 16 ..][0 .. k / 16]);
    return .{ .mlx = .{
        .words = view(.u32, &.{ n, k / 8 }, words),
        .scales = view(.bf16, &.{ n, k / 32 }, scales),
        .biases = view(.bf16, &.{ n, k / 32 }, biases),
        .bits = 4,
        .group = 32,
    } };
}

test "a bf16 row of 0..31 becomes one group: scale 31/15, bias 0, codes rounding to the nearest step" {
    const a = std.testing.allocator;
    var bytes: [64]u8 = undefined;
    const bf16 = @import("../cuda_draft_affine.zig").bf16;
    for (0..32) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bf16(@floatFromInt(i)), .little);
    const h = try affine4(a, .{ .dense = .{ .weight = view(.bf16, &.{ 1, 32 }, &bytes) } });
    const m = h.mlx;
    defer a.free(m.words.bytes);
    defer a.free(m.scales.bytes);
    defer a.free(m.biases.bytes);
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, m.biases.bytes[0..2], .little));
    try std.testing.expectEqual(@as(u8, 0x00), m.words.bytes[0] & 0x0f);
    try std.testing.expectEqual(@as(u8, 0xf0), m.words.bytes[15] & 0xf0);
}
