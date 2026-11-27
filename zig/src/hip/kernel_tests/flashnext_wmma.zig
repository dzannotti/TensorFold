//! WMMA sparse attention: the Loom prompt kernels' bytes equal the hipcc decode kernels' at any row count.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const ptr = fx.ptr;

const heads = 24;
const kv_heads = 2;
const dims = 256;
const parts = 16;
const cap = 2300;
const stride = 2052;
const scale: f32 = 0.0625;

test "WMMA sparse attention: Loom's bytes are the hipcc kernel's, whole launches and single rows" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    if (t.on.fnx.sparse_wmma == null or t.on.attn.f == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x3A7A };
    // empty parts, tiles of one key, part and tile seams, listed rows of the full budget and its tail
    const count = [_]i32{ 1, 3, 5, 31, 33, 128, 129, 255, 257, 700, 2048, 2051, 2049, 1300 };
    const sparse = [_]i32{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1 };
    const rows = count.len;
    const hk = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hk);
    const hv = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hv);
    const hq = try gpa.alloc(f32, rows * heads * dims);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit() * 6.0;
    const ids = try gpa.alloc(i32, rows * stride);
    defer gpa.free(ids);
    for (0..rows) |r| for (0..stride) |j| {
        ids[r * stride + j] = @intCast((j * 37 + r * 11 + (j * j) % 13) % cap);
    };
    var dk = try t.upload(hk);
    defer dk.free();
    var dv = try t.upload(hv);
    defer dv.free();
    var dq = try t.upload(hq);
    defer dq.free();
    var di = try t.upload(ids);
    defer di.free();
    var dc = try t.upload(&count);
    defer dc.free();
    var ds = try t.upload(&sparse);
    defer ds.free();
    var po = try t.alloc(rows * heads * parts * dims * 4);
    defer po.free();
    var pm = try t.alloc(rows * heads * parts * 2 * 4);
    defer pm.free();
    var hip_out = try t.alloc(rows * heads * dims * 4);
    defer hip_out.free();
    var loom_out = try t.alloc(rows * heads * dims * 4);
    defer loom_out.free();
    var solo_out = try t.alloc(rows * heads * dims * 4);
    defer solo_out.free();
    const st = t.stream.handle;
    try t.on.fn_sparse_attention(ptr(?[*]const f32, dq), ptr(?*const anyopaque, dk), ptr(?*const anyopaque, dv), 2, ptr(?[*]const i32, di), stride, ptr(?[*]const i32, dc), ptr(?[*]const i32, ds), ptr(?[*]f32, po), ptr(?[*]f32, pm), ptr(?[*]f32, hip_out), rows, heads, kv_heads, dims, cap, scale, parts, st);
    try t.on.attn.attend(&t.r, ptr(?[*]const f32, dq), ptr(?*const anyopaque, dk), ptr(?*const anyopaque, dv), ptr(?[*]const i32, di), stride, ptr(?[*]const i32, dc), ptr(?[*]const i32, ds), ptr(?[*]f32, loom_out), rows, heads, kv_heads, cap, scale, st);
    for (0..rows) |r| {
        const q_at = @intFromPtr(dq.ptr) + r * heads * dims * 4;
        try t.on.attn.attend(&t.r, @ptrFromInt(q_at), ptr(?*const anyopaque, dk), ptr(?*const anyopaque, dv), @ptrFromInt(@intFromPtr(di.ptr) + r * stride * 4), stride, @ptrFromInt(@intFromPtr(dc.ptr) + r * 4), @ptrFromInt(@intFromPtr(ds.ptr) + r * 4), @ptrFromInt(@intFromPtr(solo_out.ptr) + r * heads * dims * 4), 1, heads, kv_heads, cap, scale, st);
    }
    try t.stream.synchronize();
    const want = try rig.download(u32, hip_out);
    defer gpa.free(want);
    for ([_]@TypeOf(loom_out){ loom_out, solo_out }, 0..) |b, w| {
        const got = try rig.download(u32, b);
        defer gpa.free(got);
        for (want, got, 0..) |x, y, i| try rig.expect(x == y, "{s}: row {d} word {d}: hipcc {x} loom {x}", .{
            if (w == 0) "whole launch" else "single rows", i / (heads * dims), i % (heads * dims), x, y,
        });
    }
}
