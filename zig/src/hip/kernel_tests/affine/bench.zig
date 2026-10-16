//! Decode product timings on Qwen3.8-27B's shapes by rows (TF_AFFINE_BENCH=1; TF_AFFINE_BENCH_ENTRY names an entry).

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;

const Shape = struct { name: []const u8, n: usize, k: usize };
const shapes = [_]Shape{
    .{ .name = "qkv", .n = 10240, .k = 5120 },
    .{ .name = "z", .n = 6144, .k = 5120 },
    .{ .name = "a/b", .n = 48, .k = 5120 },
    .{ .name = "out", .n = 5120, .k = 6144 },
    .{ .name = "gate|up", .n = 34816, .k = 5120 },
    .{ .name = "down", .n = 5120, .k = 17408 },
    .{ .name = "q", .n = 12288, .k = 5120 },
    .{ .name = "k|v", .n = 1024, .k = 5120 },
    .{ .name = "o", .n = 5120, .k = 6144 },
    .{ .name = "lm_head", .n = 248320, .k = 5120 },
    // the DFlash2 drafter's
    .{ .name = "dft conv", .n = 1280, .k = 5120 },
    .{ .name = "dft qkv", .n = 6144, .k = 5120 },
    .{ .name = "dft o", .n = 5120, .k = 4096 },
    .{ .name = "dft o+256", .n = 5120, .k = 4352 },
    .{ .name = "dft o+512", .n = 5120, .k = 4608 },
    .{ .name = "dft down", .n = 5120, .k = 17408 },
    .{ .name = "dft o 2x", .n = 10240, .k = 4096 },
    .{ .name = "dft gate", .n = 17408, .k = 5120 },
    .{ .name = "dft head", .n = 98592, .k = 5120 },
};
const rows = [_]c_int{ 1, 2, 4, 8, 12, 16, 32, 64, 128 };
const bits = 4;
const group = 64;
/// Weight copies a timed run cycles through: at least this many bytes, past the 32 MB MALL.
const spread = 256 << 20;
const reps = 7;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

test "affine decode timings on the 27B shapes (TF_AFFINE_BENCH=1)" {
    const mode = std.mem.span(std.c.getenv("TF_AFFINE_BENCH") orelse return);
    const quick = std.mem.eql(u8, mode, "quick");
    const want: ?[]const u8 = if (std.c.getenv("TF_AFFINE_BENCH_ENTRY")) |e| std.mem.span(e) else null;
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    const k = &b.on.affine;
    const s = b.stream.handle;
    std.debug.print("\nshape        M   entry                          us      GB/s   TFLOPS\n", .{});
    for (shapes) |sh| {
        const words_row = sh.k * bits / 32;
        const groups = sh.k / group;
        const wbytes = sh.n * words_row * 4;
        const tbytes = sh.n * groups * 2;
        const copies = @max(1, spread / (wbytes + 2 * tbytes));
        var words = try Buffer.alloc(b, copies * wbytes);
        defer words.free();
        var tables = try Buffer.alloc(b, copies * 2 * tbytes);
        defer tables.free();
        var x = try Buffer.alloc(b, 128 * sh.k * 2);
        defer x.free();
        var out = try Buffer.alloc(b, 128 * sh.n * 4);
        defer out.free();
        try words.buf.fill32(0x7b3d1e95);
        try tables.buf.fill32(0x3c003c00);
        try x.buf.fill32(0x3f803f80);
        for (if (quick) &[_]c_int{ 1, 16, 128 } else &rows) |m| {
            var arg: affine.Arg = .{ .x = x.ptr, .words = words.ptr, .scale = .{ .p = tables.ptr, .kind = 1 }, .bias = .{ .p = tables.ptr + tbytes, .kind = 1 }, .out = 0, .out16 = out.ptr, .m = m, .n = @intCast(sh.n), .k = @intCast(sh.k), .bits = bits, .group = group, .fp16 = 0 };
            var shape = affine.Kernels.shapeOf(arg, 1);
            shape.round = true;
            const e = if (want) |id| blk: {
                for (k.reg.entries) |*x_| if (std.mem.eql(u8, x_.id, id)) break :blk x_;
                return error.NoEntry;
            } else k.choose(.project, .decode, shape) orelse return error.NoTile;
            if (!e.roundsAct(shape)) {
                arg.out = out.ptr;
                arg.out16 = 0;
            }
            var times: [reps]f64 = undefined;
            const launches = @max(copies, 8);
            for (0..reps + 1) |r| {
                const t0 = nowNs();
                for (0..launches) |i| {
                    const c = i % copies;
                    arg.words = words.ptr + c * wbytes;
                    arg.scale.p = tables.ptr + c * 2 * tbytes;
                    arg.bias.p = arg.scale.p + tbytes;
                    try e.launch(k, .{ .r = &b.r, .s = s, .arg = arg });
                }
                try b.stream.synchronize();
                if (r > 0) times[r - 1] = @as(f64, @floatFromInt(nowNs() - t0)) / 1e3 / @as(f64, @floatFromInt(launches));
            }
            const us = median(&times);
            const gbs = @as(f64, @floatFromInt(wbytes + 2 * tbytes)) / us / 1e3;
            const tf = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(sh.n * sh.k)) / us / 1e6;
            std.debug.print("{s:<10} {d:>4}   {s:<28} {d:>8.1} {d:>8.1} {d:>8.2}\n", .{ sh.name, @as(u32, @intCast(m)), e.id, us, gbs, tf });
        }
    }
}
