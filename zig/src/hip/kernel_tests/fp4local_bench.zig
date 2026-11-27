//! hip-fp4local-bench: FP4-Local's products at checkpoint shapes, weights cold: roofline share, power and energy.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const fp4 = @import("../launches/fp4local.zig");
const Event = @import("../stream.zig").Event;

const bandwidth = 241.0; // GB/s, measured read ceiling
const matrix = 55.5; // TFLOPS, measured bf16 WMMA ceiling
const read_target = 768 << 20;
const batches = 7;
const prompt_rows = 4096;
const top_k = 10;
const experts = 512;

const Kind = enum { plain, routed, pair };
const Case = struct { name: []const u8, fmt: fp4.Format, n: usize, k: usize, kind: Kind = .plain };
const cases = [_]Case{
    .{ .name = "mxfp8 qkv", .fmt = .mxfp8, .n = 10240, .k = 2560 },
    .{ .name = "mxfp8 z", .fmt = .mxfp8, .n = 6144, .k = 2560 },
    .{ .name = "mxfp8 q", .fmt = .mxfp8, .n = 12288, .k = 2560 },
    .{ .name = "mxfp8 out", .fmt = .mxfp8, .n = 2560, .k = 6144 },
    .{ .name = "mxfp8 index", .fmt = .mxfp8, .n = 640, .k = 2560 },
    .{ .name = "mxfp8 shared down", .fmt = .mxfp8, .n = 2560, .k = 640 },
    .{ .name = "nvfp4 gate|up routed", .fmt = .nvfp4, .n = 1280, .k = 2560, .kind = .routed },
    .{ .name = "nvfp4 gate|up swiglu", .fmt = .nvfp4, .n = 1280, .k = 2560, .kind = .pair },
    .{ .name = "nvfp4 down routed", .fmt = .nvfp4, .n = 2560, .k = 640, .kind = .routed },
    .{ .name = "bf16 lm_head", .fmt = .bf16, .n = 248320, .k = 2560 },
    .{ .name = "bf16 hc down", .fmt = .bf16, .n = 320, .k = 10240 },
    .{ .name = "bf16 hc up", .fmt = .bf16, .n = 10240, .k = 320 },
    .{ .name = "bf16 router", .fmt = .bf16, .n = 512, .k = 2560 },
};
const decode_rows = [_]usize{ 1, 2, 4, 8 };

/// Weight bytes of one (n, k) projection in its format.
fn weightBytes(fmt: fp4.Format, n: usize, k: usize) usize {
    return switch (fmt) {
        .nvfp4 => n * k / 2 + n * k / 16,
        .mxfp8 => n * k + n * k / 32,
        .bf16 => n * k * 2,
    };
}

/// The GPU's average power in W (hwmon power1_average, µW), or 0 where the file is not there.
fn power() f64 {
    for (0..16) |h| {
        var buf: [96]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&buf, "/sys/class/drm/card0/device/hwmon/hwmon{d}/power1_average", .{h}, 0) catch continue;
        const fd = std.c.open(path, .{});
        if (fd < 0) continue;
        defer _ = std.c.close(fd);
        var tmp: [32]u8 = undefined;
        const got = std.c.read(fd, &tmp, tmp.len);
        if (got <= 0) continue;
        const uw = std.fmt.parseInt(u64, std.mem.trim(u8, tmp[0..@intCast(got)], " \n"), 10) catch continue;
        return @as(f64, @floatFromInt(uw)) / 1e6;
    }
    return 0;
}

const Sampler = struct {
    stop: std.atomic.Value(bool) = .init(false),
    sum: f64 = 0,
    count: usize = 0,

    fn loop(s: *Sampler) void {
        while (!s.stop.load(.acquire)) {
            s.sum += power();
            s.count += 1;
            _ = std.c.nanosleep(&.{ .sec = 0, .nsec = 5 * std.time.ns_per_ms }, null);
        }
    }
};

/// A plan of `pairs` = rows x top_k pairs over the experts: random experts, grouped by expert in items.
const Plan = struct { items: []i32, members: []i32, count: usize, most: usize };

fn plan(rng: *rig.Rng, rows: usize) !Plan {
    const pairs = rows * top_k;
    const expert = try gpa.alloc(u32, pairs);
    defer gpa.free(expert);
    for (0..rows) |r| {
        var picked: [top_k]u32 = undefined;
        for (0..top_k) |s| {
            var e: u32 = undefined;
            while (true) {
                e = @intCast(rng.next() % experts);
                if (std.mem.indexOfScalar(u32, picked[0..s], e) == null) break;
            }
            picked[s] = e;
            expert[r * top_k + s] = e;
        }
    }
    var counts: [experts]usize = @splat(0);
    for (expert) |e| counts[e] += 1;
    var first: [experts]usize = undefined;
    var used: usize = 0;
    var acc: usize = 0;
    var most: usize = 0;
    for (counts, 0..) |c, e| {
        first[e] = acc;
        acc += c;
        if (c > 0) used += 1;
        most = @max(most, c);
    }
    const members = try gpa.alloc(i32, pairs);
    var fill = first;
    for (expert, 0..) |e, p| {
        members[fill[e]] = @intCast(p);
        fill[e] += 1;
    }
    const items = try gpa.alloc(i32, 3 * used);
    var i: usize = 0;
    for (counts, 0..) |c, e| if (c > 0) {
        items[3 * i] = @intCast(e);
        items[3 * i + 1] = @intCast(first[e]);
        items[3 * i + 2] = @intCast(c);
        i += 1;
    };
    return .{ .items = items, .members = members, .count = used, .most = most };
}

test "hip-fp4local-bench: FP4-Local's products at the checkpoint's shapes, roofline share and energy" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.on.fp4.matrix) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xF4C0 };
    var start = try Event.init(&t.r, true);
    defer start.deinit();
    var end = try Event.init(&t.r, true);
    defer end.deinit();
    for (cases) |cs| for ([_]bool{ false, true }) |prompt| {
        if (cs.kind == .pair and prompt) continue;
        const routed = cs.kind != .plain;
        const one = weightBytes(cs.fmt, cs.n, cs.k);
        const stack: usize = if (routed) experts else 1;
        const copies: usize = if (routed) 1 else @max(1, (read_target + one - 1) / one);
        const words_one = switch (cs.fmt) {
            .nvfp4 => cs.n * cs.k / 2,
            .mxfp8 => cs.n * cs.k,
            .bf16 => cs.n * cs.k * 2,
        };
        const scales_one = switch (cs.fmt) {
            .nvfp4 => cs.n * cs.k / 16,
            .mxfp8 => cs.n * cs.k / 32,
            .bf16 => 0,
        };
        const host = try gpa.alloc(u8, (words_one + scales_one) * stack);
        defer gpa.free(host);
        for (host[0 .. words_one * stack]) |*b| b.* = @truncate(rng.next());
        if (cs.fmt == .mxfp8) for (host[0 .. words_one * stack]) |*b| if (b.* & 0x7f == 0x7f) {
            b.* ^= 1;
        };
        if (cs.fmt == .bf16) for (std.mem.bytesAsSlice(u16, host[0 .. words_one * stack])) |*v| {
            v.* = rig.bf16Bits(@floatCast(rng.normal() * 0.03));
        };
        for (host[words_one * stack ..]) |*b| b.* = if (cs.fmt == .nvfp4) @intCast(0x28 + rng.next() % 16) else @intCast(118 + rng.next() % 10);
        var w = try t.alloc(copies * words_one * stack + 64);
        defer w.free();
        var sc = try t.alloc(copies * @max(scales_one, 16) * stack + 64);
        defer sc.free();
        for (0..copies) |c| {
            try w.upload(c * words_one * stack, host[0 .. words_one * stack]);
            if (scales_one > 0) try sc.upload(c * scales_one * stack, host[words_one * stack ..]);
        }
        const globals = try gpa.alloc(f32, 2 * stack);
        defer gpa.free(globals);
        for (globals) |*g| g.* = 0x1p-6;
        var gb = try t.upload(globals);
        defer gb.free();
        const rows_list: []const usize = if (prompt) &.{prompt_rows} else &decode_rows;
        for (rows_list) |m| {
            const pl: ?Plan = if (routed) try plan(&rng, m) else null;
            defer if (pl) |p| {
                gpa.free(p.items);
                gpa.free(p.members);
            };
            const pairs = if (routed) m * top_k else m;
            const hx = try gpa.alloc(u16, m * cs.k);
            defer gpa.free(hx);
            for (hx) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal()));
            var x = try t.upload(hx);
            defer x.free();
            var y = try t.alloc(pairs * cs.n * 4);
            defer y.free();
            var items_b = try t.upload(if (pl) |p| p.items else &[_]i32{0});
            defer items_b.free();
            var members_b = try t.upload(if (pl) |p| p.members else &[_]i32{0});
            defer members_b.free();
            var a: fp4.Arg = .{ .x = at(x), .words = at(w), .scale = .{ .p = at(sc), .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = at(y), .m = @intCast(m), .n = @intCast(cs.n), .k = @intCast(cs.k), .bits = 0, .group = if (cs.fmt == .bf16) fp4.bf16Fold(@intCast(cs.k)) else 128, .fp16 = 0 };
            if (cs.fmt == .nvfp4) {
                a.global = at(gb);
                a.global_layout = if (cs.n == 1280) .expert_pair else .expert;
            }
            var items: c_int = 1;
            if (pl) |p| {
                a.route = .{ .items = at(items_b), .members = at(members_b), .x_div = top_k };
                a.m = @intCast(p.most);
                items = @intCast(p.count);
            }
            const expert_rows: c_int = @intCast((m * top_k + experts - 1) / experts);
            const launch = struct {
                fn go(r: *Rig, c: Case, arg0: fp4.Arg, copy: usize, stride_w: usize, stride_s: usize, its: c_int, er: c_int, pr: bool) !void {
                    var arg = arg0;
                    arg.words += copy * stride_w;
                    arg.scale.p += copy * stride_s;
                    const k = &r.on.fp4;
                    if (pr) return k.promptGo(r.on.r, c.fmt, arg, its, er, r.stream.handle);
                    if (c.kind == .pair) {
                        arg.out16 = arg.out;
                        arg.out = 0;
                        return k.streamGo(r.on.r, c.fmt, arg, .{ .pair_cols = @intCast(c.n / 2) }, its, r.stream.handle);
                    }
                    return k.streamGo(r.on.r, c.fmt, arg, .{}, its, r.stream.handle);
                }
            }.go;
            const sw = words_one * stack;
            const ss = scales_one * stack;
            for (0..copies) |c| try launch(t, cs, a, c, sw, ss, items, expert_rows, prompt);
            try t.stream.synchronize();
            // enough launches for ~0.3 s of power samples a batch
            var reps: usize = 1;
            try start.record(t.stream);
            for (0..copies) |c| try launch(t, cs, a, c, sw, ss, items, expert_rows, prompt);
            try end.record(t.stream);
            try end.synchronize();
            const probe = @as(f64, try Event.elapsedMs(start, end));
            reps = @max(1, @as(usize, @intFromFloat(300.0 / @max(probe, 0.001))));
            var us: [batches]f64 = undefined;
            var sampler: Sampler = .{};
            const th = try std.Thread.spawn(.{}, Sampler.loop, .{&sampler});
            for (&us) |*u| {
                try start.record(t.stream);
                for (0..reps) |_| for (0..copies) |c| try launch(t, cs, a, c, sw, ss, items, expert_rows, prompt);
                try end.record(t.stream);
                try end.synchronize();
                u.* = @as(f64, try Event.elapsedMs(start, end)) * 1000 / @as(f64, @floatFromInt(copies * reps));
            }
            sampler.stop.store(true, .release);
            th.join();
            std.mem.sort(f64, &us, {}, std.sort.asc(f64));
            const mid = us[batches / 2];
            const watts = if (sampler.count > 0) sampler.sum / @as(f64, @floatFromInt(sampler.count)) else 0;
            // bytes a launch must read: the experts the plan touches (or the matrix), x once, out once
            const touched: usize = if (pl) |p| p.count else 1;
            const out_bytes = pairs * (if (cs.kind == .pair) cs.n else cs.n * 2);
            const bytes = touched * one + m * cs.k * 2 + out_bytes;
            const flops = 2.0 * @as(f64, @floatFromInt(pairs * cs.n * cs.k));
            const gbs = @as(f64, @floatFromInt(bytes)) / mid / 1e3;
            const tfs = flops / mid / 1e6;
            if (prompt) {
                std.debug.print("FP4L prompt {s:<22} m{d:>5} n{d:>6} k{d:>5}: {d:>9.1} us {d:>6.2} TFLOPS {d:>5.1}% of {d:.1}  {d:>5.1} W {d:>6.1} J/PFLOP (noise {d:.1}%)\n", .{
                    cs.name, m, cs.n, cs.k, mid, tfs, tfs / matrix * 100, matrix, watts, watts / tfs * 1e3, (us[batches - 1] / us[0] - 1) * 100,
                });
            } else {
                std.debug.print("FP4L decode {s:<22} m{d:>5} n{d:>6} k{d:>5}: {d:>9.2} us {d:>6.1} GB/s {d:>5.1}% of {d:.0}  {d:>5.1} W {d:>6.3} J/GB (noise {d:.1}%)\n", .{
                    cs.name, m, cs.n, cs.k, mid, gbs, gbs / bandwidth * 100, bandwidth, watts, watts / gbs, (us[batches - 1] / us[0] - 1) * 100,
                });
            }
        }
    };
}
