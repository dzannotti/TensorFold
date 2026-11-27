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
/// `sides`: a group's products of one x in one launch (n is their sum), as a DeltaNet or attention layer runs them.
const Case = struct { name: []const u8, fmt: fp4.Format, n: usize, k: usize, kind: Kind = .plain, sides: []const usize = &.{} };
const cases = [_]Case{
    .{ .name = "mxfp8 qkv|z|b|a", .fmt = .mxfp8, .n = 16480, .k = 2560, .sides = &.{ 10240, 6144, 48, 48 } },
    .{ .name = "mxfp8 q|k|v|index", .fmt = .mxfp8, .n = 13952, .k = 2560, .sides = &.{ 12288, 512, 512, 640 } },
    .{ .name = "mxfp8 qkv", .fmt = .mxfp8, .n = 10240, .k = 2560 },
    .{ .name = "mxfp8 z", .fmt = .mxfp8, .n = 6144, .k = 2560 },
    .{ .name = "mxfp8 q", .fmt = .mxfp8, .n = 12288, .k = 2560 },
    .{ .name = "mxfp8 out", .fmt = .mxfp8, .n = 2560, .k = 6144 },
    .{ .name = "mxfp8 index", .fmt = .mxfp8, .n = 640, .k = 2560 },
    .{ .name = "mxfp8 shared gate|up", .fmt = .mxfp8, .n = 1280, .k = 2560 },
    .{ .name = "mxfp8 shared down", .fmt = .mxfp8, .n = 2560, .k = 640 },
    .{ .name = "nvfp4 gate|up routed", .fmt = .nvfp4, .n = 1280, .k = 2560, .kind = .routed },
    .{ .name = "nvfp4 gate|up swiglu", .fmt = .nvfp4, .n = 1280, .k = 2560, .kind = .pair },
    .{ .name = "nvfp4 down routed", .fmt = .nvfp4, .n = 2560, .k = 640, .kind = .routed },
    .{ .name = "bf16 lm_head", .fmt = .bf16, .n = 248320, .k = 2560 },
    .{ .name = "bf16 hc down", .fmt = .bf16, .n = 320, .k = 10240 },
    .{ .name = "bf16 hc down+inject", .fmt = .bf16, .n = 324, .k = 10240 },
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
            const kn = t.on.fp4;
            const pairs = if (routed) m * top_k else m;
            const hx = try gpa.alloc(u16, m * cs.k);
            defer gpa.free(hx);
            for (hx) |*v| v.* = rig.bf16Bits(@floatCast(rng.normal()));
            var x = try t.upload(hx);
            defer x.free();
            var y = try t.alloc(pairs * cs.n * 4);
            defer y.free();
            var base: fp4.Arg = .{ .x = at(x), .words = at(w), .scale = .{ .p = at(sc), .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = at(y), .m = @intCast(m), .n = @intCast(cs.n), .k = @intCast(cs.k), .bits = 0, .group = if (cs.fmt == .bf16) fp4.bf16Fold(@intCast(cs.k)) else 128, .fp16 = 0 };
            if (cs.fmt == .nvfp4) {
                base.global = at(gb);
                base.global_layout = if (cs.n == 1280) .expert_pair else .expert;
            }
            // a routed decode reads a few experts: fresh plans rotate past the caches, as weight copies do for the rest
            var touched: usize = 0;
            const plans: usize = if (!routed) copies else if (prompt) 1 else @max(1, read_target / (m * top_k * one));
            const args = try gpa.alloc(fp4.Arg, plans);
            defer gpa.free(args);
            const its = try gpa.alloc(c_int, plans);
            defer gpa.free(its);
            var bufs: std.ArrayList(@TypeOf(x)) = .empty;
            defer {
                for (bufs.items) |*b| b.free();
                bufs.deinit(gpa);
            }
            for (args, its, 0..) |*a, *it, c| {
                a.* = base;
                it.* = 1;
                if (routed) {
                    const p = try plan(&rng, m);
                    defer {
                        gpa.free(p.items);
                        gpa.free(p.members);
                    }
                    try bufs.append(gpa, try t.upload(p.items));
                    try bufs.append(gpa, try t.upload(p.members));
                    a.route = .{ .items = at(bufs.items[bufs.items.len - 2]), .members = at(bufs.items[bufs.items.len - 1]), .x_div = top_k };
                    a.m = @intCast(p.most);
                    it.* = @intCast(p.count);
                    touched += p.count;
                } else {
                    a.words += c * words_one;
                    a.scale.p += c * scales_one;
                    touched += 1;
                }
                if (cs.kind == .pair) {
                    a.out16 = a.out;
                    a.out = 0;
                }
            }
            const expert_rows: c_int = @intCast((m * top_k + experts - 1) / experts);
            const launch = struct {
                fn go(r: *Rig, k: *const fp4.Kernels, c: Case, arg0: fp4.Arg, it: c_int, er: c_int, pr: bool, tile: bool) !void {
                    var arg = arg0;
                    const s = r.stream.handle;
                    const stream = if (tile) &fp4.Kernels.tileGo else &fp4.Kernels.streamGo;
                    if (pr and c.kind == .pair) {
                        if (!tile and k.experts.takesGateUp(arg, 0)) return k.experts.gateUpGo(r.on.r, arg, it, s);
                        return k.pairGo(r.on.r, c.fmt, arg, it, 0, s);
                    }
                    if (pr) return if (tile) k.tilePromptGo(r.on.r, c.fmt, arg, it, er, s) else k.promptGo(r.on.r, c.fmt, arg, it, er, s);
                    if (c.kind == .pair) return k.streamGo(r.on.r, c.fmt, arg, .{ .pair_cols = @intCast(c.n / 2) }, it, s);
                    if (c.sides.len > 0) {
                        var sides: fp4.StreamSides = .{ .count = @intCast(c.sides.len), .out_half = 1 };
                        var row: usize = 0;
                        for (c.sides, 0..) |n, i| {
                            sides.words[i] = arg.words + row * c.k;
                            sides.scale[i] = arg.scale.p + row * c.k / 32;
                            sides.out[i] = arg.out + row * 2 * @as(usize, @intCast(arg.m));
                            sides.n[i] = @intCast(n);
                            row += n;
                        }
                        arg.n = @intCast(c.sides[0]);
                        return stream(k, r.on.r, c.fmt, arg, sides, it, s);
                    }
                    return stream(k, r.on.r, c.fmt, arg, .{}, it, s);
                }
            }.go;
            // twice where a Loom kernel may serve it: as served, then on the hipcc tiles alone
            const both = (cs.fmt == .mxfp8 and !prompt and kn.gemv.f != null) or (cs.fmt == .nvfp4 and prompt and kn.experts.f != null) or
                (cs.fmt == .bf16 and prompt and kn.dense.f != null);
            for (0..if (both) 2 else 1) |pass| {
                const tile = pass == 1;
                const who: []const u8 = if (!both) "" else if (tile) " [hipcc]" else " [served]";
                for (args, its) |a, it| try launch(t, &kn, cs, a, it, expert_rows, prompt, tile);
                try t.stream.synchronize();
                try start.record(t.stream);
                for (args, its) |a, it| try launch(t, &kn, cs, a, it, expert_rows, prompt, tile);
                try end.record(t.stream);
                try end.synchronize();
                const probe = @as(f64, try Event.elapsedMs(start, end));
                // enough launches for ~0.3 s of power samples a batch
                const reps = @max(1, @as(usize, @intFromFloat(300.0 / @max(probe, 0.001))));
                var us: [batches]f64 = undefined;
                var sampler: Sampler = .{};
                const th = try std.Thread.spawn(.{}, Sampler.loop, .{&sampler});
                for (&us) |*u| {
                    try start.record(t.stream);
                    for (0..reps) |_| for (args, its) |a, it| try launch(t, &kn, cs, a, it, expert_rows, prompt, tile);
                    try end.record(t.stream);
                    try end.synchronize();
                    u.* = @as(f64, try Event.elapsedMs(start, end)) * 1000 / @as(f64, @floatFromInt(plans * reps));
                }
                sampler.stop.store(true, .release);
                th.join();
                std.mem.sort(f64, &us, {}, std.sort.asc(f64));
                const mid = us[batches / 2];
                const watts = if (sampler.count > 0) sampler.sum / @as(f64, @floatFromInt(sampler.count)) else 0;
                // bytes a launch must read: the experts its plan touches (or the matrix), x once, out once
                const out_bytes = pairs * (if (cs.kind == .pair or cs.sides.len > 0) cs.n * 2 else cs.n * 4);
                const bytes = @as(f64, @floatFromInt(touched * one)) / @as(f64, @floatFromInt(plans)) + @as(f64, @floatFromInt(m * cs.k * 2 + out_bytes));
                const flops = 2.0 * @as(f64, @floatFromInt(pairs * cs.n * cs.k));
                const gbs = bytes / mid / 1e3;
                const tfs = flops / mid / 1e6;
                if (prompt) {
                    std.debug.print("FP4L prompt {s:<22} m{d:>5} n{d:>6} k{d:>5}: {d:>9.1} us {d:>6.2} TFLOPS {d:>5.1}% of {d:.1}  {d:>5.1} W {d:>6.1} J/PFLOP (noise {d:.1}%){s}\n", .{
                        cs.name, m, cs.n, cs.k, mid, tfs, tfs / matrix * 100, matrix, watts, watts / tfs * 1e3, (us[batches - 1] / us[0] - 1) * 100, who,
                    });
                } else {
                    std.debug.print("FP4L decode {s:<22} m{d:>5} n{d:>6} k{d:>5}: {d:>9.2} us {d:>6.1} GB/s {d:>5.1}% of {d:.0}  {d:>5.1} W {d:>6.3} J/GB (noise {d:.1}%){s}\n", .{
                        cs.name, m, cs.n, cs.k, mid, gbs, gbs / bandwidth * 100, bandwidth, watts, watts / gbs, (us[batches - 1] / us[0] - 1) * 100, who,
                    });
                }
            }
        }
    };
}
