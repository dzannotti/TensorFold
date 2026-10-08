//! Device-level checks the HIP port leans on: named features, occupancy and function attributes, VMM (growth and
//! one chunk mapped as a ring), what the free-memory counts do around a 1 GiB allocation, and streaming-read bandwidth.

const std = @import("std");
const cuda = @import("cuda");
const vmm = @import("flashnext").vmm;
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

pub fn occupancy(gpu: Gpu) !void {
    const f = try gpu.ctx.features();
    std.debug.print("RESULT features: hip {} arch {s} clusters {} pdl {} warp {d} sms {d}\n", .{ f.hip, f.arch(), f.clusters, f.pdl, f.warp, f.sms });
    var probe = try cuda.Module.load(gpu.d, cuda.kernels.probe);
    defer probe.unload();
    const fill = try probe.function("tf_probe_fill");
    for ([_]u32{ 64, 256, 1024 }) |threads| {
        const n = try fill.occupancy(threads, 0);
        try expect(n > 0, "occupancy of {d}-thread blocks is 0", .{threads});
        std.debug.print("RESULT tf_probe_fill: {d} blocks of {d} threads a multiprocessor\n", .{ n, threads });
    }
    const regs = try fill.attribute(.num_regs);
    const max = try fill.attribute(.max_threads_per_block);
    try expect(max >= 256, "max threads a block {d}", .{max});
    const optin = try gpu.ctx.attribute(.max_shared_memory_per_block_optin);
    try fill.allowDynamicShared(@intCast(optin));
    const over = if (fill.allowDynamicShared(@intCast(optin + 1))) |_| false else |_| true;
    try expect(over, "dynamic shared above the opt-in limit is refused", .{});
    check.pass("occupancy and attributes: {d} registers, max {d} threads, dynamic shared up to {d} bytes", .{ regs, max, optin });
}

fn region(gpu: Gpu, base: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = gpu.d, .ptr = base, .len = len };
}

/// A region grown in two steps keeps its address and contents; a ring's one chunk is seen at every granule.
pub fn vmmCheck(gpu: Gpu) !void {
    var v = try vmm.Vmm.init(gpu.ctx);
    defer v.deinit();
    const g = v.granularity;
    var r = try vmm.Region.reserve(&v, 4 * g);
    defer r.deinit(gpu.gpa);
    _ = try r.growTo(gpu.gpa, g);
    const head = try gpu.gpa.alloc(u8, g);
    defer gpu.gpa.free(head);
    for (head, 0..) |*b, i| b.* = @truncate(i *% 31);
    try region(gpu, r.base, g).upload(0, head);
    _ = try r.growTo(gpu.gpa, 3 * g);
    try region(gpu, r.base + g, 2 * g).fill8(0x77, null);
    const back = try gpu.gpa.alloc(u8, g);
    defer gpu.gpa.free(back);
    try region(gpu, r.base, g).download(0, back);
    try check.sameBytes("VMM first chunk after growth", back, head);
    try region(gpu, r.base + 2 * g, g).download(0, back);
    try expect(std.mem.allEqual(u8, back, 0x77), "VMM grown chunk", .{});
    check.pass("VMM: {d} KiB granularity, region grown 1 -> 3 granules in place, contents kept", .{g >> 10});

    var ring = try vmm.Region.reserve(&v, 4 * g);
    defer ring.deinit(gpu.gpa);
    _ = try ring.mapRing(gpu.gpa, g);
    try region(gpu, ring.base, g).upload(0, head);
    for (1..4) |k| {
        try region(gpu, ring.base + k * g, g).download(0, back);
        try check.sameBytes("ring granule aliases the chunk", back, head);
    }
    var word = [1]u32{0xabcdef01};
    try region(gpu, ring.base + 3 * g + 64, 4).upload(0, std.mem.asBytes(&word));
    try region(gpu, ring.base + 64, 4).download(0, std.mem.asBytes(&word));
    try expect(word[0] == 0xabcdef01, "ring write through the last granule seen at the first", .{});
    // a kernel's writes (the fill) through one mapping, read through another
    try region(gpu, ring.base + 2 * g, g).fill32(0x5a5a5a5a, null);
    try region(gpu, ring.base, g).download(0, back);
    try expect(std.mem.allEqual(u8, back, 0x5a), "ring fill through granule 2 seen at granule 0", .{});
    check.pass("VMM ring: one physical chunk mapped at 4 addresses, copies and fills seen through every mapping", .{});
}

/// MemAvailable as cuda_engine.zig reads it for the sequence budget.
fn memAvailable(io: std.Io) !usize {
    var buf: [8192]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf);
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var w = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        return (try std.fmt.parseInt(usize, w.next() orelse return error.NoMemAvailable, 10)) << 10;
    };
    return error.NoMemAvailable;
}

/// The device's free count and the kernel's MemAvailable before, after allocating and touching, and after freeing 1 GiB.
pub fn meminfo(gpu: Gpu) !void {
    const gib: f64 = 1 << 30;
    const report = struct {
        fn line(g: Gpu, what: []const u8) !void {
            const m = try g.ctx.memInfo();
            const avail = try memAvailable(g.io);
            std.debug.print("RESULT {s}: device free {d:.3} GiB of {d:.3}, MemAvailable {d:.3} GiB\n", .{ what, @as(f64, @floatFromInt(m.free)) / gib, @as(f64, @floatFromInt(m.total)) / gib, @as(f64, @floatFromInt(avail)) / gib });
        }
    }.line;
    try report(gpu, "before");
    var b = try cuda.DeviceBuffer.alloc(gpu.d, 1 << 30);
    try report(gpu, "1 GiB allocated");
    try b.fill8(1, null);
    try gpu.ctx.synchronize();
    try report(gpu, "1 GiB written");
    b.free();
    try report(gpu, "freed");
}

/// Best of `reps` streaming reads of a 1 GiB buffer by block size and blocks a multiprocessor.
pub fn bandwidth(gpu: Gpu, reps: usize) !void {
    const bytes: usize = 1 << 30;
    var src = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer src.free();
    try src.fill32(0x01020304, null);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 1 << 20);
    defer out.free();
    var probe = try cuda.Module.load(gpu.d, cuda.kernels.probe);
    defer probe.unload();
    const read = try probe.function("tf_probe_read");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    const sms: u32 = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var best_all: f64 = 0;
    for ([_]u32{ 64, 128, 256, 512, 1024 }) |threads| for ([_]u32{ 4, 16, 64 }) |per| {
        var args: cuda.Args = .{};
        args.add(src.ptr);
        args.add(@as(u64, bytes / 16));
        args.add(out.ptr);
        const cfg: cuda.Config = .{ .grid = .{ .x = sms * per }, .block = .{ .x = threads } };
        try cuda.launch.launch(read, cfg, stream, &args);
        var best: f32 = std.math.inf(f32);
        for (0..reps) |_| {
            try t0.record(stream);
            try cuda.launch.launch(read, cfg, stream, &args);
            try t1.record(stream);
            try t1.synchronize();
            best = @min(best, try t0.elapsedMs(t1));
        }
        const gbs = @as(f64, @floatFromInt(bytes)) / (@as(f64, best) * 1e6);
        best_all = @max(best_all, gbs);
        std.debug.print("RESULT read 1 GiB: {d} threads x {d} blocks: best {d:.3} ms, {d:.1} GB/s\n", .{ threads, sms * per, best, gbs });
    };
    check.pass("streaming read: best {d:.1} GB/s (best of {d}, a shared GPU)", .{ best_all, reps });
}
