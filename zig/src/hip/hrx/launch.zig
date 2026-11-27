//! hipModuleLaunchKernel on HRX: the explicit arguments packed at their metadata offsets, hidden ones left to HRX.
const std = @import("std");
const abi = @import("../abi.zig");
const c = @import("c.zig");
const drv = @import("driver.zig");
const modules = @import("modules.zig");
const streams = @import("streams.zig");
const meta = @import("meta.zig");

pub const max_bytes = 4096;

/// The explicit kernel argument bytes from HIP's array of pointers to each argument value.
pub fn pack(k: meta.Kernel, params: ?[*]?*anyopaque, out: *[max_bytes]u8) ?[]const u8 {
    if (k.explicit > max_bytes) return null;
    const bytes = out[0..k.explicit];
    @memset(bytes, 0);
    if (k.args.len == 0) return bytes;
    const p = params orelse return null;
    for (k.args, 0..) |a, i| {
        const src: [*]const u8 = @ptrCast(p[i] orelse return null);
        @memcpy(bytes[a.offset..][0..a.size], src[0..a.size]);
    }
    return bytes;
}

const acquire = [1]c.MemoryBarrier{.{ .source_scope = 0, .target_scope = c.scope_dispatch_read | c.scope_dispatch_write }};
const before = [1]c.Barrier{.{ .source_stage_mask = c.stage_dispatch | c.stage_transfer, .target_stage_mask = c.stage_dispatch, .memory_barrier_count = 1, .memory_barriers = &acquire }};
const after = [1]c.Barrier{.{ .source_stage_mask = c.stage_dispatch, .target_stage_mask = 0, .flags = c.barrier_release_system }};
/// HIP's binding's barriers for a direct dispatch: acquire what came before, release the writes to the system after.
const direct_barriers: c.QueueBarriers = .{ .before = &.{ .count = 1, .values = &before }, .after = &.{ .count = 1, .values = &after } };

/// Records one kernel on a stream batch; dynamic shared memory and cooperation need `local()` true.
pub fn record(s: c.Stream, f: *const modules.Function, cfg: *const c.DispatchConfig, shared: u32, constants: []const u8, flags: u32) c.Status {
    const dv = drv.d();
    if (shared == 0 and flags & c.dispatch_cooperative == 0)
        return dv.f.hrx_stream_dispatch(s, f.module.exe, f.ordinal, cfg, constants.ptr, constants.len, null, 0, c.dispatch_direct_arguments | flags);
    return dv.x.hrx_stream_dispatch_with_local_memory.?(s, f.module.exe, f.ordinal, cfg, shared, constants.ptr, constants.len, null, 0, c.dispatch_direct_arguments | flags);
}

/// True when libhrx can give a dispatch dynamic shared memory or run it cooperatively (patched libhrx only).
pub fn local() bool {
    return drv.d().x.hrx_stream_dispatch_with_local_memory != null;
}

fn launch(h: abi.Function, grid: [3]c_uint, block: [3]c_uint, shared: c_uint, sh: abi.Stream, params: ?[*]?*anyopaque, flags: u32) abi.Result {
    if (h == null) return drv.invalid_handle;
    const f = modules.function(h);
    var buf: [max_bytes]u8 = undefined;
    const constants = pack(f.k, params, &buf) orelse return drv.invalid_value;
    const cfg: c.DispatchConfig = .{ .workgroup_count = grid, .workgroup_size = block };
    const s = streams.of(sh);
    if (s.capture) |cap| {
        if (flags & c.dispatch_cooperative != 0 or (shared != 0 and !local())) return drv.not_supported;
        return cap.kernel(f, cfg, shared, constants);
    }
    const dv = drv.d();
    if (flags & c.dispatch_cooperative != 0) return cooperative(s, f, cfg, shared, constants);
    if (shared != 0) {
        if (!local()) return drv.not_supported;
        return s.flush(record(s.h, f, &cfg, shared, constants, 0));
    }
    const t = s.step();
    const waits = t.waits();
    const signals = t.signals();
    return t.finish(dv.f.hrx_queue_dispatch(s.queue, &waits, &signals, f.module.exe, f.ordinal, &cfg, constants.ptr, constants.len, null, 0, &direct_barriers, c.dispatch_direct_arguments));
}

/// Cooperative queues serialize with each other, so only cooperative launches go to the one such stream, in order.
fn cooperative(s: *streams.Stream, f: *modules.Function, cfg: c.DispatchConfig, shared: u32, constants: []const u8) abi.Result {
    if (!local()) return drv.not_supported;
    const dv = drv.d();
    drv.lock(&dv.lock);
    defer dv.lock.unlock();
    const k = dv.coop orelse (streams.Stream.create(true, c.queue_cooperative) catch return drv.out_of_memory);
    dv.coop = k;
    var r = k.after(s);
    if (r == drv.success) r = k.flush(record(k.h, f, &cfg, shared, constants, c.dispatch_cooperative));
    if (r == drv.success) r = s.after(k);
    if (r == drv.success) r = s.flush(null);
    return r;
}

pub fn hipModuleLaunchKernel(h: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: abi.Stream, params: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
    if (extra != null) return drv.not_supported;
    return launch(h, .{ gx, gy, gz }, .{ bx, by, bz }, shared, s, params, 0);
}

pub fn hipModuleLaunchCooperativeKernel(h: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: abi.Stream, params: ?[*]?*anyopaque) callconv(.c) abi.Result {
    return launch(h, .{ gx, gy, gz }, .{ bx, by, bz }, shared, s, params, c.dispatch_cooperative);
}

test "arguments land at their metadata offsets and padding stays zero" {
    const args = [_]meta.Arg{ .{ .offset = 0, .size = 4 }, .{ .offset = 8, .size = 8 } };
    const k: meta.Kernel = .{ .args = &args, .explicit = 16 };
    var a: u32 = 0x11223344;
    var b: u64 = 0x5566778899aabbcc;
    var ptrs = [_]?*anyopaque{ &a, &b };
    var buf: [max_bytes]u8 = undefined;
    const got = pack(k, &ptrs, &buf).?;
    try std.testing.expectEqualSlices(u8, &.{ 0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55 }, got);
    try std.testing.expect(pack(k, null, &buf) == null);
}
