//! The process's one HRX device: libhrx loaded once, its runtime, device, queues, address table and kernel facts.
const std = @import("std");
const platform = @import("sys");
const c = @import("c.zig");
const sysfs = @import("sysfs.zig");
const Table = @import("table.zig").Table;
const Stream = @import("streams.zig").Stream;

pub const gpa = std.heap.c_allocator;

pub const success: c_int = 0;
pub const invalid_value: c_int = 1;
pub const out_of_memory: c_int = 2;
pub const not_initialized: c_int = 3;
pub const invalid_handle: c_int = 400;
pub const not_found: c_int = 500;
pub const not_ready: c_int = 600;
pub const not_supported: c_int = 801;
pub const unknown: c_int = 999;

pub const Driver = struct {
    lib: platform.DynLib,
    f: c.Lib,
    x: c.Extra,
    runtime: c.Runtime,
    group: c.Group,
    device: c.Device,
    queue: c.Queue,
    scope: c.Scope,
    facts: sysfs.Facts,
    arch: [64]u8 = @splat(0),
    name: [256]u8 = @splat(0),
    table: Table(c.Buffer) = .{},
    acquire_queues: bool = false,
    null_stream: *Stream = undefined,
    /// The one stream on a cooperative queue, made on first use; cooperative launches run there in stream order.
    coop: ?*Stream = null,
    gtt_base: ?u64 = null,
    lock: std.atomic.Mutex = .unlocked,
    streams: std.ArrayList(*Stream) = .empty,
};

var state: ?Driver = null;
var state_lock: std.atomic.Mutex = .unlocked;

pub fn d() *Driver {
    return &state.?;
}

pub fn lock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.Thread.yield() catch {};
}

/// Opens libhrx and the first amdgpu device once a process; later opens reuse it.
pub fn open(path: []const u8) !void {
    lock(&state_lock);
    defer state_lock.unlock();
    if (state != null) return;
    var lib = platform.DynLib.open(path) catch return error.DriverUnavailable;
    errdefer lib.close();
    const f = try @import("../symbols.zig").resolve(c.Lib, &lib);
    var x: c.Extra = .{};
    inline for (@typeInfo(c.Extra).@"struct".field_names) |name| @field(x, name) = lib.lookup(@typeInfo(@FieldType(c.Extra, name)).optional.child, name);
    const facts = sysfs.gpu() catch return error.DriverUnavailable;
    var rt: ?c.Runtime = null;
    try ok(f, f.hrx_runtime_create(null, f.hrx_host_allocator_system(), &rt));
    var count: usize = 0;
    var infos: ?[*]c.DeviceInfo = null;
    try ok(f, f.hrx_runtime_query_available_devices(rt.?, "amdgpu", f.hrx_host_allocator_system(), &count, &infos));
    defer f.hrx_host_allocator_free(f.hrx_host_allocator_system(), infos);
    if (count == 0) return error.DriverUnavailable;
    const params = [1]c.DeviceParams{.{ .uri = infos.?[0].uri, .completion_wait_flags = c.wait_yield }};
    var group: ?c.Group = null;
    try ok(f, f.hrx_device_group_create(rt.?, 1, &params, &group));
    const dev = f.hrx_device_group_get_device(group.?, 0) orelse return error.DriverUnavailable;
    var scope: c.Scope = .{};
    try ok(f, f.hrx_device_query_queue_family_memory_scope(dev, 0, c.site_program, &scope));
    var family: c.QueueFamilyInfo = undefined;
    try ok(f, f.hrx_device_query_queue_family(dev, 0, &family));
    state = .{
        .lib = lib,
        .f = f,
        .x = x,
        .runtime = rt.?,
        .group = group.?,
        .device = dev,
        .queue = f.hrx_device_get_queue(dev, 0, 0) orelse return error.DriverUnavailable,
        .scope = scope,
        .facts = facts,
        .acquire_queues = family.supports_acquisition,
    };
    const s = &state.?;
    errdefer state = null;
    try ok(f, f.hrx_device_get_property(dev, c.property_architecture, &s.arch, s.arch.len - 1));
    try ok(f, f.hrx_device_get_property(dev, c.property_name, &s.name, s.name.len - 1));
    s.null_stream = try Stream.create(s.acquire_queues, 0);
}

/// An HRX status as HIP's result; the status is consumed.
pub fn hip(st: c.Status) c_int {
    const s = st orelse return success;
    const f = &d().f;
    const code = f.hrx_status_code(s);
    f.hrx_status_ignore(s);
    return switch (code) {
        c.status_out_of_memory => out_of_memory,
        c.status_not_found => not_found,
        c.status_unimplemented => not_supported,
        c.status_deadline, c.status_unavailable => not_ready,
        3, 11 => invalid_value,
        else => unknown,
    };
}

fn report(f: *const c.Lib, s: c.Status) void {
    var msg: ?[*]u8 = null;
    var n: usize = 0;
    const r = f.hrx_status_to_string(s, &msg, &n);
    if (r == null) if (msg) |m| std.debug.print("hrx: {s}\n", .{m[0..n]});
    f.hrx_status_free_message(msg);
    if (r) |x| f.hrx_status_ignore(x);
}

fn ok(f: c.Lib, st: c.Status) !void {
    const s = st orelse return;
    report(&f, s);
    f.hrx_status_ignore(s);
    return error.HipFailed;
}

/// Free bytes: the GTT budget less the GTT in use at the first query and this process's live allocations.
pub fn memory(free: *usize, total: *usize) c_int {
    const s = d();
    var buf: [128]u8 = undefined;
    const gtt_total = sysfs.number(s.facts.pciPath(&buf, "mem_info_gtt_total") catch return unknown) orelse return unknown;
    const used = sysfs.number(s.facts.pciPath(&buf, "mem_info_gtt_used") catch return unknown) orelse return unknown;
    lock(&s.lock);
    const base = s.gtt_base orelse used;
    s.gtt_base = base;
    s.lock.unlock();
    const held = base + s.table.bytes;
    total.* = gtt_total;
    free.* = if (held < gtt_total) gtt_total - held else 0;
    return success;
}
