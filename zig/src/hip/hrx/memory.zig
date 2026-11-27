//! HIP allocations, copies and fills on HRX buffers; HIP's u64 device addresses resolve through the driver's table.
const abi = @import("../abi.zig");
const c = @import("c.zig");
const drv = @import("driver.zig");
const streams = @import("streams.zig");

const device_params: c.BufferParams = .{ .type = c.memory_device_local, .access = c.access_all, .usage = c.usage_default };
const host_params: c.BufferParams = .{
    .type = c.memory_host_local | c.memory_device_visible,
    .access = c.access_all,
    .usage = c.usage_default | c.usage_mapping_persistent,
};

const Span = struct { buffer: c.Buffer, offset: usize };

fn span(p: abi.DevicePtr, n: usize) ?Span {
    const hit = drv.d().table.find(@intFromPtr(p), n) orelse return null;
    return .{ .buffer = hit.entry.buffer, .offset = @intCast(hit.offset) };
}

fn allocate(params: c.BufferParams, size: usize, host: bool) !struct { c.Buffer, u64, ?[*]u8 } {
    const dv = drv.d();
    const f = &dv.f;
    var b: ?c.Buffer = null;
    if (drv.hip(f.hrx_allocator_allocate_buffer(f.hrx_device_allocator(dv.device), params, size, &b)) != drv.success) return error.OutOfMemory;
    errdefer f.hrx_buffer_release(b.?);
    var address: u64 = 0;
    if (drv.hip(f.hrx_buffer_query_address(b.?, dv.scope, c.interface_device_address, &address)) != drv.success) return error.HipFailed;
    var mapped: ?*anyopaque = null;
    if (host and drv.hip(f.hrx_buffer_map(b.?, c.map_read_write, 0, size, &mapped)) != drv.success) return error.HipFailed;
    const h: ?[*]u8 = if (mapped) |m| @ptrCast(m) else null;
    try dv.table.insert(drv.gpa, .{ .base = address, .len = size, .buffer = b.?, .host = h });
    return .{ b.?, address, h };
}

pub fn hipMalloc(out: *abi.DevicePtr, size: usize) callconv(.c) abi.Result {
    out.* = null;
    if (size == 0) return drv.success;
    _, const address, _ = allocate(device_params, size, false) catch return drv.out_of_memory;
    out.* = @ptrFromInt(address);
    return drv.success;
}

/// Like HIP, a free waits for the device first: queued work may still read the buffer.
fn release(base: u64) abi.Result {
    const dv = drv.d();
    const r = streams.hipDeviceSynchronize();
    const e = dv.table.remove(base) orelse return drv.invalid_value;
    if (e.host != null) _ = drv.hip(dv.f.hrx_buffer_unmap(e.buffer));
    dv.f.hrx_buffer_release(e.buffer);
    return r;
}

pub fn hipFree(p: abi.DevicePtr) callconv(.c) abi.Result {
    if (p == null) return drv.success;
    return release(@intFromPtr(p));
}

pub fn hipHostMalloc(out: *abi.DevicePtr, size: usize, flags: c_uint) callconv(.c) abi.Result {
    out.* = null;
    if (size == 0 or flags & ~(abi.host_malloc_portable | abi.host_malloc_mapped) != 0) return drv.invalid_value;
    _, _, const host = allocate(host_params, size, true) catch return drv.out_of_memory;
    out.* = @ptrCast(host);
    return drv.success;
}

pub fn hipHostFree(p: abi.DevicePtr) callconv(.c) abi.Result {
    if (p == null) return drv.success;
    const e = drv.d().table.findHost(@intFromPtr(p)) orelse return drv.invalid_value;
    return release(e.base);
}

pub fn hipHostGetDevicePointer(out: *abi.DevicePtr, host: ?*anyopaque, flags: c_uint) callconv(.c) abi.Result {
    if (flags != 0) return drv.invalid_value;
    const e = drv.d().table.findHost(@intFromPtr(host)) orelse return drv.invalid_value;
    out.* = @ptrFromInt(e.base);
    return drv.success;
}

fn sync(s: *streams.Stream, r: abi.Result) abi.Result {
    return if (r != drv.success) r else s.wait();
}

fn direct(s: *streams.Stream, src: Span, dst: Span, n: usize) abi.Result {
    const t = s.step();
    const waits = t.waits();
    const signals = t.signals();
    return t.finish(drv.d().f.hrx_queue_copy_linear(s.queue, &waits, &signals, src.buffer, src.offset, dst.buffer, dst.offset, n, null));
}

/// A pinned host range (in a hipHostMalloc mapping) copies buffer to buffer; other host memory goes via HRX staging.
fn pinned(host: usize, n: usize) ?Span {
    const hit = drv.d().table.findMapped(host, n) orelse return null;
    return .{ .buffer = hit.entry.buffer, .offset = @intCast(hit.offset) };
}

fn h2d(dst: abi.DevicePtr, src: [*]const u8, n: usize, s: *streams.Stream) abi.Result {
    if (n == 0) return drv.success;
    if (s.capture != null) return drv.not_supported;
    const d = span(dst, n) orelse return drv.invalid_value;
    if (pinned(@intFromPtr(src), n)) |h| return direct(s, h, d, n);
    return s.flush(drv.d().f.hrx_stream_copy_h2d_linear(s.h, src, d.buffer, d.offset, n));
}

fn d2h(dst: [*]u8, src: abi.DevicePtr, n: usize, s: *streams.Stream) abi.Result {
    if (n == 0) return drv.success;
    if (s.capture != null) return drv.not_supported;
    const x = span(src, n) orelse return drv.invalid_value;
    if (pinned(@intFromPtr(dst), n)) |h| return direct(s, x, h, n);
    return s.flush(drv.d().f.hrx_stream_copy_d2h_linear(s.h, x.buffer, x.offset, dst, n));
}

fn d2d(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, s: *streams.Stream) abi.Result {
    if (n == 0) return drv.success;
    const a = span(src, n) orelse return drv.invalid_value;
    const b = span(dst, n) orelse return drv.invalid_value;
    if (s.capture) |cap| return cap.copy(a.buffer, a.offset, b.buffer, b.offset, n);
    return direct(s, a, b, n);
}

fn fill(dst: abi.DevicePtr, pattern: u32, width: usize, n: usize, s: *streams.Stream) abi.Result {
    if (n == 0) return drv.success;
    const b = span(dst, n) orelse return drv.invalid_value;
    if (s.capture) |cap| return cap.fill(b.buffer, b.offset, n, pattern, width);
    return s.flush(drv.d().f.hrx_stream_fill_buffer_linear(s.h, b.buffer, b.offset, n, &pattern, width));
}

pub fn hipMemcpyHtoD(dst: abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) abi.Result {
    const s = streams.of(null);
    return sync(s, h2d(dst, src, n, s));
}

pub fn hipMemcpyDtoH(dst: [*]u8, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    const s = streams.of(null);
    return sync(s, d2h(dst, src, n, s));
}

pub fn hipMemcpyDtoD(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    const s = streams.of(null);
    return sync(s, d2d(dst, src, n, s));
}

pub fn hipMemcpyHtoDAsync(dst: abi.DevicePtr, src: [*]const u8, n: usize, h: abi.Stream) callconv(.c) abi.Result {
    return h2d(dst, src, n, streams.of(h));
}

pub fn hipMemcpyDtoHAsync(dst: [*]u8, src: abi.DevicePtr, n: usize, h: abi.Stream) callconv(.c) abi.Result {
    return d2h(dst, src, n, streams.of(h));
}

pub fn hipMemcpyDtoDAsync(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, h: abi.Stream) callconv(.c) abi.Result {
    return d2d(dst, src, n, streams.of(h));
}

pub fn hipMemset(dst: abi.DevicePtr, value: c_int, n: usize) callconv(.c) abi.Result {
    return fill(dst, @as(u8, @truncate(@as(c_uint, @bitCast(value)))), 1, n, streams.of(null));
}

pub fn hipMemsetAsync(dst: abi.DevicePtr, value: c_int, n: usize, h: abi.Stream) callconv(.c) abi.Result {
    return fill(dst, @as(u8, @truncate(@as(c_uint, @bitCast(value)))), 1, n, streams.of(h));
}

pub fn hipMemsetD32(dst: abi.DevicePtr, value: c_int, count: usize) callconv(.c) abi.Result {
    return fill(dst, @bitCast(value), 4, count * 4, streams.of(null));
}

pub fn hipMemsetD32Async(dst: abi.DevicePtr, value: c_int, count: usize, h: abi.Stream) callconv(.c) abi.Result {
    return fill(dst, @bitCast(value), 4, count * 4, streams.of(h));
}

pub fn hipMemGetInfo(free: *usize, total: *usize) callconv(.c) abi.Result {
    return drv.memory(free, total);
}

/// A module global's storage joins the table so HIP copies reach it by address; the module owns the buffer.
pub fn adopt(b: c.Buffer, len: usize) !u64 {
    const dv = drv.d();
    var address: u64 = 0;
    if (drv.hip(dv.f.hrx_buffer_query_address(b, dv.scope, c.interface_device_address, &address)) != drv.success) return error.HipFailed;
    try dv.table.insert(drv.gpa, .{ .base = address, .len = len, .buffer = b, .held = false });
    return address;
}

pub fn disown(base: u64) void {
    _ = drv.d().table.remove(base);
}
