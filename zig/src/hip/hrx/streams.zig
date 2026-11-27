//! HIP streams and events on HRX streams, each on its own hardware queue; every enqueue is flushed at once.
const std = @import("std");
const abi = @import("../abi.zig");
const c = @import("c.zig");
const drv = @import("driver.zig");
const Capture = @import("capture.zig").Capture;

pub const Stream = struct {
    h: c.Stream,
    queue: c.Queue,
    owned: bool,
    capture: ?*Capture = null,

    /// `features` asks for queue features; a cooperative queue serializes with other cooperative queues.
    pub fn create(own_queue: bool, features: u64) !*Stream {
        const dv = drv.d();
        const f = &dv.f;
        var queue = dv.queue;
        var owned = false;
        if (own_queue) {
            var q: ?c.Queue = null;
            const params: c.QueueParams = .{ .features = features };
            if (f.hrx_queue_acquire(dv.device, 0, &params, &q)) |st| {
                f.hrx_status_ignore(st);
            } else {
                queue = q.?;
                owned = true;
            }
        }
        errdefer if (owned) f.hrx_queue_release(queue);
        var h: ?c.Stream = null;
        if (drv.hip(f.hrx_stream_create(queue, 0, &h)) != drv.success) return error.HipFailed;
        const s = try drv.gpa.create(Stream);
        s.* = .{ .h = h.?, .queue = queue, .owned = owned };
        return s;
    }

    pub fn destroy(s: *Stream) void {
        const f = &drv.d().f;
        _ = drv.hip(f.hrx_stream_synchronize(s.h, c.wait_yield));
        f.hrx_stream_release(s.h);
        if (s.owned) f.hrx_queue_release(s.queue);
        drv.gpa.destroy(s);
    }

    /// Submits what the last call recorded, so the device starts it now.
    pub fn flush(s: *Stream, st: c.Status) c_int {
        const r = drv.hip(st);
        if (r != drv.success) return r;
        return drv.hip(drv.d().f.hrx_stream_flush(s.h));
    }

    /// `next` starts after everything submitted to `s` so far.
    pub fn after(next: *Stream, s: *Stream) c_int {
        const f = &drv.d().f;
        var at: c.TimelinePoint = undefined;
        const r = drv.hip(f.hrx_stream_get_timeline_position(s.h, &at));
        if (r != drv.success) return r;
        return drv.hip(f.hrx_stream_wait_on(next.h, at));
    }

    pub fn wait(s: *Stream) c_int {
        return drv.hip(drv.d().f.hrx_stream_synchronize(s.h, c.wait_active));
    }
};

/// The driver's stream for a HIP handle; null is the null stream.
pub fn of(h: abi.Stream) *Stream {
    return if (h) |p| @ptrCast(@alignCast(p)) else drv.d().null_stream;
}

fn handle(s: *Stream) abi.Stream {
    return @ptrCast(s);
}

pub fn hipStreamCreateWithFlags(out: *abi.Stream, flags: c_uint) callconv(.c) abi.Result {
    if (flags > 1) return drv.invalid_value;
    const dv = drv.d();
    const s = Stream.create(dv.acquire_queues, 0) catch return drv.out_of_memory;
    drv.lock(&dv.lock);
    defer dv.lock.unlock();
    dv.streams.append(drv.gpa, s) catch {
        s.destroy();
        return drv.out_of_memory;
    };
    out.* = handle(s);
    return drv.success;
}

pub fn hipStreamDestroy(h: abi.Stream) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const dv = drv.d();
    const s = of(h);
    drv.lock(&dv.lock);
    for (dv.streams.items, 0..) |x, i| if (x == s) {
        _ = dv.streams.swapRemove(i);
        break;
    };
    dv.lock.unlock();
    s.destroy();
    return drv.success;
}

pub fn hipStreamSynchronize(h: abi.Stream) callconv(.c) abi.Result {
    const s = of(h);
    if (s.capture != null) return 900;
    return s.wait();
}

pub fn hipStreamQuery(h: abi.Stream) callconv(.c) abi.Result {
    var done = false;
    const r = drv.hip(drv.d().f.hrx_stream_query(of(h).h, &done));
    if (r != drv.success) return r;
    return if (done) drv.success else drv.not_ready;
}

pub fn hipDeviceSynchronize() callconv(.c) abi.Result {
    const dv = drv.d();
    var r = dv.null_stream.wait();
    drv.lock(&dv.lock);
    defer dv.lock.unlock();
    if (dv.coop) |k| {
        const x = k.wait();
        if (r == drv.success) r = x;
    }
    for (dv.streams.items) |s| {
        const x = s.wait();
        if (r == drv.success) r = x;
    }
    return r;
}

pub fn hipStreamWaitEvent(h: abi.Stream, e: abi.Event, flags: c_uint) callconv(.c) abi.Result {
    if (flags != 0 or e == null) return drv.invalid_value;
    const s = of(h);
    if (s.capture != null) return drv.not_supported;
    return s.flush(drv.d().f.hrx_stream_wait_event(s.h, @ptrCast(e)));
}

pub fn hipEventCreateWithFlags(out: *abi.Event, flags: c_uint) callconv(.c) abi.Result {
    if (flags & ~abi.event_disable_timing != 0) return drv.invalid_value;
    const dv = drv.d();
    var e: ?c.Event = null;
    const r = drv.hip(dv.f.hrx_event_create(dv.device, @intCast(flags & abi.event_disable_timing), &e));
    if (r == drv.success) out.* = @ptrCast(e);
    return r;
}

pub fn hipEventDestroy(e: abi.Event) callconv(.c) abi.Result {
    if (e == null) return drv.invalid_handle;
    drv.d().f.hrx_event_release(@ptrCast(e));
    return drv.success;
}

pub fn hipEventRecord(e: abi.Event, h: abi.Stream) callconv(.c) abi.Result {
    if (e == null) return drv.invalid_handle;
    const s = of(h);
    if (s.capture != null) return drv.not_supported;
    return s.flush(drv.d().f.hrx_event_record(@ptrCast(e), s.h));
}

pub fn hipEventSynchronize(e: abi.Event) callconv(.c) abi.Result {
    if (e == null) return drv.invalid_handle;
    return drv.hip(drv.d().f.hrx_event_synchronize(@ptrCast(e), c.wait_active));
}

pub fn hipEventQuery(e: abi.Event) callconv(.c) abi.Result {
    if (e == null) return drv.invalid_handle;
    var done = false;
    const r = drv.hip(drv.d().f.hrx_event_query(@ptrCast(e), &done));
    if (r != drv.success) return r;
    return if (done) drv.success else drv.not_ready;
}

pub fn hipEventElapsedTime(ms: *f32, a: abi.Event, b: abi.Event) callconv(.c) abi.Result {
    if (a == null or b == null) return drv.invalid_handle;
    return drv.hip(drv.d().f.hrx_event_elapsed_time(@ptrCast(a), @ptrCast(b), ms));
}
