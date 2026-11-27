//! FP4-Local's routed experts onto the device: baked on threads into two pinned halves, one copied as the other packs.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const mh = @import("../modelopt_host.zig");

const quant = core.quant;
const Projection = quant.Projection;

/// Pinned bytes each half of the staging holds: a load stages at most twice this, a whole number of experts a half.
pub const half_bytes: usize = 128 << 20;

/// The pinned staging and its two copy streams, kept for a whole load.
pub const Stage = struct {
    host: [2]hip.HostBuffer,
    streams: [2]hip.Stream,

    pub fn init(r: *const hip.Runtime) !Stage {
        var host0 = try hip.HostBuffer.alloc(r, half_bytes);
        errdefer host0.free();
        var host1 = try hip.HostBuffer.alloc(r, half_bytes);
        errdefer host1.free();
        var s0 = try hip.Stream.init(r);
        errdefer s0.deinit();
        return .{ .host = .{ host0, host1 }, .streams = .{ s0, try hip.Stream.init(r) } };
    }

    /// Every queued copy done, then the pinned bytes and streams released.
    pub fn deinit(s: *Stage) void {
        for (&s.streams) |*st| {
            st.synchronize() catch {};
            st.deinit();
        }
        for (&s.host) |*h| h.free();
    }
};

/// The stacked experts as the forward reads them: fused gate | up [E][2 width][dims], down [E][dims][width].
pub const Stacked = struct { fused: Projection, down: Projection };

/// `x`'s experts baked a half at a time into device buffers kept in `buffers`, with their globals.
pub fn upload(x: mh.Experts, st: *Stage, up: *const hip.Upload) !Stacked {
    const r = up.r;
    const z = x.sizes();
    const e = x.count;
    const planes = [4]usize{ z.up_words, z.up_scales, z.down_words, z.down_scales };
    var dev: [4]hip.DeviceBuffer = undefined;
    // each buffer joins the model's list at once, so the model frees it on any later failure
    for (&dev, planes) |*d, size| {
        d.* = try hip.DeviceBuffer.alloc(r, e * size);
        up.buffers.append(up.gpa, d.*) catch |err| {
            d.free();
            return err;
        };
    }
    const chunk = @max(1, @min(e, half_bytes / z.expert()));
    if (chunk * z.expert() > half_bytes) return error.UnexpectedTensor;
    var lo: usize = 0;
    var k: usize = 0;
    // a half is packed again only after its stream's copies from two chunks back are done
    while (lo < e) : ({
        lo += chunk;
        k += 1;
    }) {
        const hi = @min(e, lo + chunk);
        const m = hi - lo;
        const h = k % 2;
        try st.streams[h].synchronize();
        try x.bake(lo, hi, st.host[h].bytes);
        var at: usize = 0;
        for (dev, planes) |d, size| {
            try d.uploadAsync(lo * size, st.host[h].view(at, at + m * size), st.streams[h]);
            at += m * size;
        }
    }
    for (st.streams) |s| try s.synchronize();
    const up_globals = try up.gpa.alloc(f32, 2 * e);
    defer up.gpa.free(up_globals);
    const down_globals = try up.gpa.alloc(f32, e);
    defer up.gpa.free(down_globals);
    x.globals(up_globals, down_globals);
    const u = up.uploader();
    const w = x.width;
    const d = x.dims;
    const g = quant.nvfp4_format.group;
    const fused: quant.Device = .{ .nvfp4 = .{
        .words = buf(dev[0], .u32, &.{ e, 2 * w, d / 8 }),
        .scales = buf(dev[1], .u8, &.{ e, 2 * w, d / g }),
        .globals = try u.tensor(quant.types.view(.f32, &.{ e, 2 }, std.mem.sliceAsBytes(up_globals))),
    } };
    const down: quant.Device = .{ .nvfp4 = .{
        .words = buf(dev[2], .u32, &.{ e, d, w / 8 }),
        .scales = buf(dev[3], .u8, &.{ e, d, w / g }),
        .globals = try u.tensor(quant.types.view(.f32, &.{ e, 1 }, std.mem.sliceAsBytes(down_globals))),
    } };
    return .{ .fused = try quant.view(fused, true), .down = try quant.view(down, true) };
}

fn buf(d: hip.DeviceBuffer, dtype: quant.DType, shape: []const usize) quant.Buf {
    var b: quant.Buf = .{ .ptr = d.base(), .len = d.len, .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1) };
    @memcpy(b.shape[0..shape.len], shape);
    return b;
}
