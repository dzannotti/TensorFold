//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const hip = @import("driver.zig").hip;
const hip_api = @import("hip.zig");

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.cuCtxSynchronize();
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuDeviceGetAttribute(&v, a, self.device), "cuDeviceGetAttribute");
        return v;
    }

    /// What the device can do, named, so callers stop reading features off the compute capability.
    pub const Features = struct {
        hip: bool,
        /// "sm_121" on CUDA, the AMDGPU target ("gfx1151") on HIP
        arch_buf: [32]u8 = @splat(0),
        arch_len: usize = 0,
        /// thread-block clusters and distributed shared memory (CUDA sm_90+; none on AMD)
        clusters: bool,
        /// programmatic dependent launch, griddepcontrol (CUDA sm_90+; none on AMD)
        pdl: bool,
        /// threads a warp (wave32 on RDNA)
        warp: u32,
        /// SMs on CUDA; what HIP reports as multiprocessors on AMD (gfx1151: 20, its WGPs, of 40 CUs)
        sms: u32,

        pub fn arch(f: *const Features) []const u8 {
            return f.arch_buf[0..f.arch_len];
        }
    };

    pub fn features(self: *const Context) Error!Features {
        var f: Features = .{
            .hip = hip,
            .clusters = false,
            .pdl = false,
            .warp = @intCast(try self.attribute(.warp_size)),
            .sms = @intCast(try self.attribute(.multiprocessor_count)),
        };
        if (hip) {
            f.arch_len = hip_api.archName(self.device, &f.arch_buf).len;
        } else {
            const cap = try self.capability();
            f.clusters = cap >= 90;
            f.pdl = cap >= 90;
            f.arch_len = (std.fmt.bufPrint(&f.arch_buf, "sm_{d}", .{cap}) catch unreachable).len;
        }
        return f;
    }

    /// Compute capability as 10 * major + minor (GB10: 121; HIP reports gfx1151 as 115, so use `features`).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};
