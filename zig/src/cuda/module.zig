//! Loaded GPU code: cubin, fatbin, PTX or AMDGPU code-object images, their kernels by symbol name and their device globals.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const hip = @import("driver.zig").hip;
const mangle = @import("mangle.zig");

/// What zig/build/hip.zig embeds for a kernel that hipcc could not build: this, its name and the compiler's errors.
pub const failed_marker = "TF_HIP_BUILD_FAILED ";

pub const Module = struct {
    d: *const Driver,
    handle: abi.Module,

    /// Whether `image` holds GPU code (not empty, not a failed HIP build's marker): callers whose kernel is optional
    /// check this and skip the module instead of failing to load it.
    pub fn built(image: []const u8) bool {
        return image.len > 0 and !std.mem.startsWith(u8, image, failed_marker);
    }

    /// A cubin or fatbin image (8-byte aligned); the driver picks the SASS for this device and copies what it needs.
    pub fn load(d: *const Driver, image: []const u8) Error!Module {
        if (image.len == 0) {
            std.log.err("empty GPU image: this binary was built without kernels (-Dnvcc or -Dfatbins)", .{});
            return error.Invalid;
        }
        if (std.mem.startsWith(u8, image, failed_marker)) {
            std.log.err("GPU module not built for HIP (hipcc failed at build time): {s}", .{image[failed_marker.len..@min(image.len, 2048)]});
            return error.Invalid;
        }
        if (@intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var m: abi.Module = null;
        try d.check(d.api.cuModuleLoadData(&m, image.ptr), "cuModuleLoadData");
        return .{ .d = d, .handle = m };
    }

    /// PTX text JIT-compiled by the driver; its error log is written into `log` on failure.
    pub fn loadPtx(d: *const Driver, ptx: [:0]const u8, log: []u8) Error!Module {
        @memset(log, 0);
        var opts = [_]c_int{ abi.jit_error_log_buffer, abi.jit_error_log_buffer_size };
        var vals = [_]?*anyopaque{ log.ptr, @ptrFromInt(log.len) };
        var m: abi.Module = null;
        try d.check(d.api.cuModuleLoadDataEx(&m, ptx.ptr, opts.len, &opts, &vals), "cuModuleLoadDataEx");
        return .{ .d = d, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.d.api.cuModuleUnload(self.handle);
        self.* = undefined;
    }

    /// A kernel by its exact (mangled or extern "C") symbol; the function lives as long as the module. HIP builds
    /// look up the CUDA C++ name as hipcc mangles it (mangle.zig), then as given.
    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: abi.Function = null;
        if (hip) {
            var buf: [2048]u8 = undefined;
            const h = mangle.toHip(name, buf[0 .. buf.len - 1]);
            if (h.ptr != name.ptr) {
                buf[h.len] = 0;
                if (self.d.api.cuModuleGetFunction(&f, self.handle, buf[0..h.len :0].ptr) == abi.success) return .{ .d = self.d, .handle = f };
            }
        }
        self.d.check(self.d.api.cuModuleGetFunction(&f, self.handle, name.ptr), "cuModuleGetFunction") catch |e| {
            std.log.err("kernel symbol not in module: {s}", .{name});
            return e;
        };
        return .{ .d = self.d, .handle = f };
    }

    pub const Global = struct { ptr: abi.DevicePtr, len: usize };

    pub fn global(self: Module, name: [:0]const u8) Error!Global {
        var g: Global = .{ .ptr = 0, .len = 0 };
        try self.d.check(self.d.api.cuModuleGetGlobal_v2(&g.ptr, &g.len, self.handle, name.ptr), "cuModuleGetGlobal");
        return g;
    }
};

pub const Function = struct {
    d: *const Driver,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuFuncGetAttribute(&v, a, self.handle), "cuFuncGetAttribute");
        return v;
    }

    pub fn setAttribute(self: Function, a: abi.FunctionAttribute, value: c_int) Error!void {
        try self.d.check(self.d.api.cuFuncSetAttribute(self.handle, a, value), "cuFuncSetAttribute");
    }

    /// Resident blocks an SM holds of this kernel at `threads` a block and `shared` dynamic bytes.
    pub fn occupancy(self: Function, threads: u32, shared: usize) Error!u32 {
        var n: c_int = 0;
        try self.d.check(self.d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, @intCast(threads), shared), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
        return @intCast(@max(n, 0));
    }

    /// Dynamic shared memory above 48 KiB must be opted into per kernel before launch.
    pub fn allowDynamicShared(self: Function, bytes: u32) Error!void {
        try self.setAttribute(.max_dynamic_shared_size_bytes, @intCast(bytes));
    }
};
