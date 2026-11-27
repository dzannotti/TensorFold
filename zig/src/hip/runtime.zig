//! Complete model-free HIP runtime admission; partial symbol tables never escape.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const abi = @import("abi.zig");
const hrx = @import("hrx/api.zig");
pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, Invalid };

/// The runtime's file names, most specific first: Windows' unversioned DLL, then the SDK's versioned copies.
const names: []const []const u8 = if (builtin.os.tag == .windows)
    &.{ "amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll" }
else
    &.{"libamdhip64.so"};

pub const Runtime = struct {
    lib: platform.DynLib,
    api: abi.Api,

    /// TENSORFOLD_GPU=hrx selects the native HRX driver (libhrx, TENSORFOLD_HRX names its path); otherwise HIP.
    pub fn open() Error!Runtime {
        const env = platform.environment;
        if ((env.read(u32, "TENSORFOLD_GPU", 0) catch return error.Invalid) == 1) {
            const path = env.path("TENSORFOLD_HRX", env.rawText("TENSORFOLD_HRX")) catch return error.Invalid;
            return openHrx(path orelse hrx.library);
        }
        if (platform.getenv("TENSORFOLD_HIP")) |path| return openPath(path);
        for (names) |name| {
            return openPath(name) catch |err| switch (err) {
                error.DriverUnavailable => continue,
                else => return err,
            };
        }
        return error.DriverUnavailable;
    }

    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = platform.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(abi.Api, &lib);
        try check(api.hipInit(0));
        return .{ .lib = lib, .api = api };
    }

    pub fn openHrx(path: []const u8) Error!Runtime {
        var lib = platform.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = hrx.open(path) catch |err| return switch (err) {
            error.MissingSymbol => error.MissingSymbol,
            error.HipFailed => error.HipFailed,
            else => error.DriverUnavailable,
        };
        return .{ .lib = lib, .api = api };
    }

    /// HIP's name for a failed call's result, for logs (`hipErrorOutOfMemory` ...).
    pub fn errorName(self: *const Runtime, result: abi.Result) []const u8 {
        const name = self.api.hipGetErrorName(result) orelse return "unknown HIP error";
        return std.mem.span(name);
    }

    /// The GPUs this process sees.
    pub fn deviceCount(self: *const Runtime) Error!c_int {
        var n: c_int = 0;
        try check(self.api.hipGetDeviceCount(&n));
        return n;
    }

    pub fn close(self: *Runtime) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub fn check(result: abi.Result) Error!void {
    if (result != 0) return error.HipFailed;
}

test "complete runtime refuses absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/hip-runtime.so"));
    try std.testing.expectError(error.DriverUnavailable, Runtime.openHrx("/nonexistent/libhrx.so"));
}
