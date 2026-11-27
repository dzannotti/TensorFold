//! HIP's entry-point table on libhrx's native API (TENSORFOLD_GPU=hrx); device facts come from sysfs, not HRX.
const std = @import("std");
const abi = @import("../abi.zig");
const drv = @import("driver.zig");
const memory = @import("memory.zig");
const streams = @import("streams.zig");
const modules = @import("modules.zig");
const launch = @import("launch.zig");
const graph = @import("graph.zig");

/// The default libhrx name when TENSORFOLD_HRX names no path.
pub const library = "libhrx.so.0";

pub fn open(path: []const u8) !abi.Api {
    try drv.open(path);
    return table;
}

/// True when `api` is this driver's table (the runtime then has no HIP library to look symbols up in).
pub fn owns(api: *const abi.Api) bool {
    return api.hipInit == table.hipInit;
}

/// The device's architecture name ("gfx1151"), as HIP's gcnArchName gives it.
pub fn arch(out: []u8) ?[]const u8 {
    const a = std.mem.sliceTo(&drv.d().arch, 0);
    if (a.len == 0 or a.len >= out.len) return null;
    @memcpy(out[0..a.len], a);
    out[a.len] = 0;
    return out[0..a.len];
}

fn hipInit(flags: c_uint) callconv(.c) abi.Result {
    return if (flags == 0) drv.success else drv.invalid_value;
}

fn hipGetDeviceCount(n: *c_int) callconv(.c) abi.Result {
    n.* = 1;
    return drv.success;
}

fn hipDeviceGet(d: *abi.Device, ordinal: c_int) callconv(.c) abi.Result {
    if (ordinal != 0) return 101;
    d.* = 0;
    return drv.success;
}

var context_token: u8 = 0;

fn hipDevicePrimaryCtxRetain(ctx: *abi.Context, d: abi.Device) callconv(.c) abi.Result {
    if (d != 0) return 101;
    ctx.* = @ptrCast(&context_token);
    return drv.success;
}

fn hipDevicePrimaryCtxRelease(d: abi.Device) callconv(.c) abi.Result {
    return if (d == 0) drv.success else 101;
}

fn hipCtxSetCurrent(ctx: abi.Context) callconv(.c) abi.Result {
    return if (ctx == null or @intFromPtr(ctx) == @intFromPtr(&context_token)) drv.success else 201;
}

fn hipGetErrorName(r: abi.Result) callconv(.c) ?[*:0]const u8 {
    return switch (r) {
        drv.success => "hipSuccess",
        drv.invalid_value => "hipErrorInvalidValue",
        drv.out_of_memory => "hipErrorOutOfMemory",
        drv.not_initialized => "hipErrorNotInitialized",
        101 => "hipErrorInvalidDevice",
        200 => "hipErrorInvalidImage",
        201 => "hipErrorInvalidContext",
        drv.invalid_handle => "hipErrorInvalidHandle",
        drv.not_found => "hipErrorNotFound",
        drv.not_ready => "hipErrorNotReady",
        drv.not_supported => "hipErrorNotSupported",
        900 => "hipErrorStreamCaptureUnsupported",
        901 => "hipErrorStreamCaptureInvalidated",
        else => "hipErrorUnknown",
    };
}

fn hipRuntimeGetVersion(v: *c_int) callconv(.c) abi.Result {
    v.* = 70000000;
    return drv.success;
}

fn hipDeviceGetName(out: [*]u8, len: c_int, d: abi.Device) callconv(.c) abi.Result {
    if (d != 0) return 101;
    if (len <= 1) return drv.invalid_value;
    const name = std.mem.sliceTo(&drv.d().name, 0);
    const n = @min(name.len, @as(usize, @intCast(len)) - 1);
    @memcpy(out[0..n], name[0..n]);
    out[n] = 0;
    return drv.success;
}

fn hipDeviceGetAttribute(v: *c_int, a: abi.DeviceAttribute, d: c_int) callconv(.c) abi.Result {
    if (d != 0) return 101;
    v.* = drv.d().facts.attribute(a) orelse return drv.invalid_value;
    return drv.success;
}

const table: abi.Api = .{
    .hipInit = hipInit,
    .hipGetDeviceCount = hipGetDeviceCount,
    .hipDeviceGet = hipDeviceGet,
    .hipDevicePrimaryCtxRetain = hipDevicePrimaryCtxRetain,
    .hipDevicePrimaryCtxRelease = hipDevicePrimaryCtxRelease,
    .hipCtxSetCurrent = hipCtxSetCurrent,
    .hipDeviceSynchronize = streams.hipDeviceSynchronize,
    .hipMalloc = memory.hipMalloc,
    .hipFree = memory.hipFree,
    .hipHostMalloc = memory.hipHostMalloc,
    .hipHostFree = memory.hipHostFree,
    .hipMemcpyHtoDAsync = memory.hipMemcpyHtoDAsync,
    .hipMemcpyDtoHAsync = memory.hipMemcpyDtoHAsync,
    .hipMemcpyHtoD = memory.hipMemcpyHtoD,
    .hipMemcpyDtoH = memory.hipMemcpyDtoH,
    .hipMemset = memory.hipMemset,
    .hipMemsetAsync = memory.hipMemsetAsync,
    .hipStreamCreateWithFlags = streams.hipStreamCreateWithFlags,
    .hipStreamDestroy = streams.hipStreamDestroy,
    .hipStreamSynchronize = streams.hipStreamSynchronize,
    .hipModuleLoadData = modules.hipModuleLoadData,
    .hipModuleUnload = modules.hipModuleUnload,
    .hipModuleGetFunction = modules.hipModuleGetFunction,
    .hipStreamQuery = streams.hipStreamQuery,
    .hipStreamWaitEvent = streams.hipStreamWaitEvent,
    .hipEventCreateWithFlags = streams.hipEventCreateWithFlags,
    .hipEventDestroy = streams.hipEventDestroy,
    .hipEventRecord = streams.hipEventRecord,
    .hipEventSynchronize = streams.hipEventSynchronize,
    .hipEventQuery = streams.hipEventQuery,
    .hipEventElapsedTime = streams.hipEventElapsedTime,
    .hipMemGetInfo = memory.hipMemGetInfo,
    .hipModuleLaunchKernel = launch.hipModuleLaunchKernel,
    .hipGetErrorName = hipGetErrorName,
    .hipRuntimeGetVersion = hipRuntimeGetVersion,
    .hipDeviceGetName = hipDeviceGetName,
    .hipDeviceGetAttribute = hipDeviceGetAttribute,
    .hipHostGetDevicePointer = memory.hipHostGetDevicePointer,
    .hipMemcpyDtoD = memory.hipMemcpyDtoD,
    .hipMemcpyDtoDAsync = memory.hipMemcpyDtoDAsync,
    .hipMemsetD32 = memory.hipMemsetD32,
    .hipMemsetD32Async = memory.hipMemsetD32Async,
    .hipModuleGetGlobal = modules.hipModuleGetGlobal,
    .hipFuncGetAttribute = modules.hipFuncGetAttribute,
    .hipModuleOccupancyMaxActiveBlocksPerMultiprocessor = modules.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
    .hipModuleLaunchCooperativeKernel = launch.hipModuleLaunchCooperativeKernel,
    .hipStreamBeginCapture = graph.hipStreamBeginCapture,
    .hipStreamEndCapture = graph.hipStreamEndCapture,
    .hipStreamIsCapturing = graph.hipStreamIsCapturing,
    .hipGraphCreate = graph.hipGraphCreate,
    .hipGraphDestroy = graph.hipGraphDestroy,
    .hipGraphAddKernelNode = graph.hipGraphAddKernelNode,
    .hipGraphKernelNodeSetParams = graph.hipGraphKernelNodeSetParams,
    .hipGraphAddDependencies = graph.hipGraphAddDependencies,
    .hipGraphGetNodes = graph.hipGraphGetNodes,
    .hipGraphInstantiateWithFlags = graph.hipGraphInstantiateWithFlags,
    .hipGraphUpload = graph.hipGraphUpload,
    .hipGraphLaunch = graph.hipGraphLaunch,
    .hipGraphExecDestroy = graph.hipGraphExecDestroy,
    .hipGraphExecUpdate = graph.hipGraphExecUpdate,
    .hipGraphExecKernelNodeSetParams = graph.hipGraphExecKernelNodeSetParams,
};

test {
    _ = @import("table.zig");
    _ = @import("sysfs.zig");
    _ = @import("capture.zig");
    _ = launch;
    _ = graph;
    _ = memory;
}
