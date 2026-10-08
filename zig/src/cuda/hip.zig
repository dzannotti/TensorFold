//! The driver table filled from AMD's HIP runtime (libamdhip64): each `abi.Api` field bound to the HIP entry point
//! with the same ABI, or to a shim here where HIP's enums or structs differ (device attributes, kernel nodes,
//! launch attributes, graph updates). Call sites keep the CUDA names and types.

const std = @import("std");
const abi = @import("abi.zig");

const R = abi.Result;
const Ptr = ?*anyopaque;
const Params = ?[*]?*anyopaque;

/// Where libamdhip64 is looked for: the loader's path first, then ROCm's default install.
pub const paths = [_][]const u8{ "libamdhip64.so.7", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so" };

/// HIP's name for each `abi.Api` field whose ABI matches CUDA's; fields absent here are shims.
pub const names = .{
    .cuInit = "hipInit",
    .cuDriverGetVersion = "hipDriverGetVersion",
    .cuDeviceGet = "hipDeviceGet",
    .cuDeviceGetCount = "hipGetDeviceCount",
    .cuDeviceGetName = "hipDeviceGetName",
    .cuDeviceTotalMem_v2 = "hipDeviceTotalMem",
    .cuDevicePrimaryCtxRetain = "hipDevicePrimaryCtxRetain",
    .cuDevicePrimaryCtxRelease_v2 = "hipDevicePrimaryCtxRelease",
    .cuCtxSetCurrent = "hipCtxSetCurrent",
    .cuCtxGetCurrent = "hipCtxGetCurrent",
    .cuCtxSynchronize = "hipDeviceSynchronize",
    .cuMemGetInfo_v2 = "hipMemGetInfo",
    .cuMemAlloc_v2 = "hipMalloc",
    .cuMemFree_v2 = "hipFree",
    .cuMemHostAlloc = "hipHostAlloc", // hipHostMallocPortable/Mapped are CUDA's 1 and 2
    .cuMemHostGetDevicePointer_v2 = "hipHostGetDevicePointer",
    .cuMemFreeHost = "hipHostFree",
    .cuMemcpyHtoD_v2 = "hipMemcpyHtoD",
    .cuMemcpyDtoH_v2 = "hipMemcpyDtoH",
    .cuMemcpyDtoD_v2 = "hipMemcpyDtoD",
    .cuMemcpyHtoDAsync_v2 = "hipMemcpyHtoDAsync",
    .cuMemcpyDtoHAsync_v2 = "hipMemcpyDtoHAsync",
    .cuMemcpyDtoDAsync_v2 = "hipMemcpyDtoDAsync",
    .cuMemsetD8_v2 = "hipMemsetD8",
    .cuMemsetD32_v2 = "hipMemsetD32",
    .cuMemsetD8Async = "hipMemsetD8Async",
    .cuMemsetD32Async = "hipMemsetD32Async",
    .cuStreamCreate = "hipStreamCreateWithFlags", // hipStreamNonBlocking = 1
    .cuStreamCreateWithPriority = "hipStreamCreateWithPriority",
    .cuStreamDestroy_v2 = "hipStreamDestroy",
    .cuStreamSynchronize = "hipStreamSynchronize",
    .cuStreamWaitEvent = "hipStreamWaitEvent",
    .cuStreamQuery = "hipStreamQuery",
    .cuEventCreate = "hipEventCreateWithFlags", // hipEventDisableTiming = 2
    .cuEventDestroy_v2 = "hipEventDestroy",
    .cuEventRecord = "hipEventRecord",
    .cuEventSynchronize = "hipEventSynchronize",
    .cuEventQuery = "hipEventQuery",
    .cuEventElapsedTime = "hipEventElapsedTime",
    .cuModuleLoadData = "hipModuleLoadData",
    .cuModuleLoadDataEx = "hipModuleLoadDataEx", // hipJitOption error log 5/6 as CUDA's
    .cuModuleUnload = "hipModuleUnload",
    .cuModuleGetFunction = "hipModuleGetFunction",
    .cuModuleGetGlobal_v2 = "hipModuleGetGlobal",
    .cuFuncGetAttribute = "hipFuncGetAttribute", // HIP_FUNC_ATTRIBUTE_* 0..8 are CUDA's
    .cuOccupancyMaxActiveBlocksPerMultiprocessor = "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    .cuLaunchKernel = "hipModuleLaunchKernel",
    .cuStreamBeginCapture_v2 = "hipStreamBeginCapture", // capture modes and statuses 0..2 are CUDA's
    .cuStreamEndCapture = "hipStreamEndCapture",
    .cuStreamIsCapturing = "hipStreamIsCapturing",
    .cuGraphCreate = "hipGraphCreate",
    .cuGraphDestroy = "hipGraphDestroy",
    .cuGraphAddDependencies = "hipGraphAddDependencies",
    .cuGraphGetNodes = "hipGraphGetNodes",
    .cuGraphInstantiateWithFlags = "hipGraphInstantiateWithFlags",
    .cuGraphUpload = "hipGraphUpload",
    .cuGraphLaunch = "hipGraphLaunch",
    .cuGraphExecDestroy = "hipGraphExecDestroy",
    .cuGetErrorName = "hipDrvGetErrorName",
    .cuGetErrorString = "hipDrvGetErrorString",
};

/// hipKernelNodeParams (hip_runtime_api.h): CUDA's fields in another order.
const KernelNodeParams = extern struct {
    block: abi.Dim3,
    extra: Params,
    func: abi.Function,
    grid: abi.Dim3,
    params: Params,
    shared_bytes: c_uint,
};

comptime {
    std.debug.assert(@sizeOf(KernelNodeParams) == 64 and @offsetOf(KernelNodeParams, "func") == 24 and @offsetOf(KernelNodeParams, "params") == 48);
}

/// The HIP entry points the shims call (HIP signatures).
const Raw = struct {
    hipDeviceGetAttribute: *const fn (*c_int, c_int, abi.Device) callconv(.c) R,
    hipModuleLaunchKernel: *const fn (abi.Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, abi.Stream, Params, Params) callconv(.c) R,
    hipModuleLaunchCooperativeKernel: *const fn (abi.Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, abi.Stream, Params) callconv(.c) R,
    hipGraphAddKernelNode: *const fn (*abi.GraphNode, abi.Graph, ?[*]const abi.GraphNode, usize, *const KernelNodeParams) callconv(.c) R,
    hipGraphKernelNodeGetParams: *const fn (abi.GraphNode, *KernelNodeParams) callconv(.c) R,
    hipGraphKernelNodeSetParams: *const fn (abi.GraphNode, *const KernelNodeParams) callconv(.c) R,
    hipGraphExecKernelNodeSetParams: *const fn (abi.GraphExec, abi.GraphNode, *const KernelNodeParams) callconv(.c) R,
    hipGraphExecUpdate: *const fn (abi.GraphExec, abi.Graph, *abi.GraphNode, *abi.ExecUpdateResult) callconv(.c) R,
    hipGetDevicePropertiesR0600: *const fn (*[prop_size]u8, abi.Device) callconv(.c) R,
};

// ponytail: one process-wide table, since a process loads one HIP runtime; per-Driver state if that ever changes
var raw: Raw = undefined;

const invalid_value: R = 1;
const not_supported: R = 801;

/// hipDeviceProp_t (R0600) and its gcnArchName, the one property no attribute carries.
const prop_size = 1472;
const arch_offset = 1160;

/// Fills `api` from `lib`; null, or the name of the symbol it lacks.
pub fn bind(lib: *std.DynLib, api: *abi.Api) ?[]const u8 {
    const r = @typeInfo(Raw).@"struct";
    inline for (r.field_names, r.field_types) |name, T| {
        @field(raw, name) = lib.lookup(T, name) orelse return name;
    }
    const a = @typeInfo(abi.Api).@"struct";
    inline for (a.field_names, a.field_types) |name, T| {
        if (@hasField(@TypeOf(names), name)) {
            const sym = @field(names, name);
            @field(api, name) = lib.lookup(T, sym) orelse return sym;
        } else @field(api, name) = &@field(shims, name);
    }
    return null;
}

/// The device's AMDGPU target ("gfx1151"), feature flags (":xnack-") cut off.
pub fn archName(dev: abi.Device, out: []u8) []const u8 {
    var p: [prop_size]u8 align(8) = undefined;
    if (raw.hipGetDevicePropertiesR0600(&p, dev) != abi.success) return "";
    const s = std.mem.sliceTo(p[arch_offset..][0..256], 0);
    const base = s[0 .. std.mem.indexOfScalar(u8, s, ':') orelse s.len];
    const n = @min(base.len, out.len);
    @memcpy(out[0..n], base[0..n]);
    return out[0..n];
}

/// hipDeviceAttribute_t numbers CUDA's attributes differently.
fn deviceAttribute(a: abi.DeviceAttribute) c_int {
    return switch (a) {
        .max_threads_per_block => 56,
        .warp_size => 87,
        .clock_rate => 5,
        .multiprocessor_count => 63,
        .integrated => 16,
        .l2_cache_size => 19,
        .max_threads_per_multiprocessor => 57,
        .compute_capability_major => 23,
        .compute_capability_minor => 61,
        .max_shared_memory_per_multiprocessor => 10002,
        .max_shared_memory_per_block_optin => 75,
    };
}

fn toHip(p: *const abi.KernelNodeParams) KernelNodeParams {
    return .{
        .block = .{ .x = p.block_x, .y = p.block_y, .z = p.block_z },
        .extra = p.extra,
        .func = p.func,
        .grid = .{ .x = p.grid_x, .y = p.grid_y, .z = p.grid_z },
        .params = p.params,
        .shared_bytes = p.shared_bytes,
    };
}

const shims = struct {
    pub fn cuDeviceGetAttribute(v: *c_int, a: abi.DeviceAttribute, d: abi.Device) callconv(.c) R {
        return raw.hipDeviceGetAttribute(v, deviceAttribute(a), d);
    }

    /// AMD kernels need no opt-in for dynamic LDS up to the 64 KiB a workgroup has; above that the launch fails.
    pub fn cuFuncSetAttribute(_: abi.Function, a: abi.FunctionAttribute, value: c_int) callconv(.c) R {
        return switch (a) {
            .max_dynamic_shared_size_bytes => if (value <= 64 << 10) abi.success else invalid_value,
            .non_portable_cluster_size_allowed => not_supported,
            else => abi.success,
        };
    }

    /// AMD has no L1/shared split to prefer.
    pub fn cuFuncSetCacheConfig(_: abi.Function, _: c_int) callconv(.c) R {
        return abi.success;
    }

    /// Cooperative grids launch as such; PDL is dropped (the launch then waits for the previous grid, which is
    /// what PDL relaxes); clusters do not exist on AMD and are refused.
    pub fn cuLaunchKernelEx(c: *const abi.LaunchConfig, f: abi.Function, params: Params, extra: Params) callconv(.c) R {
        var coop = false;
        if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| switch (at.id) {
            .cooperative => coop = at.value.int != 0,
            .programmatic_stream_serialization, .priority => {},
            .cluster_dimension, .cluster_scheduling_policy_preference => return not_supported,
        };
        if (coop) {
            if (extra != null) return not_supported;
            return raw.hipModuleLaunchCooperativeKernel(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, params);
        }
        return raw.hipModuleLaunchKernel(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, params, extra);
    }

    pub fn cuGraphAddKernelNode_v2(n: *abi.GraphNode, g: abi.Graph, deps: ?[*]const abi.GraphNode, count: usize, p: *const abi.KernelNodeParams) callconv(.c) R {
        const h = toHip(p);
        return raw.hipGraphAddKernelNode(n, g, deps, count, &h);
    }

    pub fn cuGraphKernelNodeGetParams_v2(n: abi.GraphNode, p: *abi.KernelNodeParams) callconv(.c) R {
        var h: KernelNodeParams = undefined;
        const res = raw.hipGraphKernelNodeGetParams(n, &h);
        if (res != abi.success) return res;
        p.* = .{ .func = h.func, .grid_x = h.grid.x, .grid_y = h.grid.y, .grid_z = h.grid.z, .block_x = h.block.x, .block_y = h.block.y, .block_z = h.block.z, .shared_bytes = h.shared_bytes, .params = h.params, .extra = h.extra };
        return res;
    }

    pub fn cuGraphKernelNodeSetParams_v2(n: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) R {
        const h = toHip(p);
        return raw.hipGraphKernelNodeSetParams(n, &h);
    }

    pub fn cuGraphExecKernelNodeSetParams_v2(e: abi.GraphExec, n: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) R {
        const h = toHip(p);
        return raw.hipGraphExecKernelNodeSetParams(e, n, &h);
    }

    /// hipGraphExecUpdate returns the error node and verdict as two out-parameters.
    pub fn cuGraphExecUpdate_v2(e: abi.GraphExec, g: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) R {
        info.error_from_node = null;
        return raw.hipGraphExecUpdate(e, g, &info.error_node, &info.result);
    }
};

test "every Api field is a HIP symbol or a shim" {
    const a = @typeInfo(abi.Api).@"struct";
    inline for (a.field_names, a.field_types) |name, T| {
        const named = @hasField(@TypeOf(names), name);
        const shimmed = @hasDecl(shims, name);
        try std.testing.expect(named != shimmed);
        if (shimmed) try std.testing.expectEqual(T, @TypeOf(&@field(shims, name)));
    }
}
