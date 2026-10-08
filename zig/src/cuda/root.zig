//! TensorFold's CUDA runtime: the driver, cuBLASLt and NCCL through dlopen, our kernels and Triton's, no Python.

pub const abi = @import("abi.zig");
pub const Driver = @import("driver.zig").Driver;
/// True in -Dgpu=hip builds (AMD HIP runtime, AMDGPU code objects); `Context.features` names what the device can do.
pub const hip = @import("driver.zig").hip;
pub const mangle = @import("mangle.zig");
/// Where a HIP build looks for libamdhip64.
pub const hip_paths = @import("hip.zig").paths;
pub const Error = @import("driver.zig").Error;
pub const Context = @import("context.zig").Context;
pub const DeviceBuffer = @import("memory.zig").DeviceBuffer;
pub const HostBuffer = @import("memory.zig").HostBuffer;
pub const Stream = @import("stream.zig").Stream;
pub const Event = @import("stream.zig").Event;
pub const Module = @import("module.zig").Module;
pub const Function = @import("module.zig").Function;
pub const launch = @import("launch.zig");
pub const Args = launch.Args;
pub const Config = launch.Config;
pub const Dim3 = launch.Dim3;
pub const graph = @import("graph.zig");
pub const cublaslt = @import("cublaslt.zig");
pub const nccl = @import("nccl.zig");
pub const roce = @import("roce.zig");
pub const triton = @import("triton.zig");
pub const aot = @import("aot.zig");
pub const kernels = @import("kernels.zig");
pub const segments = @import("segments.zig");

test {
    _ = launch;
    _ = abi;
    _ = aot;
    _ = segments;
    _ = mangle;
    _ = @import("hip.zig");
    _ = roce;
}
