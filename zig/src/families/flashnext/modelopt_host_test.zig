//! The backend-neutral ModelOpt readers and HIP's FP4-Local admission, built and run with core alone, no GPU backend.
test {
    _ = @import("modelopt_config.zig");
    _ = @import("mxfp8_read.zig");
    _ = @import("nvfp4_read.zig");
    _ = @import("ple_host.zig");
    _ = @import("names_modelopt.zig");
    _ = @import("modelopt_host.zig");
    _ = @import("hip/modelopt_admit.zig");
    _ = @import("hip/plan.zig");
    _ = @import("hip/tensors.zig");
}
