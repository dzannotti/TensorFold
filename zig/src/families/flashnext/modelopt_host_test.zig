//! The backend-neutral ModelOpt readers built and run with core alone: none of them may import a GPU backend.
test {
    _ = @import("modelopt_config.zig");
    _ = @import("mxfp8_read.zig");
    _ = @import("nvfp4_read.zig");
    _ = @import("ple_host.zig");
    _ = @import("names_modelopt.zig");
}
