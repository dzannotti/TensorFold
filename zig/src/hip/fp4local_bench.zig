//! `zig build hip-fp4local-bench`: times FP4-Local's products at the checkpoint's shapes (GB/s, TFLOPS, power).

test {
    _ = @import("kernel_tests/fp4local_bench.zig");
}
