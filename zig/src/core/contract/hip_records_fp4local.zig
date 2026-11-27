//! Gate 5's records of FP4-Local's kernels (fp4local/fp4local.hip): gfx11 only; RDNA2 and gfx12 keep trap bodies.
const Record = @import("hip_records.zig").Record;

pub const records = [37]Record{
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 90, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 89, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 163, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 161, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 161, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x8", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi8ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 206, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 206, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 1x8 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi1ELi8ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 202, .sgprs = 47, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 47, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 84, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 84, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 83, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 154, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 152, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x8", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi8ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 227, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 227, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 2x8 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi2ELi8ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 223, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 4x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi4ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 120, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 4x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi4ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 119, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 4x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi4ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 171, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 171, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream NVFP4 4x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi0ELi4ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 168, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 168, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 1x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi1ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 113, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 1x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi1ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 112, .sgprs = 31, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 31, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 1x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi1ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 199, .sgprs = 36, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 36, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 1x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi1ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 199, .sgprs = 36, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 199, .sgprs = 36, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 2x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi2ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 84, .sgprs = 42, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 84, .sgprs = 42, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 2x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi2ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 84, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 84, .sgprs = 43, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 2x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi2ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 159, .sgprs = 44, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 159, .sgprs = 44, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 2x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi2ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 159, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 159, .sgprs = 45, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 4x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi4ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 120, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 4x2 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi4ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 120, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 4x4", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi4ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 173, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 173, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream MXFP8 4x4 swiglu", .symbol = "_ZN2tf4rocm11fp4l_streamILi1ELi4ELi4ELb1EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 173, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 173, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream BF16 1x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi2ELi1ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 93, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream BF16 2x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi2ELi2ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 79, .sgprs = 42, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 79, .sgprs = 42, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local stream BF16 4x2", .symbol = "_ZN2tf4rocm11fp4l_streamILi2ELi4ELi2ELb0EEEvNS0_6AffineENS0_11StreamSidesEi", .image = "fp4local", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 115, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local pair tile NVFP4", .symbol = "_ZN2tf4rocm9fp4l_pairILi0EEEvNS0_6AffineEf", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 208, .sgprs = 42, .lds = 35840, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 42, .lds = 35840, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local pair tile MXFP8", .symbol = "_ZN2tf4rocm9fp4l_pairILi1EEEvNS0_6AffineEf", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 214, .sgprs = 32, .lds = 35840, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 214, .sgprs = 32, .lds = 35840, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local item tile NVFP4", .symbol = "_ZN2tf4rocm9fp4l_itemILi0EEEvNS0_6AffineE", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 120, .sgprs = 44, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 44, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local matrix tile NVFP4", .symbol = "_ZN2tf4rocm9fp4l_wmmaILi0EEEvNS0_6AffineE", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 211, .sgprs = 42, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 211, .sgprs = 42, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local matrix tile MXFP8", .symbol = "_ZN2tf4rocm9fp4l_wmmaILi1EEEvNS0_6AffineE", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 218, .sgprs = 33, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 218, .sgprs = 33, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "FP4-Local matrix tile BF16", .symbol = "_ZN2tf4rocm9fp4l_wmmaILi2EEEvNS0_6AffineE", .image = "fp4local", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 230, .sgprs = 37, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 230, .sgprs = 37, .lds = 37376, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
};
