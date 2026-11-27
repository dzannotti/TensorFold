//! Gate 5's records of the Loom MXFP8 GEMV (zig/kernels/loom/mxfp8_gemv): gfx1151 only, -Dloom-target's chip.
const Record = @import("hip_records.zig").Record;

pub const records = [28]Record{
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r1_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 116, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r2_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 116, .sgprs = 38, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r4_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 144, .sgprs = 44, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k2560 l16 s1 c1 w4 g1 p1 x4", .symbol = "mxfp8_gemv_r8_k2560_l16_s1_c1_w4_g1_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r1_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r1_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r2_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r2_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r4_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r4_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r8_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r8_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r1_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 78, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r1_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 78, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r2_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 115, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r2_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 115, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r4_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 140, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r4_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 140, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r8_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 213, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r8_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 213, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r1_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r1 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r1_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r2_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r2 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r2_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r4_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r4 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r4_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r8_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8_gemv r8 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r8_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
};
