//! Gate 5's records of the Loom kernels (zig/kernels/loom): gfx1151 only, -Dloom-target's chip.
const Record = @import("hip_records.zig").Record;

pub const records = [40]Record{
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r1_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 116, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r2_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 116, .sgprs = 38, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k2560 l16 s1 c1 w4 g5 p1 x4", .symbol = "mxfp8_gemv_r4_k2560_l16_s1_c1_w4_g5_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 144, .sgprs = 44, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k2560 l16 s1 c1 w4 g1 p1 x4", .symbol = "mxfp8_gemv_r8_k2560_l16_s1_c1_w4_g1_p1_x4", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r1_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r1_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r2_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r2_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r4_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r4_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k2560 l16 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r8_k2560_l16_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k2560 l16 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r8_k2560_l16_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r1_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 78, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r1_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 78, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r2_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 115, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r2_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 115, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r4_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 140, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r4_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 140, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k6144 l32 s2 c1 w4 g2 p1", .symbol = "mxfp8_gemv_r8_k6144_l32_s2_c1_w4_g2_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 213, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k6144 l32 s2 c1 w4 g2 p1 bf16", .symbol = "mxfp8_gemv_r8_k6144_l32_s2_c1_w4_g2_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 213, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r1_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r1 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r1_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 50, .sgprs = 22, .lds = 512, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r2_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r2 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r2_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 68, .sgprs = 26, .lds = 1024, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r4_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r4 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r4_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 100, .sgprs = 30, .lds = 2048, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k640 l4 s2 c1 w4 g1 p1", .symbol = "mxfp8_gemv_r8_k640_l4_s2_c1_w4_g1_p1", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom mxfp8 gemv r8 k640 l4 s2 c1 w4 g1 p1 bf16", .symbol = "mxfp8_gemv_r8_k640_l4_s2_c1_w4_g1_p1_bf16", .image = "loom_mxfp8_gemv", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 167, .sgprs = 38, .lds = 4096, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom nvfp4 expert gate up", .symbol = "nvfp4_expert_gate_up", .image = "loom_nvfp4_expert_gate_up", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 256, .sgprs = 52, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom nvfp4 expert down", .symbol = "nvfp4_expert_down", .image = "loom_nvfp4_expert_down", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 240, .sgprs = 52, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom sparse attend", .symbol = "sparse_attend", .image = "loom_sparse_attend", .threads = 512 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 192, .sgprs = 34, .lds = 34304, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom idx scores", .symbol = "idx_scores", .image = "loom_idx_scores", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 216, .sgprs = 50, .lds = 32768, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom idx select", .symbol = "idx_select", .image = "loom_idx_select", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 17, .sgprs = 46, .lds = 1032, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom bf16 gemm 128x128", .symbol = "bf16_gemm_128x128", .image = "loom_bf16_gemm_128x128", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 256, .sgprs = 30, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom bf16 gemm 128x256", .symbol = "bf16_gemm_128x256", .image = "loom_bf16_gemm_128x256", .threads = 512 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 240, .sgprs = 30, .lds = 61440, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom gdn recur", .symbol = "gdn_recur", .image = "loom_gdn_recur", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 216, .sgprs = 102, .lds = 10304, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom hc norm", .symbol = "hc_norm", .image = "loom_hc_norm", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 92, .sgprs = 24, .lds = 16896, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom hc down wide", .symbol = "hc_down_wide", .image = "loom_hc_down_wide", .threads = 512 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 252, .sgprs = 32, .lds = 65536, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom hc down narrow", .symbol = "hc_down_narrow", .image = "loom_hc_down_narrow", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 256, .sgprs = 34, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
    .{
        .launch = .{ .name = "Loom hc up mix", .symbol = "hc_up_mix", .image = "loom_hc_up_mix", .threads = 512 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 226, .sgprs = 30, .lds = 65536, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
};
