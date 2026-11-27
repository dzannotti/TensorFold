// SPDX-License-Identifier: MIT

// gfx11 Low blocks of the attention core's two fp32 chains (README: Sum order); every product and sum is rounded.
#ifndef SA_LANES_ASM_H_
#define SA_LANES_ASM_H_

#include <loomcxx/kernel.h>
#include <loomcxx/low.h>

namespace sa {

struct [[loom::representation("amdgpu.gfx11.generic.core")]] Gfx11 {};

// Six heads' acc + q_i k_i for i = 0..15 in order, all rounded; q_i is lane i of this lane's row (DPP row_share).
struct F6 {
  float a0, a1, a2, a3, a4, a5;
};

LOOM_INLINE F6 qk16x6(F6 acc, F6 q, float k0, float k1, float k2, float k3, float k4, float k5, float k6, float k7,
                      float k8, float k9, float k10, float k11, float k12, float k13, float k14, float k15) {
  return loom::low::assembly<Gfx11, F6>(R"((%a0: reg<amdgpu.vgpr>, %a1: reg<amdgpu.vgpr>, %a2: reg<amdgpu.vgpr>,
    %a3: reg<amdgpu.vgpr>,
    %a4: reg<amdgpu.vgpr>, %a5: reg<amdgpu.vgpr>, %q0: reg<amdgpu.vgpr>, %q1: reg<amdgpu.vgpr>,
    %q2: reg<amdgpu.vgpr>, %q3: reg<amdgpu.vgpr>, %q4: reg<amdgpu.vgpr>, %q5: reg<amdgpu.vgpr>,
    %k0: reg<amdgpu.vgpr>, %k1: reg<amdgpu.vgpr>, %k2: reg<amdgpu.vgpr>, %k3: reg<amdgpu.vgpr>,
    %k4: reg<amdgpu.vgpr>, %k5: reg<amdgpu.vgpr>, %k6: reg<amdgpu.vgpr>, %k7: reg<amdgpu.vgpr>,
    %k8: reg<amdgpu.vgpr>, %k9: reg<amdgpu.vgpr>, %k10: reg<amdgpu.vgpr>, %k11: reg<amdgpu.vgpr>,
    %k12: reg<amdgpu.vgpr>, %k13: reg<amdgpu.vgpr>, %k14: reg<amdgpu.vgpr>, %k15: reg<amdgpu.vgpr>)
    -> (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>,
    reg<amdgpu.vgpr>) {
  %p0_0 = v_mul_f32_dpp16 %q0, %k0 {dpp_ctrl = 336}
  %p1_0 = v_mul_f32_dpp16 %q1, %k0 {dpp_ctrl = 336}
  %p2_0 = v_mul_f32_dpp16 %q2, %k0 {dpp_ctrl = 336}
  %p3_0 = v_mul_f32_dpp16 %q3, %k0 {dpp_ctrl = 336}
  %p4_0 = v_mul_f32_dpp16 %q4, %k0 {dpp_ctrl = 336}
  %p5_0 = v_mul_f32_dpp16 %q5, %k0 {dpp_ctrl = 336}
  %s0_0 = v_add_f32 %a0, %p0_0
  %s1_0 = v_add_f32 %a1, %p1_0
  %s2_0 = v_add_f32 %a2, %p2_0
  %s3_0 = v_add_f32 %a3, %p3_0
  %s4_0 = v_add_f32 %a4, %p4_0
  %s5_0 = v_add_f32 %a5, %p5_0
  %p0_1 = v_mul_f32_dpp16 %q0, %k1 {dpp_ctrl = 337}
  %p1_1 = v_mul_f32_dpp16 %q1, %k1 {dpp_ctrl = 337}
  %p2_1 = v_mul_f32_dpp16 %q2, %k1 {dpp_ctrl = 337}
  %p3_1 = v_mul_f32_dpp16 %q3, %k1 {dpp_ctrl = 337}
  %p4_1 = v_mul_f32_dpp16 %q4, %k1 {dpp_ctrl = 337}
  %p5_1 = v_mul_f32_dpp16 %q5, %k1 {dpp_ctrl = 337}
  %s0_1 = v_add_f32 %s0_0, %p0_1
  %s1_1 = v_add_f32 %s1_0, %p1_1
  %s2_1 = v_add_f32 %s2_0, %p2_1
  %s3_1 = v_add_f32 %s3_0, %p3_1
  %s4_1 = v_add_f32 %s4_0, %p4_1
  %s5_1 = v_add_f32 %s5_0, %p5_1
  %p0_2 = v_mul_f32_dpp16 %q0, %k2 {dpp_ctrl = 338}
  %p1_2 = v_mul_f32_dpp16 %q1, %k2 {dpp_ctrl = 338}
  %p2_2 = v_mul_f32_dpp16 %q2, %k2 {dpp_ctrl = 338}
  %p3_2 = v_mul_f32_dpp16 %q3, %k2 {dpp_ctrl = 338}
  %p4_2 = v_mul_f32_dpp16 %q4, %k2 {dpp_ctrl = 338}
  %p5_2 = v_mul_f32_dpp16 %q5, %k2 {dpp_ctrl = 338}
  %s0_2 = v_add_f32 %s0_1, %p0_2
  %s1_2 = v_add_f32 %s1_1, %p1_2
  %s2_2 = v_add_f32 %s2_1, %p2_2
  %s3_2 = v_add_f32 %s3_1, %p3_2
  %s4_2 = v_add_f32 %s4_1, %p4_2
  %s5_2 = v_add_f32 %s5_1, %p5_2
  %p0_3 = v_mul_f32_dpp16 %q0, %k3 {dpp_ctrl = 339}
  %p1_3 = v_mul_f32_dpp16 %q1, %k3 {dpp_ctrl = 339}
  %p2_3 = v_mul_f32_dpp16 %q2, %k3 {dpp_ctrl = 339}
  %p3_3 = v_mul_f32_dpp16 %q3, %k3 {dpp_ctrl = 339}
  %p4_3 = v_mul_f32_dpp16 %q4, %k3 {dpp_ctrl = 339}
  %p5_3 = v_mul_f32_dpp16 %q5, %k3 {dpp_ctrl = 339}
  %s0_3 = v_add_f32 %s0_2, %p0_3
  %s1_3 = v_add_f32 %s1_2, %p1_3
  %s2_3 = v_add_f32 %s2_2, %p2_3
  %s3_3 = v_add_f32 %s3_2, %p3_3
  %s4_3 = v_add_f32 %s4_2, %p4_3
  %s5_3 = v_add_f32 %s5_2, %p5_3
  %p0_4 = v_mul_f32_dpp16 %q0, %k4 {dpp_ctrl = 340}
  %p1_4 = v_mul_f32_dpp16 %q1, %k4 {dpp_ctrl = 340}
  %p2_4 = v_mul_f32_dpp16 %q2, %k4 {dpp_ctrl = 340}
  %p3_4 = v_mul_f32_dpp16 %q3, %k4 {dpp_ctrl = 340}
  %p4_4 = v_mul_f32_dpp16 %q4, %k4 {dpp_ctrl = 340}
  %p5_4 = v_mul_f32_dpp16 %q5, %k4 {dpp_ctrl = 340}
  %s0_4 = v_add_f32 %s0_3, %p0_4
  %s1_4 = v_add_f32 %s1_3, %p1_4
  %s2_4 = v_add_f32 %s2_3, %p2_4
  %s3_4 = v_add_f32 %s3_3, %p3_4
  %s4_4 = v_add_f32 %s4_3, %p4_4
  %s5_4 = v_add_f32 %s5_3, %p5_4
  %p0_5 = v_mul_f32_dpp16 %q0, %k5 {dpp_ctrl = 341}
  %p1_5 = v_mul_f32_dpp16 %q1, %k5 {dpp_ctrl = 341}
  %p2_5 = v_mul_f32_dpp16 %q2, %k5 {dpp_ctrl = 341}
  %p3_5 = v_mul_f32_dpp16 %q3, %k5 {dpp_ctrl = 341}
  %p4_5 = v_mul_f32_dpp16 %q4, %k5 {dpp_ctrl = 341}
  %p5_5 = v_mul_f32_dpp16 %q5, %k5 {dpp_ctrl = 341}
  %s0_5 = v_add_f32 %s0_4, %p0_5
  %s1_5 = v_add_f32 %s1_4, %p1_5
  %s2_5 = v_add_f32 %s2_4, %p2_5
  %s3_5 = v_add_f32 %s3_4, %p3_5
  %s4_5 = v_add_f32 %s4_4, %p4_5
  %s5_5 = v_add_f32 %s5_4, %p5_5
  %p0_6 = v_mul_f32_dpp16 %q0, %k6 {dpp_ctrl = 342}
  %p1_6 = v_mul_f32_dpp16 %q1, %k6 {dpp_ctrl = 342}
  %p2_6 = v_mul_f32_dpp16 %q2, %k6 {dpp_ctrl = 342}
  %p3_6 = v_mul_f32_dpp16 %q3, %k6 {dpp_ctrl = 342}
  %p4_6 = v_mul_f32_dpp16 %q4, %k6 {dpp_ctrl = 342}
  %p5_6 = v_mul_f32_dpp16 %q5, %k6 {dpp_ctrl = 342}
  %s0_6 = v_add_f32 %s0_5, %p0_6
  %s1_6 = v_add_f32 %s1_5, %p1_6
  %s2_6 = v_add_f32 %s2_5, %p2_6
  %s3_6 = v_add_f32 %s3_5, %p3_6
  %s4_6 = v_add_f32 %s4_5, %p4_6
  %s5_6 = v_add_f32 %s5_5, %p5_6
  %p0_7 = v_mul_f32_dpp16 %q0, %k7 {dpp_ctrl = 343}
  %p1_7 = v_mul_f32_dpp16 %q1, %k7 {dpp_ctrl = 343}
  %p2_7 = v_mul_f32_dpp16 %q2, %k7 {dpp_ctrl = 343}
  %p3_7 = v_mul_f32_dpp16 %q3, %k7 {dpp_ctrl = 343}
  %p4_7 = v_mul_f32_dpp16 %q4, %k7 {dpp_ctrl = 343}
  %p5_7 = v_mul_f32_dpp16 %q5, %k7 {dpp_ctrl = 343}
  %s0_7 = v_add_f32 %s0_6, %p0_7
  %s1_7 = v_add_f32 %s1_6, %p1_7
  %s2_7 = v_add_f32 %s2_6, %p2_7
  %s3_7 = v_add_f32 %s3_6, %p3_7
  %s4_7 = v_add_f32 %s4_6, %p4_7
  %s5_7 = v_add_f32 %s5_6, %p5_7
  %p0_8 = v_mul_f32_dpp16 %q0, %k8 {dpp_ctrl = 344}
  %p1_8 = v_mul_f32_dpp16 %q1, %k8 {dpp_ctrl = 344}
  %p2_8 = v_mul_f32_dpp16 %q2, %k8 {dpp_ctrl = 344}
  %p3_8 = v_mul_f32_dpp16 %q3, %k8 {dpp_ctrl = 344}
  %p4_8 = v_mul_f32_dpp16 %q4, %k8 {dpp_ctrl = 344}
  %p5_8 = v_mul_f32_dpp16 %q5, %k8 {dpp_ctrl = 344}
  %s0_8 = v_add_f32 %s0_7, %p0_8
  %s1_8 = v_add_f32 %s1_7, %p1_8
  %s2_8 = v_add_f32 %s2_7, %p2_8
  %s3_8 = v_add_f32 %s3_7, %p3_8
  %s4_8 = v_add_f32 %s4_7, %p4_8
  %s5_8 = v_add_f32 %s5_7, %p5_8
  %p0_9 = v_mul_f32_dpp16 %q0, %k9 {dpp_ctrl = 345}
  %p1_9 = v_mul_f32_dpp16 %q1, %k9 {dpp_ctrl = 345}
  %p2_9 = v_mul_f32_dpp16 %q2, %k9 {dpp_ctrl = 345}
  %p3_9 = v_mul_f32_dpp16 %q3, %k9 {dpp_ctrl = 345}
  %p4_9 = v_mul_f32_dpp16 %q4, %k9 {dpp_ctrl = 345}
  %p5_9 = v_mul_f32_dpp16 %q5, %k9 {dpp_ctrl = 345}
  %s0_9 = v_add_f32 %s0_8, %p0_9
  %s1_9 = v_add_f32 %s1_8, %p1_9
  %s2_9 = v_add_f32 %s2_8, %p2_9
  %s3_9 = v_add_f32 %s3_8, %p3_9
  %s4_9 = v_add_f32 %s4_8, %p4_9
  %s5_9 = v_add_f32 %s5_8, %p5_9
  %p0_10 = v_mul_f32_dpp16 %q0, %k10 {dpp_ctrl = 346}
  %p1_10 = v_mul_f32_dpp16 %q1, %k10 {dpp_ctrl = 346}
  %p2_10 = v_mul_f32_dpp16 %q2, %k10 {dpp_ctrl = 346}
  %p3_10 = v_mul_f32_dpp16 %q3, %k10 {dpp_ctrl = 346}
  %p4_10 = v_mul_f32_dpp16 %q4, %k10 {dpp_ctrl = 346}
  %p5_10 = v_mul_f32_dpp16 %q5, %k10 {dpp_ctrl = 346}
  %s0_10 = v_add_f32 %s0_9, %p0_10
  %s1_10 = v_add_f32 %s1_9, %p1_10
  %s2_10 = v_add_f32 %s2_9, %p2_10
  %s3_10 = v_add_f32 %s3_9, %p3_10
  %s4_10 = v_add_f32 %s4_9, %p4_10
  %s5_10 = v_add_f32 %s5_9, %p5_10
  %p0_11 = v_mul_f32_dpp16 %q0, %k11 {dpp_ctrl = 347}
  %p1_11 = v_mul_f32_dpp16 %q1, %k11 {dpp_ctrl = 347}
  %p2_11 = v_mul_f32_dpp16 %q2, %k11 {dpp_ctrl = 347}
  %p3_11 = v_mul_f32_dpp16 %q3, %k11 {dpp_ctrl = 347}
  %p4_11 = v_mul_f32_dpp16 %q4, %k11 {dpp_ctrl = 347}
  %p5_11 = v_mul_f32_dpp16 %q5, %k11 {dpp_ctrl = 347}
  %s0_11 = v_add_f32 %s0_10, %p0_11
  %s1_11 = v_add_f32 %s1_10, %p1_11
  %s2_11 = v_add_f32 %s2_10, %p2_11
  %s3_11 = v_add_f32 %s3_10, %p3_11
  %s4_11 = v_add_f32 %s4_10, %p4_11
  %s5_11 = v_add_f32 %s5_10, %p5_11
  %p0_12 = v_mul_f32_dpp16 %q0, %k12 {dpp_ctrl = 348}
  %p1_12 = v_mul_f32_dpp16 %q1, %k12 {dpp_ctrl = 348}
  %p2_12 = v_mul_f32_dpp16 %q2, %k12 {dpp_ctrl = 348}
  %p3_12 = v_mul_f32_dpp16 %q3, %k12 {dpp_ctrl = 348}
  %p4_12 = v_mul_f32_dpp16 %q4, %k12 {dpp_ctrl = 348}
  %p5_12 = v_mul_f32_dpp16 %q5, %k12 {dpp_ctrl = 348}
  %s0_12 = v_add_f32 %s0_11, %p0_12
  %s1_12 = v_add_f32 %s1_11, %p1_12
  %s2_12 = v_add_f32 %s2_11, %p2_12
  %s3_12 = v_add_f32 %s3_11, %p3_12
  %s4_12 = v_add_f32 %s4_11, %p4_12
  %s5_12 = v_add_f32 %s5_11, %p5_12
  %p0_13 = v_mul_f32_dpp16 %q0, %k13 {dpp_ctrl = 349}
  %p1_13 = v_mul_f32_dpp16 %q1, %k13 {dpp_ctrl = 349}
  %p2_13 = v_mul_f32_dpp16 %q2, %k13 {dpp_ctrl = 349}
  %p3_13 = v_mul_f32_dpp16 %q3, %k13 {dpp_ctrl = 349}
  %p4_13 = v_mul_f32_dpp16 %q4, %k13 {dpp_ctrl = 349}
  %p5_13 = v_mul_f32_dpp16 %q5, %k13 {dpp_ctrl = 349}
  %s0_13 = v_add_f32 %s0_12, %p0_13
  %s1_13 = v_add_f32 %s1_12, %p1_13
  %s2_13 = v_add_f32 %s2_12, %p2_13
  %s3_13 = v_add_f32 %s3_12, %p3_13
  %s4_13 = v_add_f32 %s4_12, %p4_13
  %s5_13 = v_add_f32 %s5_12, %p5_13
  %p0_14 = v_mul_f32_dpp16 %q0, %k14 {dpp_ctrl = 350}
  %p1_14 = v_mul_f32_dpp16 %q1, %k14 {dpp_ctrl = 350}
  %p2_14 = v_mul_f32_dpp16 %q2, %k14 {dpp_ctrl = 350}
  %p3_14 = v_mul_f32_dpp16 %q3, %k14 {dpp_ctrl = 350}
  %p4_14 = v_mul_f32_dpp16 %q4, %k14 {dpp_ctrl = 350}
  %p5_14 = v_mul_f32_dpp16 %q5, %k14 {dpp_ctrl = 350}
  %s0_14 = v_add_f32 %s0_13, %p0_14
  %s1_14 = v_add_f32 %s1_13, %p1_14
  %s2_14 = v_add_f32 %s2_13, %p2_14
  %s3_14 = v_add_f32 %s3_13, %p3_14
  %s4_14 = v_add_f32 %s4_13, %p4_14
  %s5_14 = v_add_f32 %s5_13, %p5_14
  %p0_15 = v_mul_f32_dpp16 %q0, %k15 {dpp_ctrl = 351}
  %p1_15 = v_mul_f32_dpp16 %q1, %k15 {dpp_ctrl = 351}
  %p2_15 = v_mul_f32_dpp16 %q2, %k15 {dpp_ctrl = 351}
  %p3_15 = v_mul_f32_dpp16 %q3, %k15 {dpp_ctrl = 351}
  %p4_15 = v_mul_f32_dpp16 %q4, %k15 {dpp_ctrl = 351}
  %p5_15 = v_mul_f32_dpp16 %q5, %k15 {dpp_ctrl = 351}
  %s0_15 = v_add_f32 %s0_14, %p0_15
  %s1_15 = v_add_f32 %s1_14, %p1_15
  %s2_15 = v_add_f32 %s2_14, %p2_15
  %s3_15 = v_add_f32 %s3_14, %p3_15
  %s4_15 = v_add_f32 %s4_14, %p4_15
  %s5_15 = v_add_f32 %s5_14, %p5_15
  return %s0_15, %s1_15, %s2_15, %s3_15, %s4_15, %s5_15
})", acc, q, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15);
}

// One key's values for one head: o_i + p_j v_i, p_j read from lane j (the key's lane after the softmax step).
struct O8 {
  float a0, a1, a2, a3, a4, a5, a6, a7;
};

LOOM_INLINE O8 pv8(O8 o, float p, int j, float v0, float v1, float v2, float v3, float v4, float v5, float v6,
                   float v7) {
  return loom::low::assembly<Gfx11, O8>(R"((%o0: reg<amdgpu.vgpr>, %o1: reg<amdgpu.vgpr>, %o2: reg<amdgpu.vgpr>,
    %o3: reg<amdgpu.vgpr>,
    %o4: reg<amdgpu.vgpr>, %o5: reg<amdgpu.vgpr>, %o6: reg<amdgpu.vgpr>, %o7: reg<amdgpu.vgpr>,
    %p: reg<amdgpu.vgpr>, %j: reg<amdgpu.sgpr>, %v0: reg<amdgpu.vgpr>, %v1: reg<amdgpu.vgpr>,
    %v2: reg<amdgpu.vgpr>, %v3: reg<amdgpu.vgpr>, %v4: reg<amdgpu.vgpr>, %v5: reg<amdgpu.vgpr>,
    %v6: reg<amdgpu.vgpr>, %v7: reg<amdgpu.vgpr>)
    -> (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>,
    reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) {
  %s = v_readlane_b32_src1_sgpr %p, %j
  %m0 = v_mul_f32 %s, %v0
  %r0 = v_add_f32 %o0, %m0
  %m1 = v_mul_f32 %s, %v1
  %r1 = v_add_f32 %o1, %m1
  %m2 = v_mul_f32 %s, %v2
  %r2 = v_add_f32 %o2, %m2
  %m3 = v_mul_f32 %s, %v3
  %r3 = v_add_f32 %o3, %m3
  %m4 = v_mul_f32 %s, %v4
  %r4 = v_add_f32 %o4, %m4
  %m5 = v_mul_f32 %s, %v5
  %r5 = v_add_f32 %o5, %m5
  %m6 = v_mul_f32 %s, %v6
  %r6 = v_add_f32 %o6, %m6
  %m7 = v_mul_f32 %s, %v7
  %r7 = v_add_f32 %o7, %m7
  return %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7
})", o, p, j, v0, v1, v2, v3, v4, v5, v6, v7);
}

}  // namespace sa

#endif  // SA_LANES_ASM_H_
