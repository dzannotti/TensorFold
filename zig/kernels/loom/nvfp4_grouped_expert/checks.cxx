// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases: fp32 oracle within tolerance, plan-cut byte equality (README: Test).
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

#include "down.cxx"
#include "gate_up.cxx"

namespace k4c {

static loom::kernel::configuration per_thread(unsigned n) { return {{(n + 255) / 256, 1, 1}, {256, 1, 1}}; }

// items[z] = {expert z % experts, first z * rows, rows}; with split, item z covers [0, cut) and item z + n [cut, rows).
[[loom::kernel(per_thread)]]
void make_plan(unsigned n, int* items, unsigned experts, unsigned rows, unsigned cut, unsigned split) {
  const unsigned z = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (z < n) {
    const unsigned e = z % experts;
    if (split) {
      items[3 * z] = (int)e;
      items[3 * z + 1] = (int)(z * rows);
      items[3 * z + 2] = (int)cut;
      items[3 * (z + n)] = (int)e;
      items[3 * (z + n) + 1] = (int)(z * rows + cut);
      items[3 * (z + n) + 2] = (int)(rows - cut);
    } else {
      items[3 * z] = (int)e;
      items[3 * z + 1] = (int)(z * rows);
      items[3 * z + 2] = (int)rows;
    }
  }
}

LOOM_INLINE float e2m1(unsigned c) {
  const unsigned m = c & 7;
  const float mag = (float)(1u << ((m >> 1) - 1)) * (1.0f + 0.5f * (float)(m & 1));
  const float v = (m >> 1) == 0 ? 0.5f * (float)(m & 1) : mag;
  return (c & 8) ? -v : v;
}

LOOM_INLINE float e4m3(unsigned s) {
  const unsigned e = (s >> 3) & 15, m = s & 7;
  const float v = e == 0 ? (float)m * 0.001953125f : __builtin_bit_cast(float, ((e + 120) << 23) | (m << 20));
  return (s & 0x80) ? -v : v;
}

// Weight (row, k) of a baked NVFP4 stack, exactly (k-th code: word k / 8, nibble (k % 8) / 2 + 4 (k % 2)).
LOOM_INLINE float weight(const unsigned* words, const unsigned char* scales, unsigned row, unsigned kk, unsigned k) {
  const unsigned w = words[row * (kk / 8) + k / 8], q = k % 8;
  const unsigned code = (w >> (4 * (q / 2) + (q & 1) * 16)) & 15;
  return e2m1(code) * e4m3(scales[row * (kk / 16) + k / 16]) * 0.00006103515625f;
}

LOOM_INLINE float bf(__bf16 v) { return (float)v; }

// act[p][f] for pairs p < pairs of one expert's rows (expert e, x row p / x_div).
[[loom::kernel(per_thread)]]
void oracle_gate_up(unsigned n, const __bf16* x, const unsigned* words, const unsigned char* scales,
                    const float* globals, unsigned rows, unsigned experts, unsigned x_div, float* act) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < n) {
    const unsigned p = i / kGuN, f = i % kGuN, expert = (p / rows) % experts;
    const unsigned* ew = words + expert * 2 * kGuN * (kGuK / 8);
    const unsigned char* es = scales + expert * 2 * kGuN * (kGuK / 16);
    float g = 0.0f, u = 0.0f;
    for (unsigned k = 0; k < kGuK; k++) {
      const float xv = bf(x[(p / x_div) * kGuK + k]);
      g = loom::scalar::fmaf(xv, weight(ew, es, f, kGuK, k), g);
      u = loom::scalar::fmaf(xv, weight(ew, es, kGuN + f, kGuK, k), u);
    }
    g *= globals[2 * expert] * 16384.0f;
    u *= globals[2 * expert + 1] * 16384.0f;
    act[i] = g / (1.0f + k4::exp_neg(g)) * u;
  }
}


// out[p][c] for pairs p of the plan make_plan builds (item p / rows, expert item % experts).
[[loom::kernel(per_thread)]]
void oracle_down(unsigned n, const __bf16* x, const unsigned* words, const unsigned char* scales,
                 const float* globals, unsigned rows, unsigned experts, float* out) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < n) {
    const unsigned p = i / kDnN, c = i % kDnN, expert = (p / rows) % experts;
    const unsigned* ew = words + expert * kDnN * (kDnK / 8);
    const unsigned char* es = scales + expert * kDnN * (kDnK / 16);
    float s = 0.0f;
    for (unsigned k = 0; k < kDnK; k++) s = loom::scalar::fmaf(bf(x[p * kDnK + k]), weight(ew, es, c, kDnK, k), s);
    out[i] = s * globals[expert] * 16384.0f;
  }
}

}  // namespace k4c

// Generated inputs for E experts of ROWS rows of K inputs.
#define K4C_WEIGHTS(E, ROWS, K)                                                                       \
  const auto words = loom::check::iota<(E) * (ROWS) * ((K) / 8), unsigned>(0u, 0x0789abcdu, 17);     \
  const auto scales = loom::check::iota<(E) * (ROWS) * ((K) / 16), unsigned char>(0x30, 1, 16);       \
  const auto globals = loom::check::fill<(E) * 2, float>(0x1p-14f);

// The plan: ITEMS items of R rows (expert z % E) as one item each, and the same rows cut at CUT into two items.
#define K4C_PLAN(E, ITEMS, R, CUT)                                                                    \
  constexpr unsigned items_n = ITEMS, experts = E, rows = R, cut = CUT, pairs = items_n * rows;      \
  constexpr unsigned blocks = (rows + 127) / 128;                                                     \
  const auto members = loom::check::iota<pairs, int>(0, 1);                                          \
  const auto items = loom::check::fill<3 * items_n, int>(0);                                          \
  const auto split = loom::check::fill<6 * items_n, int>(0);                                          \
  loom::check::launch<k4c::make_plan>(loom::kernel::workload(items_n), items_n, items, experts, rows, cut, 0u); \
  loom::check::launch<k4c::make_plan>(loom::kernel::workload(items_n), items_n, split, experts, rows, cut, 1u);

// gate|up (x row = pair / 2) against the oracle and against the cut plan.
#define K4C_GATE_UP(NAME, E, ITEMS, R, CUT)                                                                \
  LOOM_CHECK_CASE(NAME) {                                                                                  \
    K4C_PLAN(E, ITEMS, R, CUT)                                                                             \
    constexpr unsigned tokens = (pairs + 1) / 2, outs = pairs * kGuN, two = 2;                             \
    K4C_WEIGHTS(E, 2 * kGuN, kGuK)                                                                         \
    const auto x = loom::check::iota<tokens * kGuK, __bf16>((__bf16)-1.0f, (__bf16)0.0009765625f, 2048);   \
    const auto act = loom::check::fill<outs, __bf16>((__bf16)0.0f);                                        \
    const auto act_cut = loom::check::fill<outs, __bf16>((__bf16)0.0f);                                    \
    const auto want = loom::check::fill<outs, float>(0.0f);                                                \
    loom::check::launch<nvfp4_expert_gate_up>(loom::kernel::workload(items_n, blocks), x, words, scales,   \
                                              globals, items, members, act, 2, (int)kGuN, (int)kGuK);      \
    loom::check::launch<nvfp4_expert_gate_up>(loom::kernel::workload(2 * items_n, blocks), x, words,       \
                                              scales, globals, split, members, act_cut, 2, (int)kGuN,      \
                                              (int)kGuK);                                                  \
    loom::check::launch<k4c::oracle_gate_up>(loom::kernel::workload(outs), outs, x, words, scales,         \
                                             globals, rows, experts, two, want);                           \
    loom::check::expect_bitwise(act_cut, act);                                                             \
    loom::check::expect_approximate(act, want, 0.05, 0.016);                                               \
  }

// down (x row = pair) against the oracle and against the cut plan.
#define K4C_DOWN(NAME, E, ITEMS, R, CUT)                                                                   \
  LOOM_CHECK_CASE(NAME) {                                                                                  \
    K4C_PLAN(E, ITEMS, R, CUT)                                                                             \
    constexpr unsigned outs = pairs * kDnN;                                                                \
    K4C_WEIGHTS(E, kDnN, kDnK)                                                                             \
    const auto x = loom::check::iota<pairs * kDnK, __bf16>((__bf16)-1.0f, (__bf16)0.0009765625f, 2048);    \
    const auto out = loom::check::fill<outs, float>(0.0f);                                                 \
    const auto out_cut = loom::check::fill<outs, float>(0.0f);                                             \
    const auto want = loom::check::fill<outs, float>(0.0f);                                                \
    loom::check::launch<nvfp4_expert_down>(loom::kernel::workload(items_n, blocks), x, words, scales,      \
                                           globals, items, members, out, 1, (int)kDnN, (int)kDnK);         \
    loom::check::launch<nvfp4_expert_down>(loom::kernel::workload(2 * items_n, blocks), x, words, scales,  \
                                           globals, split, members, out_cut, 1, (int)kDnN, (int)kDnK);     \
    loom::check::launch<k4c::oracle_down>(loom::kernel::workload(outs), outs, x, words, scales, globals,   \
                                          rows, experts, want);                                            \
    loom::check::expect_bitwise(out_cut, out);                                                             \
    loom::check::expect_approximate(out, want, 0.002, 0.001);                                              \
  }

K4C_GATE_UP(gate_up_1_row, 2, 3, 1, 0)
K4C_GATE_UP(gate_up_37_rows, 2, 3, 37, 17)
K4C_GATE_UP(gate_up_128_rows, 3, 3, 128, 64)
K4C_GATE_UP(gate_up_200_rows, 2, 2, 200, 129)
K4C_DOWN(down_1_row, 2, 3, 1, 0)
K4C_DOWN(down_37_rows, 2, 3, 37, 17)
K4C_DOWN(down_128_rows, 3, 3, 128, 64)
K4C_DOWN(down_200_rows, 2, 2, 200, 129)

// Flash Next prompt shapes (4,096 tokens x top-10 over 512 experts); timings include the plan and oracle launches.
K4C_GATE_UP(gate_up_4096_tokens, 512, 512, 80, 40)
K4C_DOWN(down_4096_tokens, 512, 512, 80, 40)
LOOM_CHECK_BENCHMARK(bench_gate_up_4096_tokens, gate_up_4096_tokens);
LOOM_CHECK_BENCHMARK(bench_down_4096_tokens, down_4096_tokens);
