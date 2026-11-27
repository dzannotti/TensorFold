// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases for the hyper-connection read-out (README.md): fp32 oracles within tolerance,
// tile and row-cut byte identity, the Flash Next shapes at 4,096 rows.
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

#include "hc_norm.cxx"
#include "hc_readout.cxx"

namespace hcc {

constexpr unsigned kD = k5n::kDims, kW = k5::kStreams * kD, kLow = 320;
constexpr float kEps = 1e-6f;

static loom::kernel::configuration per_thread(unsigned n) { return {{(n + 255) / 256, 1, 1}, {256, 1, 1}}; }
static loom::kernel::configuration norm_rows(unsigned m) { return {{m, 1, 1}, {k5n::kThreads, 1, 1}}; }
static loom::kernel::configuration down_rows(unsigned m) { return {{2, (m + 127) / 128, 1}, {512, 1, 1}}; }
static loom::kernel::configuration up_rows(unsigned m) { return {{kW / 256, (m + 127) / 128, 1}, {512, 1, 1}}; }

LOOM_INLINE float wide(const __bf16* p, unsigned i) { return k5::bits((unsigned)((const unsigned short*)p)[i] << 16); }

// Thread (row, stream): the writeback as the kernel's, then x from a plain fp32 sum of squares in dimension order.
[[loom::kernel(per_thread)]]
void oracle_norm(unsigned total, __bf16* h, const __bf16* branch, const __bf16* inject, const float* scale, __bf16* x,
                 int add) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < total) {
    const unsigned r = i / k5::kStreams, s = i % k5::kStreams, base = i * kD;
    unsigned short* const h16 = (unsigned short*)h;
    const float inj = wide(inject, i);
    float sum = 0.0f;
    for (unsigned d = 0; d < kD; d++) {
      float v = wide(h, base + d);
      if (add != 0) {
        v = k5::bf16_round(v + k5::bf16_round(wide(branch, r * kD + d) * inj));
        h16[base + d] = (unsigned short)k5::bf16_bits(v);
      }
      sum = loom::scalar::fmaf(v, v, sum);
    }
    const float rinv = 1.0f / loom::scalar::sqrtf(sum / (float)kD + kEps);
    unsigned short* const x16 = (unsigned short*)x;
    for (unsigned d = 0; d < kD; d++) x16[base + d] = (unsigned short)k5::bf16_bits(wide(h, base + d) * rinv * scale[s * kD + d]);
  }
}

// Thread (row, column): z = x . w in K order, then the gates.
[[loom::kernel(per_thread)]]
void oracle_down(unsigned total, const __bf16* x, const __bf16* w, __bf16* act, __bf16* inject, unsigned n) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < total) {
    const unsigned r = i / n, c = i % n;
    float z = 0.0f;
    for (unsigned j = 0; j < kW; j++) z = loom::scalar::fmaf(wide(x, r * kW + j), wide(w, c * kW + j), z);
    const unsigned short g = (unsigned short)hc_gate(k5::bf16_bits(z), c < kLow);
    if (c < kLow) ((unsigned short*)act)[r * kLow + c] = g;
    else ((unsigned short*)inject)[r * k5::kStreams + c - kLow] = g;
  }
}

// Thread (row, dimension): each stream's up sum in K order, gated onto x, the streams' mean.
[[loom::kernel(per_thread)]]
void oracle_up(unsigned total, const __bf16* act, const __bf16* w, const __bf16* x, __bf16* mixed) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < total) {
    const unsigned r = i / kD, d = i % kD;
    float mix = 0.0f;
    for (unsigned s = 0; s < k5::kStreams; s++) {
      float g = 0.0f;
      for (unsigned j = 0; j < kLow; j++) g = loom::scalar::fmaf(wide(act, r * kLow + j), wide(w, (s * kD + d) * kLow + j), g);
      mix += hc_term(k5::bf16_bits(g) << 16, k5::bf16_bits(wide(x, (r * k5::kStreams + s) * kD + d)) << 16);
    }
    ((unsigned short*)mixed)[i] = (unsigned short)k5::bf16_bits(mix * 0.25f);
  }
}

// The kernels on rows row0 .. row0 + m: the rows of a later chunk.
[[loom::kernel(norm_rows)]]
void norm_from(__bf16* h, const __bf16* branch, const __bf16* inject, const float* scale, __bf16* x, int row0, int m,
               int add) {
  LOOM_WORKGROUP float part[k5::kStreams * k5n::kVirtual];
  LOOM_WORKGROUP float waves[k5::kStreams * k5n::kWaves];
  const unsigned o = (unsigned)row0;
  hc_norm_rows(h + o * kW, branch + o * kD, inject + o * k5::kStreams, scale, x + o * kW, m, (int)kD, kEps, add, part,
               waves);
}

[[loom::kernel(down_rows)]]
void down_from(const __bf16* x, const __bf16* w, __bf16* act, __bf16* inject, int row0, int m, int n) {
  LOOM_WORKGROUP __bf16 lds[arena<Hc128x256>()];
  const unsigned o = (unsigned)row0;
  hc_down_wide_rows(x + o * kW, w, act + o * kLow, inject + o * k5::kStreams, m, n, (int)kW, (int)kLow, lds);
}

[[loom::kernel(up_rows)]]
void up_from(const __bf16* act, const __bf16* w, const __bf16* x, __bf16* mixed, int row0, int m) {
  LOOM_WORKGROUP __bf16 lds[arena<Hc128x256>()];
  const unsigned o = (unsigned)row0;
  hc_up_mix_tile<Hc128x256>(act + o * kLow, w, x + o * kW, mixed + o * kD, m, (int)kD, (int)kLow, lds);
}

}  // namespace hcc

#define HCC_BF16(N, LO, STEP, PERIOD) loom::check::iota<N, __bf16>((__bf16)(LO), (__bf16)(STEP), PERIOD)

// M rows' writeback (ADD) and norm: h against the oracle's bytes, x within tolerance; rows cut at CUT byte-equal.
#define HCC_NORM(NAME, M, ADD, CUT)                                                                                 \
  LOOM_CHECK_CASE(NAME) {                                                                                           \
    constexpr unsigned m = M, cut = CUT;                                                                            \
    const auto h = HCC_BF16(m * hcc::kW, -1.0f, 0.0009765625f, 2053);                                              \
    const auto h_cut = HCC_BF16(m * hcc::kW, -1.0f, 0.0009765625f, 2053);                                          \
    const auto h_want = HCC_BF16(m * hcc::kW, -1.0f, 0.0009765625f, 2053);                                         \
    const auto branch = HCC_BF16(m * hcc::kD, -0.5f, 0.001953125f, 509);                                           \
    const auto inject = HCC_BF16(m * k5::kStreams, 0.25f, 0.0625f, 29);                                            \
    const auto scale = loom::check::iota<hcc::kW, float>(0.25f, 0.0009765625f, 1531);                              \
    const auto x = loom::check::fill<m * hcc::kW, __bf16>((__bf16)0.0f);                                           \
    const auto x_cut = loom::check::fill<m * hcc::kW, __bf16>((__bf16)0.0f);                                       \
    const auto x_want = loom::check::fill<m * hcc::kW, __bf16>((__bf16)0.0f);                                      \
    loom::check::launch<hc_norm>(loom::kernel::workload(m), h, branch, inject, scale, x, (int)m, (int)hcc::kD,       \
                                 hcc::kEps, ADD);                                                                   \
    loom::check::launch<hcc::norm_from>(loom::kernel::workload(cut), h_cut, branch, inject, scale, x_cut, 0, (int)cut, ADD); \
    loom::check::launch<hcc::norm_from>(loom::kernel::workload(m - cut), h_cut, branch, inject, scale, x_cut,       \
                                        (int)cut, (int)(m - cut), ADD);                                             \
    loom::check::launch<hcc::oracle_norm>(loom::kernel::workload(m * k5::kStreams), m * k5::kStreams, h_want,     \
                                          branch, inject, scale, x_want, ADD);                                      \
    loom::check::expect_bitwise(h, h_want);                                                                         \
    loom::check::expect_bitwise(h_cut, h);                                                                          \
    loom::check::expect_bitwise(x_cut, x);                                                                          \
    loom::check::expect_approximate(x, x_want, 0.02, 0.01);                                                         \
  }

// M rows by the N x 10240 down weights: act (and inject) within tolerance of the oracle; both tiles and the cut
// at CUT byte-equal.
#define HCC_DOWN(NAME, M, N, CUT)                                                                                   \
  LOOM_CHECK_CASE(NAME) {                                                                                           \
    constexpr unsigned m = M, n = N, cut = CUT;                                                                     \
    const auto x = HCC_BF16(m * hcc::kW, -1.0f, 0.0009765625f, 2048);                                              \
    const auto w = HCC_BF16(n * hcc::kW, -0.03125f, 0.0000305176f, 2039);                                          \
    const auto act = loom::check::fill<m * hcc::kLow, __bf16>((__bf16)0.0f);                                       \
    const auto act_narrow = loom::check::fill<m * hcc::kLow, __bf16>((__bf16)0.0f);                                \
    const auto act_cut = loom::check::fill<m * hcc::kLow, __bf16>((__bf16)0.0f);                                   \
    const auto act_want = loom::check::fill<m * hcc::kLow, __bf16>((__bf16)0.0f);                                  \
    const auto inj = loom::check::fill<m * k5::kStreams, __bf16>((__bf16)0.0f);                                    \
    const auto inj_narrow = loom::check::fill<m * k5::kStreams, __bf16>((__bf16)0.0f);                             \
    const auto inj_cut = loom::check::fill<m * k5::kStreams, __bf16>((__bf16)0.0f);                                \
    const auto inj_want = loom::check::fill<m * k5::kStreams, __bf16>((__bf16)0.0f);                               \
    loom::check::launch<hc_down_wide>(loom::kernel::workload(m), x, w, act, inj, (int)m, (int)n, (int)hcc::kW,       \
                                      (int)hcc::kLow);                                                              \
    loom::check::launch<hc_down_narrow>(loom::kernel::workload(m, n), x, w, act_narrow, inj_narrow, (int)m, (int)n, \
                                        (int)hcc::kW, (int)hcc::kLow);                                              \
    loom::check::launch<hcc::down_from>(loom::kernel::workload(cut), x, w, act_cut, inj_cut, 0, (int)cut, (int)n);   \
    loom::check::launch<hcc::down_from>(loom::kernel::workload(m - cut), x, w, act_cut, inj_cut, (int)cut,          \
                                        (int)(m - cut), (int)n);                                                    \
    loom::check::launch<hcc::oracle_down>(loom::kernel::workload(m * n), m * n, x, w, act_want, inj_want, n);       \
    loom::check::expect_bitwise(act_narrow, act);                                                                   \
    loom::check::expect_bitwise(act_cut, act);                                                                      \
    loom::check::expect_bitwise(inj_narrow, inj);                                                                   \
    loom::check::expect_bitwise(inj_cut, inj);                                                                      \
    loom::check::expect_approximate(act, act_want, 0.01, 0.01);                                                     \
    loom::check::expect_approximate(inj, inj_want, 0.02, 0.01);                                                     \
  }

// M rows' up and mix: mixed within tolerance of the oracle; the cut at CUT byte-equal.
#define HCC_UP(NAME, M, CUT)                                                                                        \
  LOOM_CHECK_CASE(NAME) {                                                                                           \
    constexpr unsigned m = M, cut = CUT;                                                                            \
    const auto act = HCC_BF16(m * hcc::kLow, -0.5f, 0.0009765625f, 1021);                                          \
    const auto w = HCC_BF16(hcc::kW * hcc::kLow, -0.125f, 0.0001220703f, 2047);                                    \
    const auto x = HCC_BF16(m * hcc::kW, -1.0f, 0.0009765625f, 2053);                                              \
    const auto mixed = loom::check::fill<m * hcc::kD, __bf16>((__bf16)0.0f);                                       \
    const auto mixed_cut = loom::check::fill<m * hcc::kD, __bf16>((__bf16)0.0f);                                   \
    const auto mixed_want = loom::check::fill<m * hcc::kD, __bf16>((__bf16)0.0f);                                  \
    loom::check::launch<hc_up_mix>(loom::kernel::workload(m, hcc::kW), act, w, x, mixed, (int)m, (int)hcc::kD,      \
                                   (int)hcc::kLow);                                                                 \
    loom::check::launch<hcc::up_from>(loom::kernel::workload(cut), act, w, x, mixed_cut, 0, (int)cut);              \
    loom::check::launch<hcc::up_from>(loom::kernel::workload(m - cut), act, w, x, mixed_cut, (int)cut, (int)(m - cut)); \
    loom::check::launch<hcc::oracle_up>(loom::kernel::workload(m * hcc::kD), m * hcc::kD, act, w, x, mixed_want);   \
    loom::check::expect_bitwise(mixed_cut, mixed);                                                                  \
    loom::check::expect_approximate(mixed, mixed_want, 0.01, 0.01);                                                 \
  }

HCC_NORM(norm_rows_37_cut_5, 37, 1, 5)
HCC_NORM(norm_rows_9_no_branch, 9, 0, 4)
HCC_DOWN(down_rows_200_inject_cut_129, 200, 324, 129)
HCC_DOWN(down_rows_70_cut_7, 70, 320, 7)
HCC_UP(up_rows_130_cut_64, 130, 64)

// Flash Next shapes at 4,096 prompt rows (timings include the oracle launches).
HCC_NORM(hc_norm_4096_rows, 4096, 1, 1)
HCC_DOWN(hc_down_4096_rows, 4096, 324, 1)
HCC_UP(hc_up_mix_4096_rows, 4096, 128)
LOOM_CHECK_BENCHMARK(bench_hc_norm_4096_rows, hc_norm_4096_rows);
LOOM_CHECK_BENCHMARK(bench_hc_down_4096_rows, hc_down_4096_rows);
LOOM_CHECK_BENCHMARK(bench_hc_up_mix_4096_rows, hc_up_mix_4096_rows);
