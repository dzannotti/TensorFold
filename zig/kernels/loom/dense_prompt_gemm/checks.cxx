// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases: an fp32 oracle within tolerance, and row-count and cut byte identity (README.md).
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

#include "bf16_gemm.cxx"

namespace k5c {

static loom::kernel::configuration per_thread(unsigned n) { return {{(n + 255) / 256, 1, 1}, {256, 1, 1}}; }

// out[r][c] = sum over K of x[r][i] w[c][i], one fp32 fma chain in K order.
[[loom::kernel(per_thread)]]
void oracle(unsigned total, const __bf16* x, const __bf16* w, float* out, unsigned n, unsigned k) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < total) {
    const unsigned r = i / n, c = i % n;
    float s = 0.0f;
    for (unsigned j = 0; j < k; j++) s = loom::scalar::fmaf((float)x[r * k + j], (float)w[c * k + j], s);
    out[i] = s;
  }
}

template <class G>
static loom::kernel::configuration configure_from(unsigned m, unsigned n) {
  return {{(n + G::kCols - 1) / G::kCols, (m + G::kRows - 1) / G::kRows, 1}, {G::kThreads, 1, 1}};
}

// The kernel on rows row0 .. row0 + m: the rows of a later chunk.
[[loom::kernel(configure_from<G128x256>)]]
void gemm_from(const __bf16* x, const __bf16* w, float* out, int row0, int m, int n, int k) {
  LOOM_WORKGROUP __bf16 lds[G128x256::kArena];
  bf16_gemm<G128x256>(x + (unsigned)row0 * (unsigned)k, w, out + (unsigned)row0 * (unsigned)n, m, n, k, lds);
}

}  // namespace k5c

// M rows of x by an N x K weight matrix: against the oracle; rows cut at CUT (two launches) byte-equal to one launch.
#define K5C_CASE(NAME, M, N, K, CUT)                                                                              \
  LOOM_CHECK_CASE(NAME) {                                                                                         \
    constexpr unsigned m = M, n = N, k = K, outs = m * n;                                                         \
    const auto x = loom::check::iota<m * k, __bf16>((__bf16)-1.0f, (__bf16)0.0009765625f, 2048);                 \
    const auto w = loom::check::iota<n * k, __bf16>((__bf16)-0.5f, (__bf16)0.0009765625f, 1031);                 \
    const auto out = loom::check::fill<outs, float>(0.0f);                                                        \
    const auto out_small = loom::check::fill<outs, float>(0.0f);                                                  \
    const auto out_cut = loom::check::fill<outs, float>(0.0f);                                                    \
    const auto want = loom::check::fill<outs, float>(0.0f);                                                       \
    loom::check::launch<bf16_gemm_128x256>(loom::kernel::workload(m, n), x, w, out, (int)m, (int)n, (int)k);      \
    loom::check::launch<bf16_gemm_128x128>(loom::kernel::workload(m, n), x, w, out_small, (int)m, (int)n, (int)k); \
    constexpr unsigned cut = CUT;                                                                                 \
    loom::check::launch<k5c::gemm_from>(loom::kernel::workload(cut, n), x, w, out_cut, 0, (int)cut, (int)n, (int)k); \
    loom::check::launch<k5c::gemm_from>(loom::kernel::workload(m - cut, n), x, w, out_cut, (int)cut, (int)(m - cut),  \
                                        (int)n, (int)k);                                                          \
    loom::check::launch<k5c::oracle>(loom::kernel::workload(outs), outs, x, w, want, n, k);                       \
    loom::check::expect_bitwise(out_small, out);                                                                  \
    loom::check::expect_bitwise(out_cut, out);                                                                    \
    loom::check::expect_approximate(out, want, 0.001, 0.001);                                                     \
  }

K5C_CASE(rows_2_cut_1, 2, 324, 640, 1)
K5C_CASE(rows_37_k320, 37, 200, 320, 7)
K5C_CASE(rows_200_cut_129, 200, 324, 1280, 129)
K5C_CASE(rows_130_k10240, 130, 324, 10240, 64)

// Flash Next hyper-connection mixer shapes at 4,096 prompt rows (timings include the oracle launches).
K5C_CASE(hc_down_4096_rows, 4096, 324, 10240, 1)
K5C_CASE(hc_up_4096_rows, 4096, 10240, 320, 128)
LOOM_CHECK_BENCHMARK(bench_hc_down_4096_rows, hc_down_4096_rows);
LOOM_CHECK_BENCHMARK(bench_hc_up_4096_rows, hc_up_4096_rows);
