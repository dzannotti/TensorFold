// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases: gdn_recur's outputs and final state byte-equal to the reference scan (oracle.h).
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

#include "gdn_recur.cxx"
#include "oracle.h"

// Flash Next's DeltaNet heads (16 key, 48 value, 128 x 128): unit-scale q and k, gates in (0.5, 1), betas in (0, 1).
#define GDN_INPUTS(LEN, SEED)                                                                                        \
  constexpr unsigned len = LEN, kh = 16, vh = 48, nq = len * kh * 128, nv = len * vh * 128, ng = len * vh;           \
  constexpr unsigned ns = vh * 128 * 128;                                                                            \
  const auto q = loom::check::fill<nq, float>(0.0f);                                                                 \
  const auto k = loom::check::fill<nq, float>(0.0f);                                                                 \
  const auto v = loom::check::fill<nv, float>(0.0f);                                                                 \
  const auto gate = loom::check::fill<ng, float>(0.0f);                                                              \
  const auto beta = loom::check::fill<ng, float>(0.0f);                                                              \
  const auto state = loom::check::fill<ns, float>(0.0f);                                                             \
  const auto want_state = loom::check::fill<ns, float>(0.0f);                                                        \
  const auto y = loom::check::fill<nv, float>(0.0f);                                                                 \
  const auto want = loom::check::fill<nv, float>(0.0f);                                                              \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(nq), nq, q, (unsigned)(SEED), -0.016f, 0.016f);      \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(nq), nq, k, (unsigned)(SEED) + 1u, -0.15f, 0.15f);   \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(nv), nv, v, (unsigned)(SEED) + 2u, -1.0f, 1.0f);     \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(ng), ng, gate, (unsigned)(SEED) + 3u, 0.5f, 1.0f);   \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(ng), ng, beta, (unsigned)(SEED) + 4u, 0.0f, 1.0f);   \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(ns), ns, state, (unsigned)(SEED) + 5u, -0.05f, 0.05f); \
  loom::check::launch<gdno::gen_uniform>(loom::kernel::workload(ns), ns, want_state, (unsigned)(SEED) + 5u, -0.05f,  \
                                         0.05f);

#define GDN_CASE(NAME, LEN, SEED)                                                                                    \
  LOOM_CHECK_CASE(NAME) {                                                                                            \
    GDN_INPUTS(LEN, SEED)                                                                                            \
    loom::check::launch<gdn_recur>(loom::kernel::workload(vh), q, k, v, gate, beta, state, y, (int)len, (int)kh,     \
                                   (int)vh);                                                                         \
    loom::check::launch<gdno::oracle_scan>(loom::kernel::workload(vh), q, k, v, gate, beta, want_state, want,        \
                                           (int)len, (int)kh, (int)vh);                                              \
    loom::check::expect_bitwise(y, want);                                                                            \
    loom::check::expect_bitwise(state, want_state);                                                                  \
  }

#define GDN_RUN \
  loom::check::launch<gdn_recur>(loom::kernel::workload(vh), q, k, v, gate, beta, state, y, (int)len, (int)kh, (int)vh);
#define GDN_RUN4 GDN_RUN GDN_RUN GDN_RUN GDN_RUN

// Timed cases: one launch and 17 carrying the state on; their difference over 16 is a launch alone.
#define GDN_BENCH_CASE(NAME, LEN)                                                                                    \
  LOOM_CHECK_CASE(NAME) {                                                                                            \
    GDN_INPUTS(LEN, 7)                                                                                               \
    GDN_RUN                                                                                                          \
  }                                                                                                                  \
  LOOM_CHECK_CASE(NAME##_x17) {                                                                                      \
    GDN_INPUTS(LEN, 7)                                                                                               \
    GDN_RUN GDN_RUN4 GDN_RUN4 GDN_RUN4 GDN_RUN4                                                                      \
  }

GDN_CASE(one_row, 1, 1)
GDN_CASE(steps_8, 8, 2)
GDN_CASE(steps_9, 9, 3)
GDN_CASE(steps_17, 17, 6)
GDN_CASE(rows_100, 100, 4)
GDN_CASE(rows_333, 333, 5)

GDN_BENCH_CASE(rows_4096, 4096)
LOOM_CHECK_BENCHMARK(bench_rows_4096, rows_4096);
LOOM_CHECK_BENCHMARK(bench_rows_4096_x17, rows_4096_x17);
