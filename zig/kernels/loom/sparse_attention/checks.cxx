// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases: the indexer byte-equal to its serial replica, attention within CUDA's bf16 bound of
// fp64 and byte-equal on a repeat (oracle.h); row-split and single-row equality are TensorFold's kernel tests.
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

#include "attend.cxx"
#include "oracle.h"
#include "scores.cxx"
#include "select.cxx"

// Attention of NROWS rows (tensors or slices at their first row) into OUT.
#define SA_ATTEND(Q, IDS, CNT, SP, OUT, NROWS)                                                                         \
  loom::check::launch<sparse_attend>(loom::kernel::workload(2u, NROWS), Q, k, v, IDS, CNT, SP, OUT,                     \
                                     (long long)key_stride, (long long)cap, 24, 2, 256, 0.0625f, 4);

// Flash Next prompt rows CTX - ROWS .. CTX - 1 (24 query heads, 2 KV heads, head_dim 256, indexer 4 x 128, ratio 4,
// top 512 blocks): scores, selection and attention, each against its oracle on the previous stage's output.
#define SA_INPUTS(CTX, ROWS, SEED)                                                                                    \
  constexpr unsigned rows = ROWS, cap = ((CTX) + 63) / 64 * 64, nb = (CTX) / 4 > 0 ? (CTX) / 4 : 1;                    \
  constexpr unsigned key_stride = 512 * 4 + 4, nq = rows * 24 * 256, nkv = 2 * cap * 256;                              \
  constexpr unsigned niq = rows * 4 * 128, npool = cap / 4 * 128, nscore = rows * nb, nkeys = rows * key_stride;       \
  const auto q = loom::check::fill<nq, float>(0.0f);                                                                    \
  const auto k = loom::check::fill<nkv, __bf16>((__bf16)0.0f);                                                          \
  const auto v = loom::check::fill<nkv, __bf16>((__bf16)0.0f);                                                          \
  const auto iq = loom::check::fill<niq, float>(0.0f);                                                                  \
  const auto pooled = loom::check::fill<npool, __bf16>((__bf16)0.0f);                                                   \
  const auto ends = loom::check::fill<rows, int>(0);                                                                    \
  const auto complete = loom::check::fill<rows, int>(0);                                                                \
  loom::check::launch<sao::gen_f32>(loom::kernel::workload(nq), nq, q, (unsigned)(SEED), 3.0f);                        \
  loom::check::launch<sao::gen_bf16>(loom::kernel::workload(nkv), nkv, k, (unsigned)(SEED) + 1u, 1.0f);                \
  loom::check::launch<sao::gen_bf16>(loom::kernel::workload(nkv), nkv, v, (unsigned)(SEED) + 2u, 1.0f);                \
  loom::check::launch<sao::gen_f32>(loom::kernel::workload(niq), niq, iq, (unsigned)(SEED) + 3u, 1.0f);                \
  loom::check::launch<sao::gen_bf16>(loom::kernel::workload(npool), npool, pooled, (unsigned)(SEED) + 4u, 1.0f);       \
  loom::check::launch<sao::gen_rows>(loom::kernel::workload(rows), rows, ends, complete, (int)(CTX) - (int)rows, 4);   \
  const auto scores = loom::check::fill<nscore, float>(0.0f);                                                           \
  const auto ids = loom::check::fill<nkeys, int>(0);                                                                    \
  const auto count = loom::check::fill<rows, int>(0);                                                                   \
  const auto sparse = loom::check::fill<rows, int>(0);                                                                  \
  const auto out = loom::check::fill<nq, float>(0.0f);                                                                  \
  loom::check::launch<idx_scores>(loom::kernel::workload(nb, rows), iq, pooled, complete, scores, (long long)nb,       \
                                  (int)rows, 4, 128, (int)nb, 512);                                                     \
  loom::check::launch<idx_select>(loom::kernel::workload(rows), scores, complete, ends, ids, count, sparse,             \
                                  (long long)nb, (long long)key_stride, 512, 4);                                        \
  SA_ATTEND(q, ids, count, sparse, out, rows)

#define SA_CASE(NAME, CTX, ROWS, SEED)                                                                                \
  LOOM_CHECK_CASE(NAME) {                                                                                               \
    SA_INPUTS(CTX, ROWS, SEED)                                                                                          \
    const auto want_scores = loom::check::fill<nscore, float>(0.0f);                                                    \
    const auto taken = loom::check::fill<nscore, int>(0);                                                               \
    const auto want_ids = loom::check::fill<nkeys, int>(0);                                                             \
    const auto want_count = loom::check::fill<rows, int>(0);                                                            \
    const auto want_sparse = loom::check::fill<rows, int>(0);                                                           \
    const auto want = loom::check::fill<nq, float>(0.0f);                                                               \
    loom::check::launch<sao::oracle_scores>(loom::kernel::workload(nscore), nscore, iq, pooled, complete, want_scores,  \
                                            nb, 512);                                                                   \
    loom::check::launch<sao::oracle_take>(loom::kernel::workload(nscore), nscore, scores, complete, taken, nb, 512);    \
    loom::check::launch<sao::oracle_place>(loom::kernel::workload(nscore), nscore, taken, complete, ends, want_ids,     \
                                           want_count, want_sparse, nb, key_stride, 512, 4);                            \
    loom::check::launch<sao::oracle_attend>(loom::kernel::workload(rows * 24), rows * 24, q, k, v, ids, count, sparse, \
                                            want, (long long)key_stride, (long long)cap, 0.0625f);                       \
    loom::check::expect_bitwise(scores, want_scores);                                                                   \
    loom::check::expect_bitwise(ids, want_ids);                                                                         \
    loom::check::expect_bitwise(count, want_count);                                                                     \
    loom::check::expect_bitwise(sparse, want_sparse);                                                                   \
    loom::check::expect_approximate(out, want, 8e-2, 1.0 / 64);                                                         \
  }

// Timed cases: the four kernels on TensorFold's 64-row prompt slice; one more attention launch must repeat its bytes.
#define SA_BENCH_CASE(NAME, CTX, ROWS)                                                                                \
  LOOM_CHECK_CASE(NAME) {                                                                                               \
    SA_INPUTS(CTX, ROWS, 7)                                                                                             \
    const auto again = loom::check::fill<nq, float>(0.0f);                                                              \
    SA_ATTEND(q, ids, count, sparse, again, rows)                                                                       \
    loom::check::expect_bitwise(again, out);                                                                            \
  }

SA_CASE(dense_first_rows, 64, 8, 1)
SA_CASE(dense_700, 700, 5, 2)
SA_CASE(crossing_2048_keys, 2080, 8, 3)
SA_CASE(sparse_4k, 4096, 3, 4)
SA_CASE(sparse_9k_tail, 9001, 2, 5)

SA_BENCH_CASE(rows64_4k, 4096, 64)
SA_BENCH_CASE(rows64_16k, 16384, 64)
SA_BENCH_CASE(rows64_31k, 31506, 64)
SA_BENCH_CASE(rows64_64k, 65536, 64)
LOOM_CHECK_BENCHMARK(bench_rows64_4k, rows64_4k);
LOOM_CHECK_BENCHMARK(bench_rows64_16k, rows64_16k);
LOOM_CHECK_BENCHMARK(bench_rows64_31k, rows64_31k);
LOOM_CHECK_BENCHMARK(bench_rows64_64k, rows64_64k);
