// SPDX-License-Identifier: MIT

// Indexer block scores: sum over heads of relu(q . key) / sqrt(128), in tf_fn_idx_scores' exact sum order.
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#ifndef SA_SCORE_ROWS
#define SA_SCORE_ROWS 16
#endif

constexpr unsigned kIdxDims = 128, kIdxHeads = 4, kScoreRows = SA_SCORE_ROWS, kScoreBlocks = 256;

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

static loom::kernel::configuration configure_scores(unsigned blocks, unsigned rows) {
  return {{(blocks + kScoreBlocks - 1) / kScoreBlocks, (rows + kScoreRows - 1) / kScoreRows, 1}, {kScoreBlocks, 1, 1}};
}

namespace sa {

LOOM_INLINE void score_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

}  // namespace sa

// A work-item a block, kScoreRows rows a workgroup; writes scores[r][b] only where complete[r] > top and b < it.
[[loom::kernel(configure_scores)]]
void idx_scores(const float* q, const __bf16* pooled, const int* complete, float* scores, long long stride, int rows,
                int heads, int dims, int blocks, int top) {
  if (heads != (int)kIdxHeads || dims != (int)kIdxDims) return;
  LOOM_WORKGROUP float qs[kScoreRows * kIdxHeads * kIdxDims];
  const unsigned tid = loom::kernel::workitem::id.x;
  const int b = (int)(loom::kernel::workgroup::id.x * kScoreBlocks + tid);
  const int r0 = (int)(loom::kernel::workgroup::id.y * kScoreRows);
  const int rn = rows - r0 < (int)kScoreRows ? rows - r0 : (int)kScoreRows;
  const float* qrow = q + (long long)r0 * kIdxHeads * kIdxDims;
  for (int i = (int)tid; i < rn * (int)(kIdxHeads * kIdxDims); i += (int)kScoreBlocks) qs[i] = qrow[i];
  sa::score_barrier();
  int most = 0;
  for (int r = 0; r < rn; r++) {
    const int c = complete[r0 + r];
    most = c > top && c > most ? c : most;
  }
  if (b >= blocks || b >= most) return;
  const u32x4* krow = (const u32x4*)(pooled + (long long)b * kIdxDims);
  float key[kIdxDims];
  [[loom::unroll]] for (unsigned c = 0; c < kIdxDims / 8; c++) {
    const u32x4 w = krow[c];
    key[8 * c] = __builtin_bit_cast(float, w.x << 16);
    key[8 * c + 1] = __builtin_bit_cast(float, w.x & 0xffff0000u);
    key[8 * c + 2] = __builtin_bit_cast(float, w.y << 16);
    key[8 * c + 3] = __builtin_bit_cast(float, w.y & 0xffff0000u);
    key[8 * c + 4] = __builtin_bit_cast(float, w.z << 16);
    key[8 * c + 5] = __builtin_bit_cast(float, w.z & 0xffff0000u);
    key[8 * c + 6] = __builtin_bit_cast(float, w.w << 16);
    key[8 * c + 7] = __builtin_bit_cast(float, w.w & 0xffff0000u);
  }
  for (int r = 0; r < rn; r++) {
    const int c = complete[r0 + r];
    if (c <= top || b >= c) continue;
    float total = 0.0f;
    // A head's dot as the reference's wave computes it: lane l sums d = l, l + 32, l + 64, l + 96 from +0, then the
    // xor butterfly over masks 16, 8, 4, 2, 1 (pairs l, l + m of the halves at each level).
    [[loom::unroll]] for (unsigned h = 0; h < kIdxHeads; h++) {
      const float* qh = qs + (r * kIdxHeads + h) * kIdxDims;
      float s[32];
      [[loom::unroll]] for (unsigned c = 0; c < 8; c++) {
        const f32x4 a = *(const f32x4*)&qh[4 * c];
        s[4 * c] = 0.0f + a.x * key[4 * c];
        s[4 * c + 1] = 0.0f + a.y * key[4 * c + 1];
        s[4 * c + 2] = 0.0f + a.z * key[4 * c + 2];
        s[4 * c + 3] = 0.0f + a.w * key[4 * c + 3];
      }
      [[loom::unroll]] for (unsigned i = 1; i < 4; i++) {
        [[loom::unroll]] for (unsigned c = 0; c < 8; c++) {
          const f32x4 a = *(const f32x4*)&qh[32 * i + 4 * c];
          s[4 * c] = s[4 * c] + a.x * key[32 * i + 4 * c];
          s[4 * c + 1] = s[4 * c + 1] + a.y * key[32 * i + 4 * c + 1];
          s[4 * c + 2] = s[4 * c + 2] + a.z * key[32 * i + 4 * c + 2];
          s[4 * c + 3] = s[4 * c + 3] + a.w * key[32 * i + 4 * c + 3];
        }
      }
      [[loom::unroll]] for (unsigned l = 0; l < 16; l++) s[l] = s[l] + s[l + 16];
      [[loom::unroll]] for (unsigned l = 0; l < 8; l++) s[l] = s[l] + s[l + 8];
      [[loom::unroll]] for (unsigned l = 0; l < 4; l++) s[l] = s[l] + s[l + 4];
      [[loom::unroll]] for (unsigned l = 0; l < 2; l++) s[l] = s[l] + s[l + 2];
      total = total + loom::scalar::maxnumf(s[0] + s[1], 0.0f);
    }
    scores[(long long)(r0 + r) * stride + b] = total / __builtin_bit_cast(float, 0x413504f3u);
  }
}
