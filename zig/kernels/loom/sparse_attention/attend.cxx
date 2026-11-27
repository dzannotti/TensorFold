// SPDX-License-Identifier: MIT

// Attention core (listed or first-n keys), bf16 KV, head_dim 256, 12 query heads a KV head, gfx11 bf16 WMMA, CUDA's
// softmax arithmetic: a workgroup a (KV head, row), a wave a part, the parts merged in order in LDS (README.md).
#include <loomcxx/kernel.h>

#include "wmma_tile.h"

static loom::kernel::configuration configure_attend(unsigned kv_heads, unsigned rows) {
  return {{kv_heads, rows, 1}, {32 * sa::kParts, 1, 1}};
}

[[loom::kernel(configure_attend)]]
void sparse_attend(const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* out, long long id_stride, long long cap, int heads, int kv_heads, int dims,
                   float scale, int parts) {
  using namespace sa;
  if (dims != (int)kDims || heads != kv_heads * (int)kGroup || parts != (int)kParts) return;
  LOOM_WORKGROUP unsigned arena[kArenaWords];
  unsigned short* const qp = (unsigned short*)arena;
  float* const sc = (float*)arena;
  const unsigned g = loom::kernel::workgroup::id.x, r = loom::kernel::workgroup::id.y;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31, col = lane & 15, half = lane >> 4;
  const unsigned part = (unsigned)loom::kernel::subgroup::id();
  const int n = count[r];
  const bool listed = sparse[r] != 0;
  const Rows rows = {(const unsigned short*)k, (const unsigned short*)v, listed ? ids + (long long)r * id_stride : count + r,
                     (long long)g * cap, listed};
  const long long qrow = (long long)r * heads + g * kGroup;
  stage_q(qp, q + qrow * kDims, tid, 32 * kParts);
  lds_barrier();

  unsigned short* const tw = (unsigned short*)(arena + kQWords + part * kSlotWords);
  const int lo = (int)((long long)part * n / kParts), hi = (int)((long long)(part + 1) * n / kParts);
  Acc st = start();
  for (int t0 = lo; t0 < hi; t0 += (int)kTile) {
    const int tn = hi - t0 < (int)kTile ? hi - t0 : (int)kTile;
    st = tile(st, rows, t0, tn, qp, tw, col, half, scale);
  }

  lds_barrier();
  st = merge_own(st, half, part == 0);
  for (unsigned p = 1; p < kParts; p++) {
    if (part == p) put_part(sc, st, lane, col, half);
    lds_barrier();
    st = merge_lds(st, sc, lane, half, part == 0);
    lds_barrier();
  }
  if (part == 0) store_out(out + qrow * kDims, st, col, half);
}
