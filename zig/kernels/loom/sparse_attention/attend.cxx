// SPDX-License-Identifier: MIT

// Prompt attention core (listed or first-n keys), bf16 KV, head_dim 256, 12 query heads a KV head, gfx11 bf16 WMMA:
// `sparse_attend` writes each part's o, m and l for `sparse_merge` (merge.cxx). ABI, sum order, geometry: README.md.
#include <loomcxx/kernel.h>

#include "wmma_tile.h"

constexpr unsigned kBlockParts = 4, kWaves = 2 * kBlockParts;

static loom::kernel::configuration configure_attend(unsigned kv_heads, unsigned rows) {
  return {{kv_heads, rows, sa::kParts / kBlockParts}, {32 * kWaves, 1, 1}};
}

[[loom::kernel(configure_attend)]]
void sparse_attend(const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* parts_o, float* parts_ms, long long id_stride, long long cap, int heads,
                   int kv_heads, int dims, float scale, int parts) {
  using namespace sa;
  if (dims != (int)kDims || heads != kv_heads * (int)kGroup || parts != (int)kParts) return;
  LOOM_WORKGROUP unsigned arena[kQWords + kWaves * kSlotWords];
  unsigned short* const qp = (unsigned short*)arena;
  const unsigned g = loom::kernel::workgroup::id.x, r = loom::kernel::workgroup::id.y;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31, col = lane & 15, half = lane >> 4;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), side = wave & 1;
  const unsigned part = loom::kernel::workgroup::id.z * kBlockParts + (wave >> 1);
  const int n = count[r];
  const Rows rows = {(const unsigned short*)k, (const unsigned short*)v, ids + (long long)r * id_stride,
                     (long long)g * cap, sparse[r] != 0};
  stage_q(qp, q + ((long long)r * heads + g * kGroup) * kDims, tid, 32 * kWaves);
  lds_barrier();

  Part st = start();
  unsigned* const mine = arena + kQWords + wave * kSlotWords;
  const unsigned* const other = arena + kQWords + (wave ^ 1) * kSlotWords;
  const int lo = (int)((long long)part * n / kParts), hi = (int)((long long)(part + 1) * n / kParts);
  // every wave runs the longest part's tile count: the barriers stay workgroup-uniform; extra tiles change nothing
  const int iters = ((n + (int)kParts - 1) / (int)kParts + (int)kTile - 1) / (int)kTile;
  for (int it = 0; it < iters; it++) {
    const int t0 = lo + it * (int)kTile;
    const int tn = hi - t0 < (int)kTile ? hi - t0 : (int)kTile;
    // past the part's keys (tn <= 0) the tile reads a listed key as +0 rows and changes nothing
    const Key key = key_of(rows, t0 < hi ? t0 : (n > 0 ? n - 1 : 0), tn, col);
    const f32x8 x = scores_half(key, tn, qp, side, col);
    const VHalf vw = load_v(key, side, half, tn);
    put8(mine, lane, x);
    lds_barrier();
    const f32x8 y = get8(other, lane);
    const Probs pr = softmax(st.m, st.l, (side ? y + x : x + y) * scale, tn, col, half);
    lds_barrier();
    st = values_half(st, vw, pr, (unsigned short*)mine, col, half);
  }
  store_part(st, parts_o, parts_ms, ((long long)r * heads + g * kGroup) * kParts + part, side, col, half);
}
