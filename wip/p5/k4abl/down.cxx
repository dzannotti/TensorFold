// SPDX-License-Identifier: MIT

// Grouped-expert down prompt GEMM, NVFP4 weights, bf16 activations, fp32 out: ABI and sum order in README.md.
#include <loomcxx/kernel.h>

#include "grouped_tile.h"
#include "nvfp4_decode.h"

#ifndef K4_DK
#define K4_DK 640
#endif
#ifndef K4_DN
#define K4_DN 2560
#endif

#ifdef ABL_NOSTORE
#define STORE_OK (acc.a0[0] == 1.2345f)
#else
#define STORE_OK true
#endif
constexpr unsigned kDnK = K4_DK, kDnN = K4_DN;

static loom::kernel::configuration configure_down(unsigned items, unsigned row_blocks) {
  return {{kDnN / 128, row_blocks, items}, {256, 1, 1}};
}

#define K4_STORE_ROW(I)                                                  \
  if (I * 16 < rows) {                                                   \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {                  \
      const unsigned r = I * 16 + 2 * e + (lane >> 4);                   \
      if (r < rows && STORE_OK) out[(unsigned)members[first + m0 + r] * kDnN + c] = acc.a##I[e] * g; \
    }                                                                    \
  }

[[loom::kernel(configure_down)]]
void nvfp4_expert_down(const __bf16* x, const unsigned* words, const unsigned char* scales, const float* globals,
                       const int* items, const int* members, float* out, int x_div, int n, int k) {
  if (n != (int)kDnN || k != (int)kDnK) return;
  LOOM_WORKGROUP __bf16 lds[k4::kArena];
  const unsigned tid = loom::kernel::workitem::id.x;
  const unsigned n0 = loom::kernel::workgroup::id.x * 128, m0 = loom::kernel::workgroup::id.y * k4::kTileRows;
  const unsigned z = loom::kernel::workgroup::id.z;
  const unsigned expert = (unsigned)items[3 * z], first = (unsigned)items[3 * z + 1];
  const unsigned count = (unsigned)items[3 * z + 2];
  if (m0 >= count) return;
  const unsigned rows = count - m0 < k4::kTileRows ? count - m0 : k4::kTileRows;
  const unsigned col = tid & 127, xr = tid >> 1;
  const unsigned src = (unsigned)members[first + m0 + (xr < rows ? xr : 0)] / (unsigned)x_div;
  const unsigned stack = expert * kDnN;
  const k4::Nvfp4Rows<kDnK>::Weights w = {words + stack * (kDnK / 8), scales + stack * (kDnK / 16)};
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), lane = tid & 31;
  const k4::Acc acc = k4::grouped_tile<k4::Nvfp4Rows<kDnK>, kDnK, 2>(w, n0 + col, x + src * kDnK, rows, lds);
  const float g = globals[expert] * 16384.0f;
  const unsigned c = n0 + wave * 16 + (lane & 15);
  K4_STORE_ROW(0) K4_STORE_ROW(1) K4_STORE_ROW(2) K4_STORE_ROW(3) K4_STORE_ROW(4) K4_STORE_ROW(5) K4_STORE_ROW(6)
  K4_STORE_ROW(7)
}

#undef K4_STORE_ROW
