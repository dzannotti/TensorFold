// SPDX-License-Identifier: MIT

// Grouped-expert gate|up prompt GEMM + SwiGLU, NVFP4 weights, bf16 activations: ABI and sum order in README.md.
#include <loomcxx/kernel.h>

#include "grouped_tile.h"
#include "nvfp4_decode.h"
#include "swiglu_exact.h"

#ifndef K4_K
#define K4_K 2560
#endif
#ifndef K4_N
#define K4_N 640
#endif

constexpr unsigned kGuK = K4_K, kGuN = K4_N;

static loom::kernel::configuration configure_gate_up(unsigned items, unsigned row_blocks) {
  return {{kGuN / 64, row_blocks, items}, {256, 1, 1}};
}

// The up waves (4..7) hand their scaled sums to the gate waves (0..3) through the free LDS arena (128 x 65 floats).
#define K4_UP_ROW(I)                                                                                   \
  if (I * 16 < rows) {                                                                                 \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++)                                                  \
      red[(I * 16 + 2 * e + (lane >> 4)) * 65 + c] = acc.a##I[e] * gs;                                 \
  }
#define K4_ACT_ROW(I)                                                                                  \
  if (I * 16 < rows) {                                                                                 \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {                                                \
      const unsigned r = I * 16 + 2 * e + (lane >> 4);                                                 \
      if (r < rows)                                                                                    \
        act[(unsigned)members[first + m0 + r] * kGuN + f0 + c] =                                       \
            k4::swiglu_bf16(acc.a##I[e] * gs, red[r * 65 + c]);                                       \
    }                                                                                                  \
  }

[[loom::kernel(configure_gate_up)]]
void nvfp4_expert_gate_up(const __bf16* x, const unsigned* words, const unsigned char* scales, const float* globals,
                          const int* items, const int* members, __bf16* act, int x_div, int n, int k) {
  if (n != (int)kGuN || k != (int)kGuK) return;
  LOOM_WORKGROUP __bf16 lds[k4::kArena];
  const unsigned tid = loom::kernel::workitem::id.x;
  const unsigned f0 = loom::kernel::workgroup::id.x * 64, m0 = loom::kernel::workgroup::id.y * k4::kTileRows;
  const unsigned z = loom::kernel::workgroup::id.z;
  const unsigned expert = (unsigned)items[3 * z], first = (unsigned)items[3 * z + 1];
  const unsigned count = (unsigned)items[3 * z + 2];
  if (m0 >= count) return;
  const unsigned rows = count - m0 < k4::kTileRows ? count - m0 : k4::kTileRows;
  const unsigned col = tid & 127, xr = tid >> 1;
  const unsigned token = (unsigned)members[first + m0 + (xr < rows ? xr : 0)] / (unsigned)x_div;
  const unsigned stack = expert * 2 * kGuN;
  const k4::Nvfp4Rows<kGuK>::Weights w = {words + stack * (kGuK / 8), scales + stack * (kGuK / 16)};
  const unsigned wrow = col < 64 ? f0 + col : kGuN + f0 + col - 64;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), lane = tid & 31;
  const k4::Acc acc = k4::grouped_tile<k4::Nvfp4Rows<kGuK>, kGuK, 3>(w, wrow, x + token * kGuK, rows, lds);
  float* const red = (float*)lds;
  const bool up = wave >= 4;
  const unsigned c = (wave & 3) * 16 + (lane & 15);
  const float gs = globals[2 * expert + (up ? 1 : 0)] * 16384.0f;
  if (up) {
    K4_UP_ROW(0) K4_UP_ROW(1) K4_UP_ROW(2) K4_UP_ROW(3) K4_UP_ROW(4) K4_UP_ROW(5) K4_UP_ROW(6) K4_UP_ROW(7)
  }
  k4::lds_barrier();
  if (!up) {
    K4_ACT_ROW(0) K4_ACT_ROW(1) K4_ACT_ROW(2) K4_ACT_ROW(3) K4_ACT_ROW(4) K4_ACT_ROW(5) K4_ACT_ROW(6) K4_ACT_ROW(7)
  }
}

#undef K4_UP_ROW
#undef K4_ACT_ROW
