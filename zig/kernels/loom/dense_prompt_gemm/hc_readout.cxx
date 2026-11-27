// SPDX-License-Identifier: MIT

// A hyper-connection read-out on the GEMM tile (README.md): the down with the gates in its epilogue, the up with the
// gates and the stream mix in its epilogue. x: the normed streams, bf16 [m][4 dims].
#include <loomcxx/kernel.h>

#include "bf16_rows.h"
#include "gemm_tile.h"
#include "hc_exact.h"
#include "hc_tiles.h"

typedef k5::Geometry<4, 2, 4> Hc128x128;
typedef k5::Geometry<4, 4, 4> Hc128x256;
typedef k5::Geometry<4, 4, 2> Hc128x128s;

template <class G>
static loom::kernel::configuration configure_hc(unsigned m, unsigned n) {
  return {{(n + G::kCols - 1) / G::kCols, (m + G::kRows - 1) / G::kRows, 1}, {G::kThreads, 1, 1}};
}

// silu(v) (c below `low`) or 2 bf16(sigmoid(v)), v = bf16(bf16(z) / 4) from zb = bf16(z): the 16 bits. silu and
// sigmoid share exp(-v) and one division (numerator v or 1); x / 4 is x 0.25 to the bit.
LOOM_INLINE unsigned hc_gate(unsigned zb, bool silu) {
  const float v = k5::bf16_round(k5::bits(zb << 16) * 0.25f);
  const float num = silu ? v : 1.0f;
  const float q = num / (1.0f + k5::exp_exact(-v));
  const float g = 2.0f * k5::bf16_round(q);
  return k5::bf16_bits(silu ? q : g);
}

LOOM_INLINE unsigned hc_gate_pair(unsigned w) { return hc_gate(w & 0xffffu, true) | (hc_gate(w >> 16, true) << 16); }

// bf16(z) of a fragment into the LDS rows [kRows][kCols].
#define K5_ZB(I, J)                                                                                  \
  if (G::NT > J) {                                                                                   \
    const unsigned cl = wc * 16 * G::NT + J * 16 + (lane & 15);                                      \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++)                                                \
      zl[(wr * 32 + I * 16 + 2 * e + (lane >> 4)) * G::kCols + cl] = (unsigned short)k5::bf16_bits(acc.a##I##J[e]); \
  }

// z = x . W^T (W: bf16 [n][k]) on columns n0.., then the gates: act [m][low], and inject [m][4] where n > low. z goes
// through the LDS as bf16 (the gates' first rounding), 8 columns a work-item.
template <class G>
LOOM_INLINE void hc_down_tile(const __bf16* x, const __bf16* wt, __bf16* act, __bf16* inject, int m, int n, int k, int low,
                         unsigned n0, __bf16* lds) {
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31;
  const unsigned m0 = loom::kernel::workgroup::id.y * G::kRows;
  const unsigned um = (unsigned)m, un = (unsigned)n, uk = (unsigned)k, ul = (unsigned)low;
  if (m0 >= um || n0 >= un || uk < 64 || uk % 64 != 0 || ul % 8 != 0 || un > ul + k5::kStreams) return;
  const unsigned rows = um - m0 < G::kRows ? um - m0 : G::kRows;
  const unsigned cols = un - n0 < G::kCols ? un - n0 : G::kCols;
  const k5::Bf16Rows::Weights w = {(const k5::u32x4*)wt, uk};
  const k5::Acc acc = k5::gemm_tile<k5::Bf16Rows, G>(w, (const k5::u32x4*)x, uk, m0, rows, n0, cols, lds);
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), wr = wave % G::WR, wc = wave / G::WR;
  unsigned short* const zl = (unsigned short*)lds;
  K5_ZB(0, 0) K5_ZB(0, 1) K5_ZB(0, 2) K5_ZB(0, 3) K5_ZB(0, 4)
  K5_ZB(1, 0) K5_ZB(1, 1) K5_ZB(1, 2) K5_ZB(1, 3) K5_ZB(1, 4)
  k5::lds_barrier();
  unsigned short* const i16 = (unsigned short*)inject;
  for (unsigned i = tid; i < rows * (G::kCols / 8); i += G::kThreads) {
    const unsigned r = i / (G::kCols / 8), cl = 8 * (i % (G::kCols / 8)), c = n0 + cl, row = m0 + r;
    const k5::u32x4 zv = *(const k5::u32x4*)(zl + r * G::kCols + cl);
    if (c + 8 <= ul) {
      const k5::u32x4 out = {hc_gate_pair(zv[0]), hc_gate_pair(zv[1]), hc_gate_pair(zv[2]), hc_gate_pair(zv[3])};
      *(k5::u32x4*)(act + row * ul + c) = out;
    } else if (c < un) {
      [[loom::unroll]] for (unsigned j = 0; j < 8; j++)
        if (c + j >= ul && c + j < un)
          i16[row * k5::kStreams + c + j - ul] = (unsigned short)hc_gate((zv[j / 2] >> (16 * (j % 2))) & 0xffffu, false);
    }
  }
}

#undef K5_ZB

// bf16(g) of stream s into the LDS rows [s][row][SPAN].
#define K5_GATE_UP(I, J)                                                                             \
  if (G::NT > J) {                                                                                   \
    const unsigned j = wc * 16 * G::NT + J * 16 + (lane & 15), s = j / SPAN, dd = j % SPAN;          \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++)                                                \
      gates[(s * G::kRows + wr * 32 + I * 16 + 2 * e + (lane >> 4)) * SPAN + dd] =                   \
          (unsigned short)k5::bf16_bits(acc.a##I##J[e]);                                             \
  }

// The term bf16(bf16(sigmoid(g)) x) of a bf16 g and x.
LOOM_INLINE float hc_term(unsigned gb, unsigned xb) {
  return k5::bf16_round(k5::bf16_round(k5::sigmoid_exact(k5::bits(gb))) * k5::bits(xb));
}

// g = act . W^T (W: bf16 [4 dims][k]), a block's columns SPAN dimensions of each stream; mixed [m][dims] = bf16 of the
// terms bf16(bf16(sigmoid(bf16(g))) x) summed in stream order from +0, / 4 (x 0.25 to the bit). g goes through the LDS
// as bf16, 8 dimensions a work-item.
template <class G>
LOOM_INLINE void hc_up_mix_tile(const __bf16* act, const __bf16* wt, const __bf16* x, __bf16* mixed, int m, int dims, int k,
                           __bf16* lds) {
  constexpr unsigned SPAN = G::kCols / k5::kStreams, C8 = SPAN / 8;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31;
  const unsigned n0 = loom::kernel::workgroup::id.x * G::kCols, m0 = loom::kernel::workgroup::id.y * G::kRows;
  const unsigned um = (unsigned)m, uk = (unsigned)k, ud = (unsigned)dims, d0 = (n0 / G::kCols) * SPAN;
  if (m0 >= um || d0 >= ud || uk < 64 || uk % 64 != 0 || ud % SPAN != 0) return;
  const unsigned rows = um - m0 < G::kRows ? um - m0 : G::kRows;
  const typename k5::StreamRows<SPAN>::Weights w = {(const k5::u32x4*)wt, uk, ud};
  const k5::Acc acc = k5::gemm_tile<k5::StreamRows<SPAN>, G>(w, (const k5::u32x4*)act, uk, m0, rows, n0, G::kCols, lds);
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), wr = wave % G::WR, wc = wave / G::WR;
  unsigned short* const gates = (unsigned short*)lds;
  K5_GATE_UP(0, 0) K5_GATE_UP(0, 1) K5_GATE_UP(0, 2) K5_GATE_UP(0, 3) K5_GATE_UP(0, 4)
  K5_GATE_UP(1, 0) K5_GATE_UP(1, 1) K5_GATE_UP(1, 2) K5_GATE_UP(1, 3) K5_GATE_UP(1, 4)
  k5::lds_barrier();
  const unsigned c8 = tid % C8, d = d0 + 8 * c8;
  for (unsigned i = tid; i < rows * C8; i += G::kThreads) {
    const unsigned r = i / C8, row = m0 + r;
    float lo[4], hi[4];
    [[loom::unroll]] for (unsigned j = 0; j < 4; j++) lo[j] = hi[j] = 0.0f;
    for (unsigned s = 0; s < k5::kStreams; s++) {
      const k5::u32x4 xv = *(const k5::u32x4*)(x + (row * k5::kStreams + s) * ud + d);
      const k5::u32x4 gv = *(const k5::u32x4*)(gates + (s * G::kRows + r) * SPAN + 8 * c8);
      [[loom::unroll]] for (unsigned j = 0; j < 4; j++) {
        lo[j] += hc_term(gv[j] << 16, xv[j] << 16);
        hi[j] += hc_term(gv[j] & 0xffff0000u, xv[j] & 0xffff0000u);
      }
    }
    const k5::u32x4 out = {k5::bf16_bits(lo[0] * 0.25f) | (k5::bf16_bits(hi[0] * 0.25f) << 16),
                           k5::bf16_bits(lo[1] * 0.25f) | (k5::bf16_bits(hi[1] * 0.25f) << 16),
                           k5::bf16_bits(lo[2] * 0.25f) | (k5::bf16_bits(hi[2] * 0.25f) << 16),
                           k5::bf16_bits(lo[3] * 0.25f) | (k5::bf16_bits(hi[3] * 0.25f) << 16)};
    *(k5::u32x4*)(mixed + row * ud + d) = out;
  }
}

#undef K5_GATE_UP

// The tile's arena, or the block's bf16 sums if larger.
template <class G>
constexpr unsigned arena() {
  return G::kArena > G::kRows * G::kCols ? G::kArena : G::kRows * G::kCols;
}

static loom::kernel::configuration configure_down(unsigned m) { return {{2, (m + 127) / 128, 1}, {512, 1, 1}}; }

// hc_down_wide: 128 rows a block, columns 0..255 on a 128 x 256 tile, 256.. (at most 128) on a 128 x 128 tile of
// the same 16 waves. hc_down_narrow: 128 x 128 blocks. hc_up_mix: 128 x 256 blocks, 64 dimensions of each stream.
LOOM_INLINE void hc_down_wide_rows(const __bf16* x, const __bf16* w, __bf16* act, __bf16* inject, int m, int n, int k,
                                   int low, __bf16* lds) {
  if (loom::kernel::workgroup::id.x == 0) hc_down_tile<Hc128x256>(x, w, act, inject, m, n, k, low, 0, lds);
  else if (n <= 256 + 128) hc_down_tile<Hc128x128s>(x, w, act, inject, m, n, k, low, 256, lds);
}

[[loom::kernel(configure_down)]]
void hc_down_wide(const __bf16* x, const __bf16* w, __bf16* act, __bf16* inject, int m, int n, int k, int low) {
  LOOM_WORKGROUP __bf16 lds[arena<Hc128x256>()];
  hc_down_wide_rows(x, w, act, inject, m, n, k, low, lds);
}

[[loom::kernel(configure_hc<Hc128x128>)]]
void hc_down_narrow(const __bf16* x, const __bf16* w, __bf16* act, __bf16* inject, int m, int n, int k, int low) {
  LOOM_WORKGROUP __bf16 lds[arena<Hc128x128>()];
  hc_down_tile<Hc128x128>(x, w, act, inject, m, n, k, low, loom::kernel::workgroup::id.x * Hc128x128::kCols, lds);
}

[[loom::kernel(configure_hc<Hc128x256>)]]
void hc_up_mix(const __bf16* act, const __bf16* w, const __bf16* x, __bf16* mixed, int m, int dims, int k) {
  LOOM_WORKGROUP __bf16 lds[arena<Hc128x256>()];
  hc_up_mix_tile<Hc128x256>(act, w, x, mixed, m, dims, k, lds);
}
