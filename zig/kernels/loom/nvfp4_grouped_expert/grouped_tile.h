// SPDX-License-Identifier: MIT

// Generic grouped-expert prompt tile, gfx11 bf16 WMMA (wave32); geometry, decoder contract and sum order: README.md.
#ifndef K4_GROUPED_TILE_H_
#define K4_GROUPED_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>

namespace k4 {

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

#ifndef K4_ROWS
#define K4_ROWS 16
#endif
constexpr unsigned kTileRows = K4_ROWS, kTileCols = 128, kStage = 32, kPitch = 40, kFold = 128;
constexpr unsigned kXBuf = kTileRows * kPitch, kWBuf = kTileCols * kPitch;
constexpr unsigned kArena = 2 * kXBuf + 2 * kWBuf;

// Half a stage of one weight row, decoded: 16 bf16 in K order.
struct Half {
  u32x4 lo, hi;
};

// A wave's sums: row fragment i as a<i> (lane: column lane % 16, rows 2e + lane / 16 of element e).
struct Acc {
  f32x8 a0, a1, a2, a3, a4, a5, a6, a7;
};

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

#define K4_MMA(I)                                                                                         \
  if constexpr (NF > I) {                                                                                 \
    const bf16x16 a = fr::load<fr::role::lhs, bf16x16>(xv, {16 * I, 16 * T}, sh);                         \
    if constexpr (FIRST) d.a##I = loom::vector::mma(a, bw, init);                                         \
    else d.a##I = loom::vector::mma(a, bw, fr::attach<fr::role::init>(d.a##I, sh));                      \
  }
#define K4_FOLD(I) \
  if constexpr (NF > I) acc.a##I += d.a##I;

template <unsigned NF, unsigned T, bool FIRST>
LOOM_INLINE void slice(Acc& d, const __bf16* xp, const __bf16* wp) {
  const fr::shape<int, int> sh{16, 16};
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const f32x8 init = fr::attach<fr::role::init>(zero, sh);
  const auto xv = loom::buffer::view<kTileRows, kPitch>(xp, {});
  const auto wv = loom::buffer::view<kStage, 16>(wp, {}, loom::encoding::layout::strided(1, kPitch));
  const bf16x16 bw = fr::load<fr::role::rhs, bf16x16>(wv, {16 * T, 0}, sh);
  K4_MMA(0) K4_MMA(1) K4_MMA(2) K4_MMA(3) K4_MMA(4) K4_MMA(5) K4_MMA(6) K4_MMA(7)
}

template <unsigned NF, bool FIRST, bool LAST>
LOOM_INLINE void stage(Acc& acc, Acc& d, const __bf16* xp, const __bf16* wp) {
  slice<NF, 0, FIRST>(d, xp, wp);
  slice<NF, 1, false>(d, xp, wp);
  if constexpr (LAST) {
    K4_FOLD(0) K4_FOLD(1) K4_FOLD(2) K4_FOLD(3) K4_FOLD(4) K4_FOLD(5) K4_FOLD(6) K4_FOLD(7)
  }
}

#undef K4_MMA
#undef K4_FOLD

template <class Dec, unsigned K, unsigned D, unsigned NF>
LOOM_INLINE Acc grouped_loop(const typename Dec::Weights& w, unsigned wrow, const __bf16* xrow, __bf16* lds) {
  constexpr unsigned kFolds = K / kFold, kPer = kFold / kStage, kStages = K / kStage;
  static_assert(kPer == 4, "the stage loop unrolls one fold of four stages");
  const unsigned tid = loom::kernel::workitem::id.x;
  const unsigned col = tid & 127, half = tid >> 7, xr = tid >> 1, xpart = tid & 1;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id();
  const u32x4* xsrc = (const u32x4*)xrow + xpart * 2;
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  Acc acc = {zero, zero, zero, zero, zero, zero, zero, zero};
  Acc d = acc;
  __bf16* const xs = lds;
  __bf16* const ws = lds + 2 * kXBuf;
  const bool xlive = xr < kTileRows;
  u32x4* const wd0 = (u32x4*)(ws + col * kPitch + half * 16);
  u32x4* const xd0 = (u32x4*)(xs + xr * kPitch + xpart * 16);
  const __bf16* const wp0 = ws + wave * 16 * kPitch;

  // Stage s computes while s + 1 (n) and, for D = 3, s + 2 (f) are in flight; s + D is fetched, clamped to the end.
  typename Dec::Raw r0 = Dec::fetch(w, wrow, 0, half);
  u32x4 x00 = xsrc[0], x01 = xsrc[1];
  typename Dec::Raw rn = Dec::fetch(w, wrow, 1, half);
  u32x4 xn0 = xsrc[4], xn1 = xsrc[5];
  typename Dec::Raw rf = Dec::fetch(w, wrow, D == 3 ? 2 : 1, half);
  u32x4 xf0 = xsrc[D == 3 ? 8 : 4], xf1 = xsrc[D == 3 ? 9 : 5];
  {
    const Half h = Dec::template decode<Half>(r0);
    wd0[0] = h.lo;
    wd0[1] = h.hi;
    if (xlive) {
      xd0[0] = x00;
      xd0[1] = x01;
    }
  }
  lds_barrier();
  for (unsigned g = 0; g < kFolds; g++) {
    [[loom::unroll]] for (unsigned q = 0; q < kPer; q++) {
      const unsigned s = g * kPer + q, b = (q & 1) * kWBuf, nb = kWBuf - b, xb = (q & 1) * kXBuf, xnb = kXBuf - xb;
      const unsigned ahead = s + D < kStages ? s + D : kStages - 1;
      const typename Dec::Raw ra = Dec::fetch(w, wrow, ahead, half);
      const u32x4 xa0 = xsrc[4 * ahead], xa1 = xsrc[4 * ahead + 1];
      if (q == 0) stage<NF, true, false>(acc, d, xs + xb, wp0 + b);
      if (q == 1 || q == 2) stage<NF, false, false>(acc, d, xs + xb, wp0 + b);
      if (q == 3) stage<NF, false, true>(acc, d, xs + xb, wp0 + b);
      const Half h = Dec::template decode<Half>(rn);
      u32x4* wd = (u32x4*)((__bf16*)wd0 + nb);
      wd[0] = h.lo;
      wd[1] = h.hi;
      if (xlive) {
        u32x4* xd = (u32x4*)((__bf16*)xd0 + xnb);
        xd[0] = xn0;
        xd[1] = xn1;
      }
      if constexpr (D == 3) {
        rn = rf;
        xn0 = xf0;
        xn1 = xf1;
        rf = ra;
        xf0 = xa0;
        xf1 = xa1;
      } else {
        rn = ra;
        xn0 = xa0;
        xn1 = xa1;
      }
      lds_barrier();
    }
  }
  return acc;
}

// `rows` (1..128) of an item; xrow: this work-item's x row (a live one past the item). The LDS arena is free after.
template <class Dec, unsigned K, unsigned D>
LOOM_INLINE Acc grouped_tile(const typename Dec::Weights& w, unsigned wrow, const __bf16* xrow, unsigned rows,
                             __bf16* lds) {
  const unsigned frags = (rows + 15) / 16;
#if K4_ROWS == 16
  return grouped_loop<Dec, K, D, 1>(w, wrow, xrow, lds);
#else
  if (frags >= 8) return grouped_loop<Dec, K, D, 8>(w, wrow, xrow, lds);
  if (frags == 7) return grouped_loop<Dec, K, D, 7>(w, wrow, xrow, lds);
  if (frags == 6) return grouped_loop<Dec, K, D, 6>(w, wrow, xrow, lds);
  if (frags == 5) return grouped_loop<Dec, K, D, 5>(w, wrow, xrow, lds);
  if (frags == 4) return grouped_loop<Dec, K, D, 4>(w, wrow, xrow, lds);
  if (frags == 3) return grouped_loop<Dec, K, D, 3>(w, wrow, xrow, lds);
  if (frags == 2) return grouped_loop<Dec, K, D, 2>(w, wrow, xrow, lds);
  return grouped_loop<Dec, K, D, 1>(w, wrow, xrow, lds);
#endif
}

}  // namespace k4

#endif  // K4_GROUPED_TILE_H_
