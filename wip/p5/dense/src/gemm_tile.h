// SPDX-License-Identifier: MIT

// Dense prompt GEMM tile, gfx11 bf16 WMMA (wave32): y = x . W^T in the HIP matrix tile's sum order (README.md).
// G: WR x WC waves, each 32 rows x 16 NT columns (2 x NT fragments). Dec: the weights' 16-byte chunk decoder.
#ifndef K5_GEMM_TILE_H_
#define K5_GEMM_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#ifdef WP_PIPE
#include "pipe_step.h"
#endif

namespace k5 {

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

// A stage is 32 inputs; LDS rows are 40 inputs apart (16-byte reads of 16 rows hit distinct banks).
constexpr unsigned kStage = 32, kPitch = 40;

template <unsigned WR_, unsigned WC_, unsigned NT_>
struct Geometry {
  static constexpr unsigned WR = WR_, WC = WC_, NT = NT_;
  static constexpr unsigned kRows = 32 * WR, kCols = 16 * NT * WC, kThreads = 32 * WR * WC;
  static constexpr unsigned kXBuf = kRows * kPitch, kWBuf = kCols * kPitch;  // one stage of x, of weights (bf16)
  static constexpr unsigned kArena = 2 * (kXBuf + kWBuf);                       // double-buffered
  static constexpr unsigned NXS = (4 * kRows + kThreads - 1) / kThreads;        // x chunk slots a thread
  static constexpr unsigned NS = NXS + (4 * kCols + kThreads - 1) / kThreads;   // all slots
  static_assert(NT <= 5 && NS <= 6 && kArena * 2 <= 65536, "fragments, slots or LDS out of range");
};

// A wave's 2 x NT fragments (rows 16 i.., columns 16 j..); lane: column lane % 16, rows 2 e + lane / 16 of element e.
struct Acc {
  f32x8 a00, a01, a02, a03, a04, a10, a11, a12, a13, a14;
};

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

// A fragment of 16 LDS rows from r0 (lane: row r0 + lane % 16), inputs 16 t.. of the stage in `buf`.
template <fr::role Role>
LOOM_INLINE bf16x16 frag(const __bf16* buf, unsigned r0, unsigned t, unsigned lane) {
  const fr::shape<int, int> sh{16, 16};
  const u32x8 v = *(const u32x8*)(buf + (r0 + (lane & 15)) * kPitch + 16 * t);
  return fr::attach<Role>(__builtin_bit_cast(bf16x16, v), sh);
}

#define K5_MMA(I, J)                                                                                  \
  if constexpr (NF > I && NT > J) {                                                                   \
    if constexpr (FIRST) d.a##I##J = loom::vector::mma(a##I, b##J, init);                             \
    else d.a##I##J = loom::vector::mma(a##I, b##J, fr::attach<fr::role::init>(d.a##I##J, sh));       \
  }
#define K5_B(J) \
  bf16x16 b##J = a0; \
  if constexpr (NT > J) b##J = frag<fr::role::rhs>(ws, wc + 16 * J, T, lane);
#define K5_FOLD(I, J) \
  if constexpr (NF > I && NT > J) acc.a##I##J += d.a##I##J;

// K16 slice T of a stage: NF row fragments (x rows xr..) times NT column fragments (weight rows wc..). FIRST starts
// a fold's chains at +0.
template <unsigned NF, unsigned NT, unsigned T, bool FIRST>
LOOM_INLINE void slice(Acc& d, const __bf16* xs, const __bf16* ws, unsigned xr, unsigned wc, unsigned lane) {
  const fr::shape<int, int> sh{16, 16};
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const f32x8 init = fr::attach<fr::role::init>(zero, sh);
  const bf16x16 a0 = frag<fr::role::lhs>(xs, xr, T, lane);
  bf16x16 a1 = a0;
  if constexpr (NF > 1) a1 = frag<fr::role::lhs>(xs, xr + 16, T, lane);
  K5_B(0) K5_B(1) K5_B(2) K5_B(3) K5_B(4)
  K5_MMA(0, 0) K5_MMA(0, 1) K5_MMA(0, 2) K5_MMA(0, 3) K5_MMA(0, 4)
  K5_MMA(1, 0) K5_MMA(1, 1) K5_MMA(1, 2) K5_MMA(1, 3) K5_MMA(1, 4)
}

// One stage (two K16 slices); LAST adds the fold's chains to the sums.
template <unsigned NF, unsigned NT, bool FIRST, bool LAST>
LOOM_INLINE void stage(Acc& acc, Acc& d, const __bf16* xs, const __bf16* ws, unsigned xr, unsigned wc, unsigned lane) {
#ifdef WP_PIPE
  if constexpr (NF == 2 && NT == 4) {
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    const __bf16* const xl = xs + (xr + (lane & 15)) * kPitch;
    const __bf16* const wl = ws + (wc + (lane & 15)) * kPitch;
    wp::Step2x4 st;
    if constexpr (FIRST) st = {zero, zero, zero, zero, zero, zero, zero, zero};
    else st = {d.a00, d.a01, d.a02, d.a03, d.a10, d.a11, d.a12, d.a13};
    st.a0 = frag<fr::role::lhs>(xs, xr, 0, lane);
    st.a1 = frag<fr::role::lhs>(xs, xr + 16, 0, lane);
    st.b0 = frag<fr::role::rhs>(ws, wc, 0, lane);
    st.b1 = frag<fr::role::rhs>(ws, wc + 16, 0, lane);
    st.b2 = frag<fr::role::rhs>(ws, wc + 32, 0, lane);
    st.b3 = frag<fr::role::rhs>(ws, wc + 48, 0, lane);
    st = wp::pipe_step(st, xl + 16, wl + 16);
    const fr::shape<int, int> sh{16, 16};
    const bf16x16 a0 = fr::attach<fr::role::lhs>(st.a0, sh), a1 = fr::attach<fr::role::lhs>(st.a1, sh);
    const bf16x16 b0 = fr::attach<fr::role::rhs>(st.b0, sh), b1 = fr::attach<fr::role::rhs>(st.b1, sh);
    const bf16x16 b2 = fr::attach<fr::role::rhs>(st.b2, sh), b3 = fr::attach<fr::role::rhs>(st.b3, sh);
    d.a00 = loom::vector::mma(a0, b0, fr::attach<fr::role::init>(st.c0_0, sh));
    d.a01 = loom::vector::mma(a0, b1, fr::attach<fr::role::init>(st.c0_1, sh));
    d.a02 = loom::vector::mma(a0, b2, fr::attach<fr::role::init>(st.c0_2, sh));
    d.a03 = loom::vector::mma(a0, b3, fr::attach<fr::role::init>(st.c0_3, sh));
    d.a10 = loom::vector::mma(a1, b0, fr::attach<fr::role::init>(st.c1_0, sh));
    d.a11 = loom::vector::mma(a1, b1, fr::attach<fr::role::init>(st.c1_1, sh));
    d.a12 = loom::vector::mma(a1, b2, fr::attach<fr::role::init>(st.c1_2, sh));
    d.a13 = loom::vector::mma(a1, b3, fr::attach<fr::role::init>(st.c1_3, sh));
  } else {
    slice<NF, NT, 0, FIRST>(d, xs, ws, xr, wc, lane);
    slice<NF, NT, 1, false>(d, xs, ws, xr, wc, lane);
  }
#else
  slice<NF, NT, 0, FIRST>(d, xs, ws, xr, wc, lane);
  slice<NF, NT, 1, false>(d, xs, ws, xr, wc, lane);
#endif
  if constexpr (LAST) {
    K5_FOLD(0, 0) K5_FOLD(0, 1) K5_FOLD(0, 2) K5_FOLD(0, 3) K5_FOLD(0, 4)
    K5_FOLD(1, 0) K5_FOLD(1, 1) K5_FOLD(1, 2) K5_FOLD(1, 3) K5_FOLD(1, 4)
  }
}

#undef K5_MMA
#undef K5_B
#undef K5_FOLD

// Staging: a stage is 4 16-byte chunks a row of x (kRows) and of weights (kCols). Slot i < NXS of thread t holds x
// chunk t + i T, the others weight chunk t + (i - NXS) T (row c / 4, part c % 4), rows clamped to live ones (their
// sums are never stored). A chunk is fetched two stages ahead into u, held in v, then stored to LDS.
#define K5_SLOT_DEF(I)                                                                                \
  constexpr bool isx##I = I < G::NXS;                                                                 \
  const unsigned c##I = tid + (isx##I ? I : I - G::NXS) * G::kThreads;                                \
  const bool on##I = I < G::NS && c##I < 4 * (isx##I ? G::kRows : G::kCols);                          \
  const unsigned q##I = c##I >> 2, lim##I = isx##I ? rows - 1 : cols - 1;                             \
  const unsigned r##I = q##I < lim##I ? q##I : lim##I;                                                \
  const unsigned so##I = isx##I ? (x0 + r##I) * (k / 8) + (c##I & 3) : Dec::base(w, w0 + r##I, c##I & 3); \
  const unsigned d0##I = (isx##I ? 0 : 2 * G::kXBuf) + q##I * kPitch + (c##I & 3) * 8;               \
  const unsigned bs##I = isx##I ? G::kXBuf : G::kWBuf;                                                \
  typename Dec::Raw u##I = Dec::zero(), v##I = u##I;
#define K5_SLOT_FETCH(I, S)                                                                           \
  if constexpr (I < G::NS) {                                                                          \
    if (on##I) {                                                                                      \
      if constexpr (isx##I) u##I = Dec::raw(x[so##I + 4 * (S)]);                                     \
      else u##I = Dec::fetch(w, so##I, (S));                                                          \
    }                                                                                                 \
  }
#define K5_SLOT_STORE(I, H)                                                                           \
  if constexpr (I < G::NS) {                                                                          \
    if (on##I) *(u32x4*)(lds + d0##I + (H) * bs##I) = isx##I ? Dec::chunk(v##I) : Dec::decode(v##I);  \
  }
#define K5_SLOT_SHIFT(I) \
  if constexpr (I < G::NS) v##I = u##I;
#define K5_SLOTS(M, ...) M(0 __VA_OPT__(,) __VA_ARGS__) M(1 __VA_OPT__(,) __VA_ARGS__) M(2 __VA_OPT__(,) __VA_ARGS__) \
  M(3 __VA_OPT__(,) __VA_ARGS__) M(4 __VA_OPT__(,) __VA_ARGS__) M(5 __VA_OPT__(,) __VA_ARGS__)

// The K loop of one block: x rows x0.. (`rows` live), weight rows w0.. (`cols` live), folds of PER stages (2 or 4).
// FULL: every row is live, else this wave computes its `live` (0..2) row fragments.
template <class Dec, class G, unsigned PER, bool FULL>
LOOM_INLINE Acc gemm_loop(const typename Dec::Weights& w, const u32x4* x, unsigned k, unsigned x0, unsigned rows,
                          unsigned w0, unsigned cols, unsigned live, __bf16* lds) {
  constexpr unsigned NT = G::NT;
  const unsigned stages = k / kStage;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id();
  const unsigned xr = (wave % G::WR) * 32, wc = (wave / G::WR) * 16 * NT;
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  Acc acc = {zero, zero, zero, zero, zero, zero, zero, zero, zero, zero};
  Acc d = acc;
  __bf16* const xs = lds;
  __bf16* const ws = lds + 2 * G::kXBuf;
  K5_SLOTS(K5_SLOT_DEF)
  K5_SLOTS(K5_SLOT_FETCH, 0)
  K5_SLOTS(K5_SLOT_SHIFT)
  K5_SLOTS(K5_SLOT_STORE, 0)
#ifndef WP_F1
  K5_SLOTS(K5_SLOT_FETCH, 1)
  K5_SLOTS(K5_SLOT_SHIFT)
#endif
  lds_barrier();
  for (unsigned s = 0; s < stages; s += PER) {
    [[loom::unroll]] for (unsigned h = 0; h < PER; h++) {
      const unsigned t = s + h;
      #ifdef WP_EARLY
      K5_SLOTS(K5_SLOT_STORE, (h & 1) ^ 1)
#endif
#ifdef ABL_NOFETCH
      K5_SLOTS(K5_SLOT_FETCH, (t & 1))
#elif defined(WP_F1)
      K5_SLOTS(K5_SLOT_FETCH, t + 1 < stages ? t + 1 : stages - 1)
      K5_SLOTS(K5_SLOT_SHIFT)
#else
      K5_SLOTS(K5_SLOT_FETCH, t + 2 < stages ? t + 2 : stages - 1)
#endif
      const __bf16* const xb = xs + (h & 1) * G::kXBuf;
      const __bf16* const wb = ws + (h & 1) * G::kWBuf;
      if (FULL || live == 2) {
        if (h == 0) stage<2, NT, true, false>(acc, d, xb, wb, xr, wc, lane);
        if (h > 0 && h < PER - 1) stage<2, NT, false, false>(acc, d, xb, wb, xr, wc, lane);
        if (h == PER - 1) stage<2, NT, false, true>(acc, d, xb, wb, xr, wc, lane);
      } else if (live == 1) {
        if (h == 0) stage<1, NT, true, false>(acc, d, xb, wb, xr, wc, lane);
        if (h > 0 && h < PER - 1) stage<1, NT, false, false>(acc, d, xb, wb, xr, wc, lane);
        if (h == PER - 1) stage<1, NT, false, true>(acc, d, xb, wb, xr, wc, lane);
      }
#if !defined(ABL_NOSTORE) && !defined(WP_EARLY)
      K5_SLOTS(K5_SLOT_STORE, (h & 1) ^ 1)
#endif
      K5_SLOTS(K5_SLOT_SHIFT)
#ifndef ABL_NOBAR
      lds_barrier();
#endif
    }
  }
  return acc;
}

#undef K5_SLOT_DEF
#undef K5_SLOT_FETCH
#undef K5_SLOT_STORE
#undef K5_SLOT_SHIFT
#undef K5_SLOTS

// One block: rows x0.. (1..kRows live) of x (16-byte units, k / 8 a row) by weight rows w0.. (1..kCols live).
// K is a multiple of 64: folds of 128 inputs where K takes them, else 64. The LDS is free after.
template <class Dec, class G>
LOOM_INLINE Acc gemm_tile(const typename Dec::Weights& w, const u32x4* x, unsigned k, unsigned x0, unsigned rows,
                          unsigned w0, unsigned cols, __bf16* lds) {
  const unsigned wr = ((unsigned)loom::kernel::subgroup::id()) % G::WR;
  const unsigned live = rows > wr * 32 + 16 ? 2 : rows > wr * 32 ? 1 : 0;
  const bool full = rows == G::kRows;
  if (k % 128 == 0) {
    if (full) return gemm_loop<Dec, G, 4, true>(w, x, k, x0, rows, w0, cols, 2, lds);
    return gemm_loop<Dec, G, 4, false>(w, x, k, x0, rows, w0, cols, live, lds);
  }
  if (full) return gemm_loop<Dec, G, 2, true>(w, x, k, x0, rows, w0, cols, 2, lds);
  return gemm_loop<Dec, G, 2, false>(w, x, k, x0, rows, w0, cols, live, lds);
}

}  // namespace k5

#endif  // K5_GEMM_TILE_H_
