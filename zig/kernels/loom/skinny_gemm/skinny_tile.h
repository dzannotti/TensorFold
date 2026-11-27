// SPDX-License-Identifier: MIT

// Skinny MXFP8 decode tile, gfx11 bf16 WMMA (wave32), 1-64 rows: fp4l_stream's sum order (README.md).
#ifndef SK_SKINNY_TILE_H_
#define SK_SKINNY_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include "mxfp8_cols.h"
#include "skinny_out.h"

namespace sk {

namespace fr = loom::vector::fragment;

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// Lanes a column of the stream tile at K: the most, up to 32, that split K's 32-input chunks evenly.
constexpr unsigned lanes_of(unsigned k) {
  unsigned l = 1;
  while (l < 32 && (k / 32) % (2 * l) == 0) l *= 2;
  return l;
}

// K, F row fragments (rows <= 16 F), NT column fragments a wave; one wave a column lane.
template <unsigned K_, unsigned F_, unsigned NT_>
struct Geo {
  static constexpr unsigned K = K_, F = F_, NT = NT_;
  static constexpr unsigned L = lanes_of(K), C = K / 32, T = C / L, W = L, kThreads = 32 * W;
  static constexpr unsigned kTile = 2 * T * NT;          // 8-column groups a tile (T blocks)
  static constexpr unsigned kScales = C * NT * 16;       // f32 multipliers a block
  static constexpr unsigned kRed = (W / 2) * 8 * 33;     // one tree level's partials, rows 33 apart
  static constexpr unsigned kLevels = W == 4 ? 2 : W == 8 ? 3 : W == 16 ? 4 : 5;
  static_assert(W >= 4 && F >= 1 && F <= 4 && NT >= 1 && NT <= 4, "geometry out of range");
};

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

// Blocks of side width `n` at tile `tile` groups: T blocks a tile.
LOOM_INLINE unsigned side_blocks(unsigned n, unsigned tile, unsigned t) { return ((n + 7) / 8 + tile - 1) / tile * t; }

template <class G>
LOOM_INLINE Place place(const Sides& s, unsigned b) {
  const unsigned w0 = s.c1, w1 = s.c2 - s.c1, w2 = s.c3 - s.c2, w3 = s.n - s.c3;
  const unsigned b0 = side_blocks(w0, G::kTile, G::T), b1 = b0 + side_blocks(w1, G::kTile, G::T);
  const unsigned b2 = b1 + side_blocks(w2, G::kTile, G::T);
  const bool four = s.count == 4;
  const unsigned side = !four || b < b0 ? 0 : b < b1 ? 1 : b < b2 ? 2 : 3;
  const unsigned start = side == 0 ? 0 : side == 1 ? b0 : side == 2 ? b1 : b2;
  const unsigned base = side == 0 ? 0 : side == 1 ? s.c1 : side == 2 ? s.c2 : s.c3;
  const unsigned width = side == 0 ? w0 : side == 1 ? w1 : side == 2 ? w2 : w3;
  const unsigned local = b - start, tile = local / G::T, p = local % G::T;
  return {side, base, width, tile * G::kTile + 2 * p, (2 * p) % G::T};
}

// Fragment nt's column c (0..15), side-local: groups group0 + 2 T nt and the next one (one rotation a wave).
template <class G>
LOOM_INLINE unsigned local_col(const Place& pl, unsigned nt, unsigned c) {
  return 8 * (pl.group0 + 2 * G::T * nt) + c;
}

// The block's e8m0 bytes as multipliers at sc[((q NT + nt) 2 + h) 8 + e], column 2 e + h of fragment nt.
template <class G>
LOOM_INLINE void stage_scales(const Args& a, const Place& pl, float* sc, unsigned tid) {
  for (unsigned i = tid; i < G::kScales; i += G::kThreads) {
    const unsigned q = i % G::C, cn = i / G::C, nt = cn / 16, c = cn % 16;
    const unsigned g = pl.base + local_col<G>(pl, nt, c), col = g < a.n ? g : a.n - 1;
    sc[((q * G::NT + nt) * 2 + (c & 1)) * 8 + c / 2] = e8m0_mult(a.e[col * G::C + q]);
  }
}

// Named registers a fragment (Loom keeps arrays of vectors in private memory): I a row fragment, J a column one.
#define SK_I(M) M(0) M(1) M(2) M(3)
#define SK_IJ(M) M(0, 0) M(0, 1) M(0, 2) M(0, 3) M(1, 0) M(1, 1) M(1, 2) M(1, 3) \
  M(2, 0) M(2, 1) M(2, 2) M(2, 3) M(3, 0) M(3, 1) M(3, 2) M(3, 3)

#define SK_COL(J)                                                \
  unsigned col##J = 0;                                           \
  if constexpr (G::NT > J) {                                     \
    const unsigned c = pl.base + local_col<G>(pl, J, lane & 15); \
    col##J = c < a.n ? c : a.n - 1;                              \
  }
#define SK_ACC(I, J) f32x8 v##I##J = zero8; f32x4 k##I##J = zero4;
#define SK_RAW(J) u32x4 cw0##J = zw, cw1##J = zw, nw0##J = zw, nw1##J = zw;
#define SK_XRAW(I) u32x8 cx0##I = zx, cx1##I = zx, nx0##I = zx, nx1##I = zx;
#define SK_FETCH_W(J)                                                 \
  if constexpr (G::NT > J) {                                          \
    const u32x4* src = (const u32x4*)(a.w + col##J * G::K + 32 * nq); \
    nw0##J = src[0];                                                  \
    nw1##J = src[1];                                                  \
  }
#define SK_FETCH_X(I)                                                \
  if constexpr (G::F > I) {                                          \
    const unsigned row = 16 * I + (lane & 15);                       \
    if (row < a.m) {                                                 \
      const u32x8* src = (const u32x8*)(a.x + row * G::K + 32 * nq); \
      nx0##I = src[0];                                               \
      nx1##I = src[1];                                               \
    }                                                                \
  }
#define SK_SHIFT_W(J) cw0##J = nw0##J; cw1##J = nw1##J;
#define SK_SHIFT_X(I) cx0##I = nx0##I; cx1##I = nx1##I;
#define SK_DEC(J)                                                         \
  bf16x16 a0##J = zb, a1##J = zb;                                         \
  f32x4 s0##J = zero4, s1##J = zero4;                                     \
  if constexpr (G::NT > J) {                                              \
    a0##J = fr::attach<fr::role::lhs>(e4m3_slice(cw0##J), sh);            \
    a1##J = fr::attach<fr::role::lhs>(e4m3_slice(cw1##J), sh);            \
    const f32x4* sp = (const f32x4*)(sc + ((q * G::NT + J) * 2 + h) * 8); \
    s0##J = sp[0];                                                        \
    s1##J = sp[1];                                                        \
  }
#define SK_MMA(I, J)                                                                                     \
  if constexpr (G::F > I && G::NT > J) {                                                                 \
    const bf16x16 b0 = fr::attach<fr::role::rhs>(__builtin_bit_cast(bf16x16, cx0##I), sh);              \
    const bf16x16 b1 = fr::attach<fr::role::rhs>(__builtin_bit_cast(bf16x16, cx1##I), sh);              \
    f32x8 d = loom::vector::mma(a0##J, b0, init);                                                        \
    d = loom::vector::mma(a1##J, b1, fr::attach<fr::role::init>(d, sh));                                 \
    if (t == 0 && G::T > 1) {                                                                            \
      [[loom::unroll]] for (unsigned e = 0; e < 4; e++) v##I##J[e] = loom::scalar::fmaf(d[e], s0##J[e], 0.0f); \
      k##I##J = {d[4], d[5], d[6], d[7]};                                                                \
    } else {                                                                                             \
      [[loom::unroll]] for (unsigned e = 0; e < 4; e++) {                                                \
        v##I##J[e] = loom::scalar::fmaf(d[e], s0##J[e], v##I##J[e]);                                     \
        v##I##J[4 + e] = loom::scalar::fmaf(d[4 + e], s1##J[e], v##I##J[4 + e]);                         \
      }                                                                                                  \
    }                                                                                                    \
  }
#define SK_KEEP_S(J)    \
  f32x4 r1##J = zero4;  \
  if constexpr (G::NT > J) r1##J = ((const f32x4*)(sc + ((q0 * G::NT + J) * 2 + h) * 8))[1];
#define SK_KEEP(I, J)                                                                                       \
  if constexpr (G::F > I && G::NT > J) {                                                                    \
    [[loom::unroll]] for (unsigned e = 0; e < 4; e++)                                                       \
      v##I##J[4 + e] = loom::scalar::fmaf(k##I##J[e], r1##J[e], v##I##J[4 + e]);                            \
  }
// The xor tree across waves (levels off = W/2 .. 1: v_w + v_(w + off)), then wave 0 repairs and stores.
#define SK_FIN(I, J)                                                                                        \
  if constexpr (G::F > I && G::NT > J) {                                                                    \
    if (16 * I < a.m) {                                                                                     \
      f32x8 v = v##I##J;                                                                                    \
      [[loom::unroll]] for (unsigned lv = 0; lv < G::kLevels; lv++) {                                       \
        const unsigned off = G::W >> (lv + 1);                                                              \
        if (wave >= off && wave < 2 * off) {                                                                \
          [[loom::unroll]] for (unsigned e = 0; e < 8; e++) red[((wave - off) * 8 + e) * 33 + lane] = v[e]; \
        }                                                                                                   \
        lds_barrier();                                                                                      \
        if (wave < off) {                                                                                   \
          [[loom::unroll]] for (unsigned e = 0; e < 8; e++) v[e] = v[e] + red[(wave * 8 + e) * 33 + lane];  \
        }                                                                                                   \
        lds_barrier();                                                                                      \
      }                                                                                                     \
      if (wave == 0) {                                                                                      \
        const unsigned r = 16 * I + (lane & 15);                                                            \
        [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {                                                 \
          const unsigned local = local_col<G>(pl, J, 2 * e + h), col = pl.base + local;                     \
          if (r < a.m && local < pl.width && col < a.n)                                                     \
            put(out, pl, r, local, a.n, repair<G::K>(a, v[e], r, col));                                     \
        }                                                                                                   \
      }                                                                                                     \
    }                                                                                                       \
  }

// One block: scales staged, chunks q = tau L + wave from the tile's rotation (chunk t + 1 fetched while t computes),
// elements 4-7 (rotation + 1) folding chunk rot last, then the tree and the stores.
template <class G>
LOOM_INLINE void skinny(const Args& a, const Sides& s, const Outs& out, float* sc, float* red) {
  const fr::shape<int, int> sh{16, 16};
  const f32x8 zero8 = {0, 0, 0, 0, 0, 0, 0, 0};
  const f32x4 zero4 = {0, 0, 0, 0};
  const u32x4 zw = {0, 0, 0, 0};
  const u32x8 zx = {0, 0, 0, 0, 0, 0, 0, 0};
  const bf16x16 zb = __builtin_bit_cast(bf16x16, zx);
  const f32x8 init = fr::attach<fr::role::init>(zero8, sh);
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31, h = lane >> 4;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id();
  const Place pl = place<G>(s, loom::kernel::workgroup::id.x);
  SK_I(SK_COL)
  SK_IJ(SK_ACC)
  SK_I(SK_RAW)
  SK_I(SK_XRAW)
  {
    const unsigned nq = pl.rot * G::L + wave;
    SK_I(SK_FETCH_W)
    SK_I(SK_FETCH_X)
  }
  stage_scales<G>(a, pl, sc, tid);
  lds_barrier();
  [[loom::unroll]] for (unsigned t = 0; t < G::T; t++) {
    const unsigned q = ((pl.rot + t) % G::T) * G::L + wave;
    SK_I(SK_SHIFT_W)
    SK_I(SK_SHIFT_X)
    if (t + 1 < G::T) {
      const unsigned nq = ((pl.rot + t + 1) % G::T) * G::L + wave;
      SK_I(SK_FETCH_W)
      SK_I(SK_FETCH_X)
    }
    SK_I(SK_DEC)
    SK_IJ(SK_MMA)
  }
  if constexpr (G::T > 1) {
    const unsigned q0 = pl.rot * G::L + wave;
    SK_I(SK_KEEP_S)
    SK_IJ(SK_KEEP)
  }
  SK_IJ(SK_FIN)
}

#undef SK_I
#undef SK_IJ
#undef SK_COL
#undef SK_ACC
#undef SK_RAW
#undef SK_XRAW
#undef SK_FETCH_W
#undef SK_FETCH_X
#undef SK_SHIFT_W
#undef SK_SHIFT_X
#undef SK_DEC
#undef SK_MMA
#undef SK_KEEP_S
#undef SK_KEEP
#undef SK_FIN

}  // namespace sk

#endif  // SK_SKINNY_TILE_H_
