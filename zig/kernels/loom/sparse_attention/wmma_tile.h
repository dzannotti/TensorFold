// SPDX-License-Identifier: MIT

// gfx11 bf16 WMMA pieces of the attention core: carriers, exact splits, score/value halves, softmax (README.md).
#ifndef SA_WMMA_TILE_H_
#define SA_WMMA_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>

#include "exp_exact.h"

namespace sa {

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

constexpr unsigned kDims = 256, kGroup = 12, kParts = 16, kTile = 16, kChunks = kDims / 16, kHalfTiles = kChunks / 2;
// q parts [3][16 heads][kQPitch] bf16; a wave's slot: its score half, then its staged values (two 16 x 16 tiles)
constexpr unsigned kQPitch = kDims + 8, kQWords = 3 * 16 * kQPitch / 2, kSlotWords = 256;

static LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

static LOOM_INLINE void wave_fence() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::subgroup, ordering::acq_rel>();
}

static LOOM_INLINE u32x8 join(u32x4 a, u32x4 b) { return {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}; }

// 16 bf16 at p (16-byte aligned) as one carrier.
static LOOM_INLINE u32x8 frag(const unsigned short* p) { return *(const u32x8*)p; }

// v_wmma_f32_16x16x16_bf16: lane l holds a's row l % 16, b's column l % 16; d's element e is row 2e + l / 16.
static LOOM_INLINE f32x8 mma(u32x8 a, u32x8 b, f32x8 c) {
  const fr::shape<int, int> sh{16, 16};
  return loom::vector::mma(fr::attach<fr::role::lhs>(__builtin_bit_cast(bf16x16, a), sh),
                           fr::attach<fr::role::rhs>(__builtin_bit_cast(bf16x16, b), sh),
                           fr::attach<fr::role::init>(c, sh));
}

static LOOM_INLINE unsigned top16(float x) { return __builtin_bit_cast(unsigned, x) & 0xffff0000u; }

// x = hi + mid + lo exactly, each a bf16 in the upper half of its word, +0 for a zero part.
struct Split {
  unsigned hi, mid, lo;
};

static LOOM_INLINE Split split(float x) {
  const unsigned a = top16(x);
  const float r = x - bits(a);
  const unsigned b = top16(r);
  const float s = r - bits(b);
  return {top16(bits(a) + 0.0f), top16(bits(b) + 0.0f), top16(s + 0.0f)};
}

static LOOM_INLINE unsigned pack(unsigned lo_elem, unsigned hi_elem) { return hi_elem | (lo_elem >> 16); }

static LOOM_INLINE float xor16(float x) { return loom::kernel::subgroup::shuffle_xor(x, 16u, 32).value; }

// Row `qrow` (12 heads x 256 fp32) times `scale`, split, as qp[part][head][dim]; heads 12..15 are +0.
static LOOM_INLINE void stage_q(unsigned short* qp, const float* qrow, float scale, unsigned tid, unsigned threads) {
  for (unsigned i = tid; i < 16 * kDims; i += threads) {
    const unsigned h = i / kDims, d = i % kDims;
    const Split s = h < kGroup ? split(qrow[h * kDims + d] * scale) : Split{0, 0, 0};
    qp[(0 * 16 + h) * kQPitch + d] = (unsigned short)(s.hi >> 16);
    qp[(1 * 16 + h) * kQPitch + d] = (unsigned short)(s.mid >> 16);
    qp[(2 * 16 + h) * kQPitch + d] = (unsigned short)(s.lo >> 16);
  }
}

// One row's cache and key list: keys ids[i] when listed, else i.
struct Rows {
  const unsigned short *k, *v;
  const int* ids;
  long long base;
  bool listed;
};

// This lane's key of a tile (key col; a live key's row stands in past the tile's tn keys, loaded as +0).
struct Key {
  const unsigned short *k, *v;
  bool live;
};

static LOOM_INLINE Key key_of(Rows rw, int t0, int tn, unsigned col) {
  const bool live = (int)col < tn;
  const int at = t0 + (live ? (int)col : 0);
  const long long key = rw.listed ? rw.ids[at] : at;
  return {rw.k + (rw.base + key) * kDims, rw.v + (rw.base + key) * kDims, live};
}

static LOOM_INLINE u32x8 masked(u32x8 x, unsigned m) { return x & u32x8{m, m, m, m, m, m, m, m}; }

struct Sums {
  f32x8 hs, ch;
};

// Chunk C: d_C = hi parts from +0 added to the running hi sum (the first one starts it); mid, lo onto the chain.
template <unsigned C>
static LOOM_INLINE Sums qk_chunk(Sums a, u32x8 kc, const unsigned short* qh) {
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32x8 q0 = frag(qh + C * 16), q1 = frag(qh + 16 * kQPitch + C * 16), q2 = frag(qh + 32 * kQPitch + C * 16);
  const f32x8 dc = mma(kc, q0, zero);
  a.hs = C == 0 ? dc : a.hs + dc;
  a.ch = mma(kc, q1, a.ch);
  a.ch = mma(kc, q2, a.ch);
  return a;
}

// Side's half of a tile's scores x_s (lane: head col; element e: key 2e + half), as README.md's Sum order.
static LOOM_INLINE f32x8 scores_half(Key key, int tn, const unsigned short* qp, unsigned side, unsigned col) {
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const unsigned short* const qh = qp + col * kQPitch + side * (kDims / 2);
  const unsigned short* const kh = key.k + side * (kDims / 2);
  const unsigned m = tn < (int)kTile && !key.live ? 0u : 0xffffffffu;
  Sums a = qk_chunk<0>(Sums{zero, zero}, masked(frag(kh), m), qh);
  [[loom::pipeline(2)]] for (unsigned c = 1; c < kHalfTiles; c++) {
    const u32x8 kc = masked(frag(kh + c * 16), m);
    const u32x8 q0 = frag(qh + c * 16), q1 = frag(qh + 16 * kQPitch + c * 16), q2 = frag(qh + 32 * kQPitch + c * 16);
    a.hs = a.hs + mma(kc, q0, zero);
    a.ch = mma(kc, q2, mma(kc, q1, a.ch));
  }
  return a.hs + a.ch;
}

// This lane's value rows for values_half (+0 past the tile's keys).
struct VHalf {
  const unsigned short* p;
  unsigned m;
};

static LOOM_INLINE VHalf load_v(Key key, unsigned side, unsigned half, int tn) {
  return {key.v + side * (kDims / 2) + half * 16, tn < (int)kTile && !key.live ? 0u : 0xffffffffu};
}

static LOOM_INLINE void put8(unsigned* slot, unsigned lane, f32x8 x) {
  u32x4* p = (u32x4*)(slot + lane * 8);
  const u32x8 b = __builtin_bit_cast(u32x8, x);
  p[0] = u32x4{b[0], b[1], b[2], b[3]};
  p[1] = u32x4{b[4], b[5], b[6], b[7]};
}

static LOOM_INLINE f32x8 get8(const unsigned* slot, unsigned lane) {
  const u32x4* p = (const u32x4*)(slot + lane * 8);
  return __builtin_bit_cast(f32x8, join(p[0], p[1]));
}

// A part's running state; o: this wave's 8 N tiles (lane: dim col of the tile; element e: head 2e + half).
struct Part {
  float m, l;
  f32x8 o0, o1, o2, o3, o4, o5, o6, o7;
};

static LOOM_INLINE Part start() {
  const f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
  return {-bits(0x7f800000u), 0.0f, z, z, z, z, z, z, z, z};
}

// P's three parts as A operands (row: head), each element's head rescale f, the part's new running max and sum.
struct Probs {
  u32x8 ph, pm, pl;
  f32x8 fe;
  float m, l;
};

// Element e of a tile's probabilities: e^(s - top) for a key of the tile and a real head, else +0.
static LOOM_INLINE float prob(f32x8 s, unsigned e, float top, int tn, unsigned col, unsigned half) {
  return (int)(2 * e + half) < tn && col < kGroup ? exp_exact(s[e] - top) : 0.0f;
}

// Element e of the parts and rescale: the pair (key 2e, key 2e + 1) of head col from this lane and its partner.
struct Elem {
  unsigned ph, pm, pl;
  float fe;
};

static LOOM_INLINE Elem elem(float pe, float f, unsigned e, unsigned half) {
  const float f0 = loom::kernel::subgroup::broadcast(f, 2 * e), f1 = loom::kernel::subgroup::broadcast(f, 2 * e + 1);
  const float other = xor16(pe);
  const Split x0 = split(half ? other : pe), x1 = split(half ? pe : other);
  return {pack(x0.hi, x1.hi), pack(x0.mid, x1.mid), pack(x0.lo, x1.lo), half ? f1 : f0};
}

// Online softmax step over a tile's scores s (keys past tn and heads past 12 give +0).
static LOOM_INLINE Probs softmax(float m, float l, f32x8 s, int tn, unsigned col, unsigned half) {
  float top = -bits(0x7f800000u);
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++)
    if ((int)(2 * e + half) < tn) top = loom::scalar::maxnumf(top, s[e]);
  top = loom::scalar::maxnumf(top, xor16(top));
  top = loom::scalar::maxnumf(top, m);
  const float p0 = prob(s, 0, top, tn, col, half), p1 = prob(s, 1, top, tn, col, half);
  const float p2 = prob(s, 2, top, tn, col, half), p3 = prob(s, 3, top, tn, col, half);
  const float p4 = prob(s, 4, top, tn, col, half), p5 = prob(s, 5, top, tn, col, half);
  const float p6 = prob(s, 6, top, tn, col, half), p7 = prob(s, 7, top, tn, col, half);
  float sum = ((((((p0 + p1) + p2) + p3) + p4) + p5) + p6) + p7;
  sum = sum + xor16(sum);
  // a tile past the part's keys leaves o, m and l as they are: f = 1, its probabilities and values are +0
  const float f = tn > 0 ? exp_exact(m - top) : 1.0f;
  const Elem e0 = elem(p0, f, 0, half), e1 = elem(p1, f, 1, half), e2 = elem(p2, f, 2, half), e3 = elem(p3, f, 3, half);
  const Elem e4 = elem(p4, f, 4, half), e5 = elem(p5, f, 5, half), e6 = elem(p6, f, 6, half), e7 = elem(p7, f, 7, half);
  return {u32x8{e0.ph, e1.ph, e2.ph, e3.ph, e4.ph, e5.ph, e6.ph, e7.ph},
          u32x8{e0.pm, e1.pm, e2.pm, e3.pm, e4.pm, e5.pm, e6.pm, e7.pm},
          u32x8{e0.pl, e1.pl, e2.pl, e3.pl, e4.pl, e5.pl, e6.pl, e7.pl},
          f32x8{e0.fe, e1.fe, e2.fe, e3.fe, e4.fe, e5.fe, e6.fe, e7.fe}, tn > 0 ? top : m, tn > 0 ? l * f + sum : l};
}

// Staged values: element (dim d, key j) of tile b at (b * 16 + d) * 16 + (j ^ 8 (d & 1)).
static LOOM_INLINE void stage_v(unsigned short* tw, u32x8 w, unsigned col, unsigned half) {
  unsigned short* const t = tw + half * 256;
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {
    t[(2 * e) * 16 + col] = (unsigned short)w[e];
    t[(2 * e + 1) * 16 + (col ^ 8)] = (unsigned short)(w[e] >> 16);
  }
}

static LOOM_INLINE u32x8 staged(const unsigned short* tw, unsigned b, unsigned col) {
  const unsigned short* const row = tw + (b * 16 + col) * 16;
  const unsigned s = (col & 1) * 8;
  return join(*(const u32x4*)(row + s), *(const u32x4*)(row + (8 ^ s)));
}

// One N tile: d = lo, mid, hi parts of P V from +0, o = fma(o, f, d).
static LOOM_INLINE f32x8 value_tile(f32x8 o, Probs pr, u32x8 b) {
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x8 d = mma(pr.pl, b, zero);
  d = mma(pr.pm, b, d);
  d = mma(pr.ph, b, d);
  using loom::scalar::fmaf;
  return {fmaf(o[0], pr.fe[0], d[0]), fmaf(o[1], pr.fe[1], d[1]), fmaf(o[2], pr.fe[2], d[2]), fmaf(o[3], pr.fe[3], d[3]),
          fmaf(o[4], pr.fe[4], d[4]), fmaf(o[5], pr.fe[5], d[5]), fmaf(o[6], pr.fe[6], d[6]), fmaf(o[7], pr.fe[7], d[7])};
}

// The tile's values for side's dims: half h stages N tile 2j + h of key col; round j updates o tiles 2j, 2j + 1.
static LOOM_INLINE Part values_half(Part st, VHalf vw, Probs pr, unsigned short* tw, unsigned col, unsigned half) {
  stage_v(tw, masked(frag(vw.p), vw.m), col, half);
  wave_fence();
  st.o0 = value_tile(st.o0, pr, staged(tw, 0, col));
  st.o1 = value_tile(st.o1, pr, staged(tw, 1, col));
  wave_fence();
  stage_v(tw, masked(frag(vw.p + 32), vw.m), col, half);
  wave_fence();
  st.o2 = value_tile(st.o2, pr, staged(tw, 0, col));
  st.o3 = value_tile(st.o3, pr, staged(tw, 1, col));
  wave_fence();
  stage_v(tw, masked(frag(vw.p + 64), vw.m), col, half);
  wave_fence();
  st.o4 = value_tile(st.o4, pr, staged(tw, 0, col));
  st.o5 = value_tile(st.o5, pr, staged(tw, 1, col));
  wave_fence();
  stage_v(tw, masked(frag(vw.p + 96), vw.m), col, half);
  wave_fence();
  st.o6 = value_tile(st.o6, pr, staged(tw, 0, col));
  st.o7 = value_tile(st.o7, pr, staged(tw, 1, col));
  wave_fence();
  st.m = pr.m;
  st.l = pr.l;
  return st;
}

// A part's o into parts_o[row0 + head * kParts][dims] (this wave's 128 dims), its m and l into parts_ms.
static LOOM_INLINE void store_part(Part st, float* parts_o, float* parts_ms, long long row0, unsigned side,
                                   unsigned col, unsigned half) {
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {
    const unsigned h = 2 * e + half;
    if (h < kGroup) {
      float* const o = parts_o + (row0 + h * kParts) * kDims + side * (kDims / 2) + col;
      o[0 * 16] = st.o0[e];
      o[1 * 16] = st.o1[e];
      o[2 * 16] = st.o2[e];
      o[3 * 16] = st.o3[e];
      o[4 * 16] = st.o4[e];
      o[5 * 16] = st.o5[e];
      o[6 * 16] = st.o6[e];
      o[7 * 16] = st.o7[e];
    }
  }
  if (side == 0 && half == 0 && col < kGroup) {
    parts_ms[(row0 + col * kParts) * 2] = st.m;
    parts_ms[(row0 + col * kParts) * 2 + 1] = st.l;
  }
}

}  // namespace sa

#endif  // SA_WMMA_TILE_H_
