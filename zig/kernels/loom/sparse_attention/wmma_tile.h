// SPDX-License-Identifier: MIT

// gfx11 bf16 WMMA pieces of the attention core: carriers, bf16 rounding, score/value halves, softmax (README.md).
#ifndef SA_WMMA_TILE_H_
#define SA_WMMA_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>


namespace sa {

static LOOM_INLINE float bits(unsigned u) { return __builtin_bit_cast(float, u); }
static LOOM_INLINE float neg_inf() { return bits(0xff800000u); }

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

constexpr unsigned kDims = 256, kGroup = 12, kParts = 16, kTile = 16, kChunks = kDims / 16, kHalfTiles = kChunks / 2;
// q parts [3][16 heads][kQPitch] bf16; a wave's slot: its score half, then its staged values (two 16 x 16 tiles)
constexpr unsigned kQPitch = kDims + 8, kQWords = 16 * kQPitch / 2, kSlotWords = 256;

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

// fp32 to bf16, round to nearest even, NaN to 0x7fc0 (TensorFold's float_to_bf16, CUDA's __float2bfloat16_rn).
static LOOM_INLINE unsigned short to_bf16(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return f != f ? (unsigned short)0x7fc0 : (unsigned short)((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
}

// CUDA's tf_exp: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
static LOOM_INLINE float tf_exp(float x) {
  using loom::scalar::fmaf;
  const float j = fmaf(x, bits(0x3fb8aa3bu), bits(0x4b400000u)) - bits(0x4b400000u);
  float f = fmaf(j, bits(0xbf317200u), x);
  f = fmaf(j, bits(0xb5bfbe8eu), f);
  float p = bits(0x39500d01u);
  p = fmaf(p, f, bits(0x3ab60b61u));
  p = fmaf(p, f, bits(0x3c088889u));
  p = fmaf(p, f, bits(0x3d2aaaabu));
  p = fmaf(p, f, bits(0x3e2aaaabu));
  p = fmaf(p, f, 0.5f);
  p = fmaf(p, f, 1.0f);
  p = fmaf(p, f, 1.0f);
  const int i = (int)j;
  const float a = bits(i > 0 ? 0x7f000000u : 0x02000000u);
  const float b = bits(i > 0 ? (unsigned)i << 23 : (unsigned)(i + 250) << 23);
  const float r = (p * a) * b;
  return x != x ? x : (x > 104.0f ? bits(0x7f800000u) : (x < -104.0f ? 0.0f : r));
}

static LOOM_INLINE float xor16(float x) { return loom::kernel::subgroup::shuffle_xor(x, 16u, 32).value; }

// Row `qrow` (12 heads x 256 fp32) rounded to bf16 as qp[head][dim]; heads 12..15 are +0.
static LOOM_INLINE void stage_q(unsigned short* qp, const float* qrow, unsigned tid, unsigned threads) {
  for (unsigned i = tid; i < 16 * kDims; i += threads) {
    const unsigned h = i / kDims, d = i % kDims;
    qp[h * kQPitch + d] = h < kGroup ? to_bf16(qrow[h * kDims + d]) : (unsigned short)0;
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

// Side's half of a tile's raw scores x_s (lane: head col; element e: key 2e + half): its 8 chunks as one chain from +0.
static LOOM_INLINE f32x8 scores_half(Key key, int tn, const unsigned short* qp, unsigned side, unsigned col) {
  const unsigned short* const qh = qp + col * kQPitch + side * (kDims / 2);
  const unsigned short* const kh = key.k + side * (kDims / 2);
  const unsigned m = tn < (int)kTile && !key.live ? 0u : 0xffffffffu;
  f32x8 x = {0, 0, 0, 0, 0, 0, 0, 0};
  [[loom::pipeline(2)]] for (unsigned c = 0; c < kHalfTiles; c++) x = mma(masked(frag(kh + c * 16), m), frag(qh + c * 16), x);
  return x;
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

// P rounded to bf16 as the A operand (row: head), each element's head rescale f, the part's new max and sum.
struct Probs {
  u32x8 pb;
  f32x8 fe;
  float m, l;
};

// Element e of a tile's probabilities: e^(s - top) for a key of the tile and a real head, else +0.
static LOOM_INLINE float prob(f32x8 s, unsigned e, float top, int tn, unsigned col, unsigned half) {
  return (int)(2 * e + half) < tn && col < kGroup ? tf_exp(s[e] - top) : 0.0f;
}

// Element e of the parts and rescale: the pair (key 2e, key 2e + 1) of head col from this lane and its partner.
struct Elem {
  unsigned pb;
  float fe;
};

static LOOM_INLINE Elem elem(float pe, float f, unsigned e, unsigned half) {
  const float f0 = loom::kernel::subgroup::broadcast(f, 2 * e), f1 = loom::kernel::subgroup::broadcast(f, 2 * e + 1);
  const float other = xor16(pe);
  const unsigned x0 = to_bf16(half ? other : pe), x1 = to_bf16(half ? pe : other);
  return {(x1 << 16) | x0, half ? f1 : f0};
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
  const float f = tn > 0 ? tf_exp(m - top) : 1.0f;
  const Elem e0 = elem(p0, f, 0, half), e1 = elem(p1, f, 1, half), e2 = elem(p2, f, 2, half), e3 = elem(p3, f, 3, half);
  const Elem e4 = elem(p4, f, 4, half), e5 = elem(p5, f, 5, half), e6 = elem(p6, f, 6, half), e7 = elem(p7, f, 7, half);
  return {u32x8{e0.pb, e1.pb, e2.pb, e3.pb, e4.pb, e5.pb, e6.pb, e7.pb},
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

// One N tile: o = mma(bf16 p, v, o f).
static LOOM_INLINE f32x8 value_tile(f32x8 o, Probs pr, u32x8 b) { return mma(pr.pb, b, o * pr.fe); }

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
