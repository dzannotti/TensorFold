// SPDX-License-Identifier: MIT

// gfx11 bf16 WMMA pieces of the attention core: carriers, bf16 rounding, tf_exp, a 32-key tile, the part merge (README.md).
#ifndef SA_WMMA_TILE_H_
#define SA_WMMA_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

namespace sa {

static LOOM_INLINE float f32_bits(unsigned u) { return __builtin_bit_cast(float, u); }
static LOOM_INLINE float neg_inf() { return f32_bits(0xff800000u); }

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

constexpr unsigned kDims = 256, kGroup = 12, kTile = 32, kParts = 4, kNTiles = kDims / 16;
// LDS words: q as bf16 [16 heads][kQPitch], then a wave's two value slots (two 16 x 16 bf16 tiles each); the merge reuses them
constexpr unsigned kQPitch = kDims + 8, kQWords = 16 * kQPitch / 2, kSlotWords = 512;
constexpr unsigned kArenaWords = kQWords + kParts * kSlotWords;
// merge scratch: elements 0..3 [16 tiles][32 lanes][4], elements 4..5 [16][32][2], then m and l of 16 heads
constexpr unsigned kMergeLo = 0, kMergeHi = kNTiles * 32 * 4, kMergeMs = kMergeHi + kNTiles * 32 * 2;
static_assert(kMergeMs + 32 <= kArenaWords, "merge scratch fits the arena");

static LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

static LOOM_INLINE void wave_fence() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::subgroup, ordering::acq_rel>();
}

static LOOM_INLINE u32x8 join(u32x4 a, u32x4 b) { return {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}; }

static LOOM_INLINE u32x8 frag(const unsigned short* p) { return *(const u32x8*)p; }

// v_wmma_f32_16x16x16_bf16: lane l holds a's row l % 16, b's column l % 16; d's element e is row 2e + l / 16.
static LOOM_INLINE f32x8 mma(u32x8 a, u32x8 b, f32x8 c) {
  const fr::shape<int, int> sh{16, 16};
  return loom::vector::mma(fr::attach<fr::role::lhs>(__builtin_bit_cast(bf16x16, a), sh),
                           fr::attach<fr::role::rhs>(__builtin_bit_cast(bf16x16, b), sh),
                           fr::attach<fr::role::init>(c, sh));
}

// fp32 to bf16, round to nearest even, NaN to 0x7fc0 (TensorFold's float_to_bf16, CUDA's __float2bfloat16_rn).
static LOOM_INLINE unsigned to_bf16(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  const unsigned r = (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
  return f != f ? 0x7fc0u : r;
}

// CUDA's tf_exp before its range guards: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
static LOOM_INLINE float exp_core(float x) {
  using loom::scalar::fmaf;
  const float j = fmaf(x, f32_bits(0x3fb8aa3bu), f32_bits(0x4b400000u)) - f32_bits(0x4b400000u);
  float f = fmaf(j, f32_bits(0xbf317200u), x);
  f = fmaf(j, f32_bits(0xb5bfbe8eu), f);
  float p = f32_bits(0x39500d01u);
  p = fmaf(p, f, f32_bits(0x3ab60b61u));
  p = fmaf(p, f, f32_bits(0x3c088889u));
  p = fmaf(p, f, f32_bits(0x3d2aaaabu));
  p = fmaf(p, f, f32_bits(0x3e2aaaabu));
  p = fmaf(p, f, 0.5f);
  p = fmaf(p, f, 1.0f);
  p = fmaf(p, f, 1.0f);
  const int i = (int)j;
  const float a = f32_bits(i > 0 ? 0x7f000000u : 0x02000000u);
  const float b = f32_bits(i > 0 ? (unsigned)i << 23 : (unsigned)(i + 250) << 23);
  return (p * a) * b;
}

// tf_exp(x) for x <= 0 or -inf (x - max in the softmax and the merge): its only live guard is 0 below -104.
static LOOM_INLINE float exp_neg(float x) {
  const float r = exp_core(x);
  return x < -104.0f ? 0.0f : r;
}

static LOOM_INLINE float xor16(float x) { return loom::kernel::subgroup::shuffle_xor(x, 16u, 32).value; }
static LOOM_INLINE float bcast(float x, unsigned lane) { return loom::kernel::subgroup::broadcast(x, lane); }

// Head 2e + half's copy of a per-head value held by lane = head.
static LOOM_INLINE float per_head(float x, unsigned e, unsigned half) {
  const float a = bcast(x, 2 * e), b = bcast(x, 2 * e + 1);
  return half ? b : a;
}

static LOOM_INLINE f32x8 heads8(float x, unsigned half) {
  return {per_head(x, 0, half), per_head(x, 1, half), per_head(x, 2, half), per_head(x, 3, half),
          per_head(x, 4, half), per_head(x, 5, half), per_head(x, 6, half), per_head(x, 7, half)};
}

// Row `qrow` (12 heads x 256 fp32) rounded to bf16 as qp[head][dim]; heads 12..15 are +0.
static LOOM_INLINE void stage_q(unsigned short* qp, const float* qrow, unsigned tid, unsigned threads) {
  for (unsigned i = tid; i < 16 * kDims; i += threads) {
    const unsigned h = i / kDims, d = i % kDims;
    qp[h * kQPitch + d] = h < kGroup ? (unsigned short)to_bf16(qrow[h * kDims + d]) : (unsigned short)0;
  }
}

// One row's keys of one KV head: ids[i] when listed, else i.
struct Rows {
  const unsigned short *k, *v;
  const int* ids;
  long long base;
  bool listed;
};

static LOOM_INLINE long long key_at(Rows rw, int i) {
  const long long listed = rw.ids[rw.listed ? i : 0];
  return rw.base + (rw.listed ? listed : (long long)i);
}

// A part's running softmax state; o: N tile j's accumulator (lane: dim col of the tile; element e: head 2e + half).
struct Acc {
  float m, l;
  f32x8 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15;
};

static LOOM_INLINE Acc start() {
  const f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
  return {neg_inf(), 0.0f, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z};
}

// Scores of the tile's 16-key fragment s (element e: key 2e + half of the fragment) scaled, -inf past the tile's tn keys.
static LOOM_INLINE f32x8 scaled(f32x8 s, unsigned frag0, int tn, unsigned half, float scale) {
#define SA_X(e) ((int)(frag0 + 2 * (e) + half) < tn ? s[e] * scale : neg_inf())
  return {SA_X(0), SA_X(1), SA_X(2), SA_X(3), SA_X(4), SA_X(5), SA_X(6), SA_X(7)};
#undef SA_X
}

static LOOM_INLINE float max8(f32x8 x, float m) {
  using loom::scalar::maxnumf;
  return maxnumf(maxnumf(maxnumf(maxnumf(m, x[0]), maxnumf(x[1], x[2])), maxnumf(maxnumf(x[3], x[4]), x[5])),
                 maxnumf(x[6], x[7]));
}

static LOOM_INLINE float prob(float x, float top) { return exp_neg(x - top); }

static LOOM_INLINE f32x8 probs(f32x8 x, float top) {
  return {prob(x[0], top), prob(x[1], top), prob(x[2], top), prob(x[3], top),
          prob(x[4], top), prob(x[5], top), prob(x[6], top), prob(x[7], top)};
}

static LOOM_INLINE float sum8(float s, f32x8 p) {
  return ((((((((s + p[0]) + p[1]) + p[2]) + p[3]) + p[4]) + p[5]) + p[6]) + p[7]);
}

// p (finite, >= 0) to bf16, round to nearest even.
static LOOM_INLINE unsigned rnd16(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

// Element e of the A operand: keys (2e, 2e + 1) of head col as bf16, from this lane and its xor-16 partner.
static LOOM_INLINE unsigned pair(float pe, unsigned half) {
  const unsigned own = rnd16(pe);
  const unsigned other = __builtin_bit_cast(unsigned, xor16(__builtin_bit_cast(float, own)));
  const unsigned lo = half ? other : own, hi = half ? own : other;
  return (hi << 16) | lo;
}

static LOOM_INLINE u32x8 operand(f32x8 p, unsigned half) {
  return {pair(p[0], half), pair(p[1], half), pair(p[2], half), pair(p[3], half),
          pair(p[4], half), pair(p[5], half), pair(p[6], half), pair(p[7], half)};
}

// Staged values: element (dim d, key j) of fragment b's tile at (b * 16 + d) * 16 + (j ^ 8 (d & 1)).
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

// The tile's probabilities as A operands and each PV element's rescale.
struct Step {
  u32x8 p0, p1;
  f32x8 fe;
};

// N tile j from this lane's prefetched 16 dims w (fragment half), staged in slot j % 2: o = mma(p1, v1, mma(p0, v0, o f)).
static LOOM_INLINE f32x8 value_tile(f32x8 o, Step st, u32x8 w, unsigned j, unsigned short* tw, unsigned col,
                                    unsigned half) {
  unsigned short* const slot = tw + (j & 1) * 512;
  stage_v(slot, w, col, half);
  wave_fence();
  o = o * st.fe;
  o = mma(st.p0, staged(slot, 0, col), o);
  return mma(st.p1, staged(slot, 1, col), o);
}

// One tile of tn (1..32) keys from list index t0 of the part: scores, the online softmax step, the values.
static LOOM_INLINE Acc tile(Acc a, Rows rw, int t0, int tn, const unsigned short* qp, unsigned short* tw, unsigned col,
                            unsigned half, float scale) {
  const int last = t0 + tn - 1;
  const int i0 = t0 + (int)col < last ? t0 + (int)col : last, i1 = t0 + 16 + (int)col < last ? t0 + 16 + (int)col : last;
  const long long key0 = key_at(rw, i0), key1 = key_at(rw, i1);
  const unsigned short* const k0 = rw.k + key0 * kDims;
  const unsigned short* const k1 = rw.k + key1 * kDims;
  const unsigned short* const qc = qp + col * kQPitch;
  f32x8 s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
  u32x8 a0 = frag(k0), a1 = frag(k1), b0 = frag(k0 + 16), b1 = frag(k1 + 16);
  [[loom::unroll]] for (unsigned c = 0; c < kNTiles; c += 2) {
    const u32x8 qa = frag(qc + 16 * c);
    s0 = mma(a0, qa, s0);
    s1 = mma(a1, qa, s1);
    if (c + 2 < kNTiles) {
      a0 = frag(k0 + 16 * (c + 2));
      a1 = frag(k1 + 16 * (c + 2));
    }
    const u32x8 qb = frag(qc + 16 * (c + 1));
    s0 = mma(b0, qb, s0);
    s1 = mma(b1, qb, s1);
    if (c + 3 < kNTiles) {
      b0 = frag(k0 + 16 * (c + 3));
      b1 = frag(k1 + 16 * (c + 3));
    }
  }
  const f32x8 x0 = scaled(s0, 0, tn, half, scale), x1 = scaled(s1, 16, tn, half, scale);
  float tm = max8(x1, max8(x0, neg_inf()));
  tm = loom::scalar::maxnumf(tm, xor16(tm));
  const bool active = tm != neg_inf();
  const float next = active ? loom::scalar::maxnumf(a.m, tm) : a.m;
  const float ea = exp_neg(a.m - next);
  const float alpha = active ? (a.m == neg_inf() ? 0.0f : ea) : 1.0f;
  const f32x8 p0 = probs(x0, next), p1 = probs(x1, next);
  float sum = sum8(sum8(0.0f, p0), p1);
  sum = sum + xor16(sum);
  const Step st = {operand(p0, half), operand(p1, half), heads8(alpha, half)};
  const unsigned short* const vrow = rw.v + (half ? key1 : key0) * kDims;
  a.m = next;
  a.l = loom::scalar::fmaf(a.l, alpha, sum);
  u32x8 w0 = frag(vrow), w1 = frag(vrow + 16), w2 = frag(vrow + 32);
#define SA_V(j, W)                                                \
  a.o##j = value_tile(a.o##j, st, W, j, tw, col, half);           \
  if ((j) + 3 < kNTiles) W = frag(vrow + 16 * ((j) + 3));
  SA_V(0, w0) SA_V(1, w1) SA_V(2, w2) SA_V(3, w0) SA_V(4, w1) SA_V(5, w2) SA_V(6, w0) SA_V(7, w1)
  SA_V(8, w2) SA_V(9, w0) SA_V(10, w1) SA_V(11, w2) SA_V(12, w0) SA_V(13, w1) SA_V(14, w2) SA_V(15, w0)
#undef SA_V
  wave_fence();
  return a;
}

// Merge weights of a part (m_k, l_k) into the running (mm, ll), CUDA's fn_attn_merge step: oo = fma(o, b, oo a).
struct Weights {
  float a, b, m, l;
};

static LOOM_INLINE Weights weights(float mm, float ll, float m, float l) {
  const bool active = l > 0.0f;
  const float next = active ? loom::scalar::maxnumf(mm, m) : mm;
  const float ea = exp_neg(mm - next), eb = exp_neg(m - next);
  const float a = active ? (mm == neg_inf() ? 0.0f : ea) : 1.0f;
  const float b = active ? eb : 0.0f;
  return {a, b, next, loom::scalar::fmaf(l, b, ll * a)};
}

// fma(o, b, oo a) per element where take, else oo unchanged.
static LOOM_INLINE float fold1(float oo, float o, float a, float b, bool take) {
  const float t = loom::scalar::fmaf(o, b, oo * a);
  return take ? t : oo;
}

static LOOM_INLINE f32x8 fold(f32x8 oo, f32x8 o, f32x8 a, f32x8 b, bool take) {
#define SA_F(e) fold1(oo[e], o[e], a[e], b[e], take)
  return {SA_F(0), SA_F(1), SA_F(2), SA_F(3), SA_F(4), SA_F(5), SA_F(6), SA_F(7)};
#undef SA_F
}

static LOOM_INLINE void put_tile(float* sc, unsigned j, unsigned lane, f32x8 o) {
  *(f32x4*)(sc + kMergeLo + (j * 32 + lane) * 4) = f32x4{o[0], o[1], o[2], o[3]};
  *(f32x2*)(sc + kMergeHi + (j * 32 + lane) * 2) = f32x2{o[4], o[5]};
}

static LOOM_INLINE f32x8 get_tile(const float* sc, unsigned j, unsigned lane) {
  const f32x4 lo = *(const f32x4*)(sc + kMergeLo + (j * 32 + lane) * 4);
  const f32x2 hi = *(const f32x2*)(sc + kMergeHi + (j * 32 + lane) * 2);
  return {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, 0.0f, 0.0f};
}

static LOOM_INLINE void put_part(float* sc, Acc a, unsigned lane, unsigned col, unsigned half) {
  put_tile(sc, 0, lane, a.o0);
  put_tile(sc, 1, lane, a.o1);
  put_tile(sc, 2, lane, a.o2);
  put_tile(sc, 3, lane, a.o3);
  put_tile(sc, 4, lane, a.o4);
  put_tile(sc, 5, lane, a.o5);
  put_tile(sc, 6, lane, a.o6);
  put_tile(sc, 7, lane, a.o7);
  put_tile(sc, 8, lane, a.o8);
  put_tile(sc, 9, lane, a.o9);
  put_tile(sc, 10, lane, a.o10);
  put_tile(sc, 11, lane, a.o11);
  put_tile(sc, 12, lane, a.o12);
  put_tile(sc, 13, lane, a.o13);
  put_tile(sc, 14, lane, a.o14);
  put_tile(sc, 15, lane, a.o15);
  if (half == 0) {
    sc[kMergeMs + col * 2] = a.m;
    sc[kMergeMs + col * 2 + 1] = a.l;
  }
}

// Wave 0's own part as the merge's first step from (-inf, 0, +0); other waves (take false) keep theirs.
static LOOM_INLINE Acc merge_own(Acc own, unsigned half, bool take) {
  const Weights w = weights(neg_inf(), 0.0f, own.m, own.l);
  const f32x8 a = heads8(w.a, half), b = heads8(w.b, half), z = {0, 0, 0, 0, 0, 0, 0, 0};
  const float m = own.m, l = own.l;
  own.m = take ? w.m : m;
  own.l = take ? w.l : l;
#define SA_M(j) own.o##j = take ? fold(z, own.o##j, a, b, true) : own.o##j;
  SA_M(0) SA_M(1) SA_M(2) SA_M(3) SA_M(4) SA_M(5) SA_M(6) SA_M(7)
  SA_M(8) SA_M(9) SA_M(10) SA_M(11) SA_M(12) SA_M(13) SA_M(14) SA_M(15)
#undef SA_M
  return own;
}

// The next part's tiles, m and l from the merge scratch folded into the running merge (take: wave 0).
static LOOM_INLINE Acc merge_lds(Acc acc, const float* sc, unsigned lane, unsigned half, bool take) {
  const Weights w = weights(acc.m, acc.l, sc[kMergeMs + (lane & 15) * 2], sc[kMergeMs + (lane & 15) * 2 + 1]);
  const f32x8 a = heads8(w.a, half), b = heads8(w.b, half);
  const float m = acc.m, l = acc.l;
  acc.m = take ? w.m : m;
  acc.l = take ? w.l : l;
#define SA_M(j) acc.o##j = fold(acc.o##j, get_tile(sc, j, lane), a, b, take);
  SA_M(0) SA_M(1) SA_M(2) SA_M(3) SA_M(4) SA_M(5) SA_M(6) SA_M(7)
  SA_M(8) SA_M(9) SA_M(10) SA_M(11) SA_M(12) SA_M(13) SA_M(14) SA_M(15)
#undef SA_M
  return acc;
}

// out[head][16 j + col] = bf16(oo / ll) kept as fp32, heads 2e + half for e < 6 (the 12 real heads).
static LOOM_INLINE void store_tile(float* orow, unsigned j, f32x8 oo, f32x8 ll, unsigned col, unsigned half) {
  [[loom::unroll]] for (unsigned e = 0; e < 6; e++) {
    const float y = f32_bits(to_bf16(oo[e] / ll[e]) << 16);
    const float x = ll[e] > 0.0f ? y : 0.0f;
    orow[(2 * e + half) * kDims + 16 * j + col] = x;
  }
}

static LOOM_INLINE void store_out(float* orow, Acc acc, unsigned col, unsigned half) {
  const f32x8 ll = heads8(acc.l, half);
#define SA_S(j) store_tile(orow, j, acc.o##j, ll, col, half);
  SA_S(0) SA_S(1) SA_S(2) SA_S(3) SA_S(4) SA_S(5) SA_S(6) SA_S(7)
  SA_S(8) SA_S(9) SA_S(10) SA_S(11) SA_S(12) SA_S(13) SA_S(14) SA_S(15)
#undef SA_S
}

}  // namespace sa

#endif  // SA_WMMA_TILE_H_
