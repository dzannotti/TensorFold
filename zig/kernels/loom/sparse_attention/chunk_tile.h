// SPDX-License-Identifier: MIT

// gfx11 bf16 WMMA pieces of the chunked attention (CUDA attn_chunks' order and rounding points, README.md).
#ifndef SA_CHUNK_TILE_H_
#define SA_CHUNK_TILE_H_

#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

namespace sc {

namespace fr = loom::vector::fragment;

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

constexpr unsigned kDims = 256, kGroup = 12, kTile = 32, kChunk = 512, kChunks = 5, kQPitch = kDims + 8, kPPitch = 40;

#define SC_V8(F) {F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7)}

static LOOM_INLINE float bits(unsigned u) { return __builtin_bit_cast(float, u); }
static LOOM_INLINE float neg_inf() { return bits(0xff800000u); }

static LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

static LOOM_INLINE void wave_fence() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::subgroup, ordering::acq_rel>();
}

static LOOM_INLINE u32x8 join(u32x4 a, u32x4 b) { return {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}; }

static LOOM_INLINE u32x8 frag(const unsigned short* p) { return join(((const u32x4*)p)[0], ((const u32x4*)p)[1]); }

static LOOM_INLINE u32x8 masked(u32x8 x, unsigned m) { return x & u32x8{m, m, m, m, m, m, m, m}; }

// d = a x b + c by v_wmma_f32_16x16x16_bf16: lane l holds a's row l % 16, b's column l % 16; d's element e is row
// 2e + l / 16, column l % 16.
static LOOM_INLINE f32x8 mma(u32x8 a, u32x8 b, f32x8 c) {
  const fr::shape<int, int> sh{16, 16};
  return loom::vector::mma(fr::attach<fr::role::lhs>(__builtin_bit_cast(bf16x16, a), sh),
                           fr::attach<fr::role::rhs>(__builtin_bit_cast(bf16x16, b), sh),
                           fr::attach<fr::role::init>(c, sh));
}

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

static LOOM_INLINE float xorf(float x, unsigned m) { return loom::kernel::subgroup::shuffle_xor(x, m, 32).value; }

// A head's max over this wave's 16 keys of the tile (lanes of a half).
static LOOM_INLINE float max16(float x) {
  x = loom::scalar::maxnumf(x, xorf(x, 1u));
  x = loom::scalar::maxnumf(x, xorf(x, 2u));
  x = loom::scalar::maxnumf(x, xorf(x, 4u));
  return loom::scalar::maxnumf(x, xorf(x, 8u));
}

// An 8-key group's sum in CUDA's warp order: pairs, pairs of pairs, then the two fours.
static LOOM_INLINE float sum8(float p) {
  float a = p + xorf(p, 1u);
  a = a + xorf(a, 2u);
  return a + xorf(a, 4u);
}

// A chunk's running state; o: this wave's 8 N tiles of 16 dims (lane: dim col; element e: head 2e + half).
struct Part {
  f32x8 m, l;
  f32x8 o0, o1, o2, o3, o4, o5, o6, o7;
};

}  // namespace sc

#endif  // SA_CHUNK_TILE_H_
