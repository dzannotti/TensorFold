// SPDX-License-Identifier: MIT
// Bare LDS -> WMMA inner loop of the K4 tile: 8 waves, 128 x 32 bf16 x and w stages (pitch 40) double-buffered in
// LDS, filled once; each wave runs 8 row fragments x 1 column fragment per K16 slice, no barriers, no global traffic.
#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
namespace fr = loom::vector::fragment;
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
#ifndef BARE_ARENA
#define BARE_ARENA 4
#endif
constexpr unsigned kRows = 128, kPitch = 40, kBuf = kRows * kPitch, kArena = BARE_ARENA * kBuf;
struct Acc { f32x8 a0, a1, a2, a3, a4, a5, a6, a7; };
static loom::kernel::configuration cfg(unsigned groups) { return {{groups, 1, 1}, {256, 1, 1}}; }

#define MMA(I) d.a##I = loom::vector::mma(fr::load<fr::role::lhs, bf16x16>(xv, {16 * I, 16 * T}, sh), bw, \
                                          fr::attach<fr::role::init>(d.a##I, sh));
#ifdef BARE_COLS2
#define MMA2(I, C, A, B) d.a##I = loom::vector::mma(A, B, fr::attach<fr::role::init>(d.a##I, sh));
template <unsigned T>
LOOM_INLINE void slice(Acc& d, const __bf16* xp, const __bf16* wp) {
  const fr::shape<int, int> sh{16, 16};
  const auto xv = loom::buffer::view<kRows, kPitch>(xp, {});
  const auto wv = loom::buffer::view<32, 32>(wp, {}, loom::encoding::layout::strided(1, kPitch));
  const bf16x16 b0 = fr::load<fr::role::rhs, bf16x16>(wv, {16 * T, 0}, sh);
  const bf16x16 b1 = fr::load<fr::role::rhs, bf16x16>(wv, {16 * T, 16}, sh);
  const bf16x16 x0 = fr::load<fr::role::lhs, bf16x16>(xv, {0, 16 * T}, sh);
  const bf16x16 x1 = fr::load<fr::role::lhs, bf16x16>(xv, {16, 16 * T}, sh);
  const bf16x16 x2 = fr::load<fr::role::lhs, bf16x16>(xv, {32, 16 * T}, sh);
  const bf16x16 x3 = fr::load<fr::role::lhs, bf16x16>(xv, {48, 16 * T}, sh);
  MMA2(0, 0, x0, b0) MMA2(1, 1, x0, b1) MMA2(2, 0, x1, b0) MMA2(3, 1, x1, b1)
  MMA2(4, 0, x2, b0) MMA2(5, 1, x2, b1) MMA2(6, 0, x3, b0) MMA2(7, 1, x3, b1)
}
#else
template <unsigned T>
LOOM_INLINE void slice(Acc& d, const __bf16* xp, const __bf16* wp) {
  const fr::shape<int, int> sh{16, 16};
  const auto xv = loom::buffer::view<kRows, kPitch>(xp, {});
  const auto wv = loom::buffer::view<32, 16>(wp, {}, loom::encoding::layout::strided(1, kPitch));
  const bf16x16 bw = fr::load<fr::role::rhs, bf16x16>(wv, {16 * T, 0}, sh);
#ifndef BARE_NF
#define BARE_NF 8
#endif
#define MMAQ(I) if constexpr (BARE_NF > I) { MMA(I) }
  MMAQ(0) MMAQ(1) MMAQ(2) MMAQ(3) MMAQ(4) MMAQ(5) MMAQ(6) MMAQ(7)
}
#endif

[[loom::kernel(cfg)]]
void bare_loop(const __bf16* x, float* out, int iters) {
  LOOM_WORKGROUP __bf16 lds[kArena];
  const unsigned tid = loom::kernel::workitem::id.x;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), lane = tid & 31;
  const u32x4* src = (const u32x4*)x;
  u32x4* dst = (u32x4*)lds;
  [[loom::unroll]] for (unsigned i = 0; i < 4 * kBuf / 8 / 256; i++) dst[i * 256 + tid] = src[(i * 256 + tid) & 511];
  loom::kernel::barrier<loom::memory_space::workgroup, loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  const __bf16* const xs = lds;
  #ifdef BARE_COLS2
  const __bf16* const wp0 = lds + 2 * kBuf + (wave & 3) * 32 * kPitch;
#else
  const __bf16* const wp0 = lds + 2 * kBuf + wave * 16 * kPitch;
#endif
  const f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
  Acc d = {z, z, z, z, z, z, z, z};
#ifdef BARE_PIPE
  [[loom::pipeline(BARE_PIPE)]]
#endif
  for (int s = 0; s < iters; s++) {
    const unsigned b = ((unsigned)s & 1) * kBuf;
    slice<0>(d, xs + b, wp0 + b);
    slice<1>(d, xs + b, wp0 + b);
#ifdef BARE_STORE
    u32x4* const sd = (u32x4*)(lds + (kBuf - b)) + tid * 2;
    sd[0] = src[tid & 511];
    sd[1] = src[(tid + 7) & 511];
    sd[2 * 256 + 0] = src[(tid + 3) & 511];
    sd[2 * 256 + 1] = src[(tid + 5) & 511];
#endif
  }
  const f32x8 t = d.a0 + d.a1 + d.a2 + d.a3 + d.a4 + d.a5 + d.a6 + d.a7;
  float v = 0;
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) v += t[e];
  out[loom::kernel::workgroup::id.x * 256 + tid] = v;
}
