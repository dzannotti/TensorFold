// SPDX-License-Identifier: MIT
// bare.cxx's loop with the fragment reads of slice s + 1 issued before the WMMAs of slice s (pipe_step.h).
#include <loomcxx/fragment.h>
#include <loomcxx/kernel.h>
#include "pipe_step.h"
namespace fr = loom::vector::fragment;
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
#ifndef BARE_ARENA
#define BARE_ARENA 4
#endif
constexpr unsigned kRows = 128, kPitch = 40, kBuf = kRows * kPitch, kArena = BARE_ARENA * kBuf;
static loom::kernel::configuration cfg(unsigned groups) { return {{groups, 1, 1}, {256, 1, 1}}; }
LOOM_INLINE bf16x16 raw(const __bf16* p) { return __builtin_bit_cast(bf16x16, *(const u32x8*)p); }

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
  const __bf16* const xl = lds + (lane & 15) * kPitch;
  const __bf16* const wl = lds + 2 * kBuf + (wave * 16 + (lane & 15)) * kPitch;
  const f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
  wp::Step8x1 st = {z, z, z, z, z, z, z, z,
                    raw(xl), raw(xl + 16 * kPitch), raw(xl + 32 * kPitch), raw(xl + 48 * kPitch),
                    raw(xl + 64 * kPitch), raw(xl + 80 * kPitch), raw(xl + 96 * kPitch), raw(xl + 112 * kPitch), raw(wl)};
  for (int s = 0; s < iters; s++) {
    const unsigned b = ((unsigned)s & 1) * kBuf, nb = kBuf - b;
    st = wp::pipe_step(st, xl + b + 16, wl + b + 16);
    st = wp::pipe_step(st, xl + nb, wl + nb);
  }
  const f32x8 t = st.c0_0 + st.c1_0 + st.c2_0 + st.c3_0 + st.c4_0 + st.c5_0 + st.c6_0 + st.c7_0;
  float v = 0;
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) v += t[e];
  out[loom::kernel::workgroup::id.x * 256 + tid] = v;
}
