// SPDX-License-Identifier: MIT

// NVFP4 (gfx1151 FP4-Local bake) to exact bf16 pairs for bf16 WMMA; storage and exactness: README.md.
#ifndef K4_NVFP4_DECODE_H_
#define K4_NVFP4_DECODE_H_

#include <loomcxx/kernel.h>
#include <loomcxx/low.h>

namespace k4 {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f2 __attribute__((ext_vector_type(2)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));

// One e4m3 byte as an fp16 pair holding the exact scale in both halves: bits placed as e4m3 x 2^-8, then x 256.
LOOM_INLINE unsigned nvfp4_scale(unsigned e4m3) {
  const unsigned b = e4m3 | (e4m3 << 16);
  const unsigned bits = ((b << 7) & 0x3f803f80u) | ((b << 8) & 0x80008000u);
  const h2 s = __builtin_bit_cast(h2, bits) * __builtin_bit_cast(h2, 0x5c005c00u);
  return __builtin_bit_cast(unsigned, s);
}

// gfx11 Low helpers: exact bit placement and the bf16 pack, 4 + 1 instructions a pair (see nvfp4_pair).
struct [[loom::representation("amdgpu.gfx11.generic.core")]] Gfx11 {};

// Nibbles P and P + 4 as fp16 bits of e2m1 x 2^-14: magnitude to bits 9-11 (25-27), sign to bit 15 (31).
template <unsigned P>
LOOM_INLINE unsigned e2m1_fp16(unsigned w) {
  if constexpr (P < 3) {
    return loom::low::assembly<Gfx11, unsigned>(R"((%w: reg<amdgpu.vgpr>, %shm: reg<amdgpu.sgpr>,
    %shs: reg<amdgpu.sgpr>, %mm: reg<amdgpu.sgpr>, %ms: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %a = v_lshlrev_b32 %shm, %w
  %m = v_and_b32 %mm, %a
  %b = v_lshlrev_b32 %shs, %w
  %r = v_bfi_b32 %ms, %b, %m
  return %r
})", w, 9 - 4 * P, 12 - 4 * P, 0x0e000e00u, 0x80008000u);
  } else {
    return loom::low::assembly<Gfx11, unsigned>(R"((%w: reg<amdgpu.vgpr>, %shm: reg<amdgpu.sgpr>, %mm: reg<amdgpu.sgpr>,
    %ms: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %a = v_lshrrev_b32 %shm, %w
  %m = v_and_b32 %mm, %a
  %r = v_bfi_b32 %ms, %w, %m
  return %r
})", w, 3u, 0x0e000e00u, 0x80008000u);
  }
}

// The high halves of two floats as one pair (lo's in the low half): v_perm_b32.
LOOM_INLINE unsigned high_pair(unsigned hi, unsigned lo) {
  return loom::low::assembly<Gfx11, unsigned>(R"((%hi: reg<amdgpu.vgpr>, %lo: reg<amdgpu.vgpr>,
    %sel: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %r = v_perm_b32 %hi, %lo, %sel
  return %r
})", hi, lo, 0x07060302u);
}

// Codes 2P and 2P + 1 times their group's fp16 scale pair: the exact bf16 pair (low half = code 2P).
template <unsigned P>
LOOM_INLINE unsigned nvfp4_pair(unsigned word, unsigned scale) {
  const h2 v = __builtin_bit_cast(h2, e2m1_fp16<P>(word)) * __builtin_bit_cast(h2, scale);
  const u2 bits = __builtin_bit_cast(u2, __builtin_convertvector(v, f2));
  return high_pair(bits.y, bits.x);
}

// The grouped tile's row decoder: a work-item's half K32 stage of one weight row is two baked words and one scale.
template <unsigned K>
struct Nvfp4Rows {
  struct Weights {
    const unsigned* words;        // [rows][K / 8], the expert's rows
    const unsigned char* scales;  // [rows][K / 16]
  };
  struct Raw {
    u2 w;
    unsigned s;
  };
  LOOM_INLINE static Raw fetch(const Weights& w, unsigned row, unsigned stage, unsigned half) {
#ifdef ABL_NOLOAD
    stage = 0;
#endif
    const u2* src = (const u2*)(w.words + row * (K / 8));
    return {src[2 * stage + half], w.scales[row * (K / 16) + 2 * stage + half]};
  }
  template <class Half>
  LOOM_INLINE static Half decode(const Raw& r) {
#ifdef ABL_NODECODE
    return {{r.w.x, r.w.y, r.s, r.w.x ^ r.s}, {r.w.y, r.w.x, r.s, r.w.y ^ r.s}};
#endif
    const unsigned sc = nvfp4_scale(r.s);
    return {{nvfp4_pair<0>(r.w.x, sc), nvfp4_pair<1>(r.w.x, sc), nvfp4_pair<2>(r.w.x, sc), nvfp4_pair<3>(r.w.x, sc)},
            {nvfp4_pair<0>(r.w.y, sc), nvfp4_pair<1>(r.w.y, sc), nvfp4_pair<2>(r.w.y, sc), nvfp4_pair<3>(r.w.y, sc)}};
  }
};

}  // namespace k4

#endif  // K4_NVFP4_DECODE_H_
