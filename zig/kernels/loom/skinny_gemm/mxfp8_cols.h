// SPDX-License-Identifier: MIT

// MXFP8 weights as WMMA fragments: e4m3 bytes [n][k], e8m0 block scales [n][k / 32] (README.md).

// A code enters as the exact bf16 of e4m3 x 2^-8; a block's scale is 2^(E - 119) as an f32 multiplier.
#ifndef SK_MXFP8_COLS_H_
#define SK_MXFP8_COLS_H_

#include <loomcxx/kernel.h>
#include <loomcxx/low.h>

namespace sk {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f2 __attribute__((ext_vector_type(2)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));

struct [[loom::representation("amdgpu.gfx11.generic.core")]] Gfx11 {};

// The high halves of two floats as one pair (lo's in the low half): v_perm_b32.
LOOM_INLINE unsigned high_pair(unsigned hi, unsigned lo) {
  return loom::low::assembly<Gfx11, unsigned>(R"((%hi: reg<amdgpu.vgpr>, %lo: reg<amdgpu.vgpr>,
    %sel: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %r = v_perm_b32 %hi, %lo, %sel
  return %r
})", hi, lo, 0x07060302u);
}

// Bytes 0 and 2 of w as fp16 bits of e4m3 x 2^-8: magnitude to bits 7-13 (23-29), sign to bit 15 (31).
LOOM_INLINE unsigned e4m3_even(unsigned w) {
  return loom::low::assembly<Gfx11, unsigned>(R"((%w: reg<amdgpu.vgpr>, %shm: reg<amdgpu.sgpr>,
    %shs: reg<amdgpu.sgpr>, %mm: reg<amdgpu.sgpr>, %ms: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %a = v_lshlrev_b32 %shm, %w
  %m = v_and_b32 %mm, %a
  %b = v_lshlrev_b32 %shs, %w
  %r = v_bfi_b32 %ms, %b, %m
  return %r
})", w, 7u, 8u, 0x3f803f80u, 0x80008000u);
}

// Bytes 1 and 3 of w, placed as e4m3_even places bytes 0 and 2.
LOOM_INLINE unsigned e4m3_odd(unsigned w) {
  return loom::low::assembly<Gfx11, unsigned>(R"((%w: reg<amdgpu.vgpr>, %shm: reg<amdgpu.sgpr>, %mm: reg<amdgpu.sgpr>,
    %ms: reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>) {
  %a = v_lshrrev_b32 %shm, %w
  %m = v_and_b32 %mm, %a
  %r = v_bfi_b32 %ms, %w, %m
  return %r
})", w, 1u, 0x3f803f80u, 0x80008000u);
}

// Four e4m3 codes (K order, low byte first) as two exact bf16 pairs: codes 0, 1 in .x and 2, 3 in .y.
LOOM_INLINE u2 e4m3_pairs(unsigned w) {
  const u2 ev = __builtin_bit_cast(u2, __builtin_convertvector(__builtin_bit_cast(h2, e4m3_even(w)), f2));
  const u2 od = __builtin_bit_cast(u2, __builtin_convertvector(__builtin_bit_cast(h2, e4m3_odd(w)), f2));
  return {high_pair(od.x, ev.x), high_pair(od.y, ev.y)};
}

// 16 codes (one K16 slice of a column) as its fragment: 16 exact bf16 in K order.
LOOM_INLINE bf16x16 e4m3_slice(u32x4 w) {
  const u2 a = e4m3_pairs(w.x), b = e4m3_pairs(w.y), c = e4m3_pairs(w.z), d = e4m3_pairs(w.w);
  const u32x8 v = {a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
  return __builtin_bit_cast(bf16x16, v);
}

// An e8m0 byte as the f32 multiplier 2^(E - 119) (the fp16 placement's 2^-8 undone).
LOOM_INLINE float e8m0_mult(unsigned e) { return __builtin_bit_cast(float, (e + 8u) << 23); }

}  // namespace sk

#endif  // SK_MXFP8_COLS_H_
