#pragma once

// gfx1151 FP4-Local weight decode (core/quant/fp4local_decode.zig): fp16 bit tricks, then exact bf16 pairs.
// Needs fp16 denormals kept in v_pk_mul_f16 and v_cvt_f32_f16 (the build's default IEEE denormal mode).

#include <hip/hip_fp16.h>

#include <cstdint>

namespace tf {
namespace rocm {

// The high halves of two floats as one 32-bit pair: lo's in the low half, hi's in the high (v_perm_b32).
__device__ inline uint32_t fp4l_high_pair(float lo, float hi) {
    return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// Both halves of an fp16 pair widened (v_cvt_f32_f16), then their high halves as a bf16 pair.
__device__ inline uint32_t fp4l_bf16_of(__half2 h) {
    return fp4l_high_pair(__low2float(h), __high2float(h));
}

// e4m3 bytes in bits 0-7 and 16-23 as fp16 bits of e4m3 x 2^-8: sign to bit 15, exponent and mantissa to 7-13.
__device__ inline uint32_t fp4l_e4m3_half2(uint32_t v) {
    return ((v << 7) & 0x3f803f80u) | ((v << 8) & 0x80008000u);
}

// e2m1 codes in bits 0-3 and 16-19 as fp16 bits of e2m1 x 2^-14: sign to bit 15, exponent and mantissa to 9-11.
__device__ inline uint32_t fp4l_e2m1_half2(uint32_t v) {
    return ((v << 9) & 0x0e000e00u) | ((v << 12) & 0x80008000u);
}

// An NVFP4 stage's two e4m3 group scales (low byte first) in fp16, each times 256 (one v_pk_mul_f16), each in both halves.
struct Nvfp4Scales {
    __half2 g0;
    __half2 g1;
};

__device__ inline Nvfp4Scales nvfp4_scales(uint32_t two) {
    const uint32_t spread = __builtin_amdgcn_perm(0u, two, 0x0c010c00u);
    const __half2 s = __hmul2(__builtin_bit_cast(__half2, fp4l_e4m3_half2(spread)), __builtin_bit_cast(__half2, 0x5c005c00u));
    const uint32_t b = __builtin_bit_cast(uint32_t, s);
    return {__builtin_bit_cast(__half2, __builtin_amdgcn_perm(b, b, 0x01000100u)),
            __builtin_bit_cast(__half2, __builtin_amdgcn_perm(b, b, 0x03020302u))};
}

// Codes 2p and 2p + 1 of a baked word (nibbles p and p + 4) times their group scale: the exact bf16 pair.
__device__ inline uint32_t nvfp4_pair(uint32_t w, int p, __half2 scale) {
    const uint32_t codes = (w >> (4 * p)) & 0x000f000fu;
    return fp4l_bf16_of(__hmul2(__builtin_bit_cast(__half2, fp4l_e2m1_half2(codes)), scale));
}

// An MXFP8 block's e8m0 byte as its f32 multiplier 2^(e - 119): the e4m3 fp16 trick's 2^-8 undone (e <= 246).
__device__ inline float mxfp8_multiplier(uint32_t e) {
    return __uint_as_float((e + 8u) << 23);
}

// Bytes b and b + 1 of a word (b 0 or 2) times the block's multiplier: the exact bf16 pair.
__device__ inline uint32_t mxfp8_pair(uint32_t word, int b, float mult) {
    const uint32_t spread = __builtin_amdgcn_perm(0u, word, b ? 0x0c030c02u : 0x0c010c00u);
    const __half2 h = __builtin_bit_cast(__half2, fp4l_e4m3_half2(spread));
    return fp4l_high_pair(__low2float(h) * mult, __high2float(h) * mult);
}

}  // namespace rocm
}  // namespace tf
