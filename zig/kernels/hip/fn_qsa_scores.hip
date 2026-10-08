// fn_qsa_scores.cu on gfx11 (HIP rewrite, the same extern "C" entry points, arguments and tiles): attention._scores'
// bits as AMD Triton compiles it (triton 3.6, gfx1151). Its TTGIR layout is the NVIDIA one (sizePerThread [1, 8],
// threadsPerWarp [2, 16]), and its AMDGCN keeps the CUDA kernel's steps (read from the compiled kernel):
//   lane partial  p_l = fma(k7,q7, ... fma(k2,q2, fma(k0,q0, k1*q1)))         (v_mul_f32, then v_fmac_f32 in dim order)
//   lane tree     DPP row_shr/shl 8, 4, quad_perm xor 2, 1 with v_add_f32: the same butterfly 8, 4, 2, 1
//   heads         total = max(S_0, 0) + 0, then total + max(S_h, 0)             (v_max_f32, v_add_f32)
//   store         total / 0x413504F3 as an IEEE division (v_div_scale / v_div_fmas / v_div_fixup), not div.full
// with fp32 denormals kept (denorm mode 3). Each step here is that operation on the same operands, so each score is
// the same fp32 word. Tiles 1 and 3 (r4b4, r4b8) ask for 96 KiB of shared memory, past gfx1151's 64 KiB a block.
#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

namespace {

constexpr int DI = 128;
constexpr int HI = 4;
// a key row's 16 chunks of 8 dims stored at chunk ^ (row & 15): 16 rows' same chunk land in 16 different banks
__device__ __forceinline__ int kslot(int b, int c) { return b * DI + ((c ^ (b & 15)) * 8); }

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ float fma_rn(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ __forceinline__ float mul_rn(float a, float b) { return a * b; }
__device__ __forceinline__ float add_rn(float a, float b) { return a + b; }
__device__ __forceinline__ float max0(float a) { return fmaxf(a, 0.0f); }
// Triton's `/` on AMD: the correctly rounded quotient (HIP's fp32 `/` is the same IEEE division)
__device__ __forceinline__ float div_sqrt(float a) { return a / __uint_as_float(0x413504F3u); }

// a uint4 of 8 bf16 as 8 fp32 (exact widenings, element 0 the low half of word 0)
struct F8 {
  float v[8];
};

__device__ __forceinline__ F8 widen(const uint4& w) {
  F8 f;
  f.v[0] = __uint_as_float(w.x << 16);
  f.v[1] = __uint_as_float(w.x & 0xFFFF0000u);
  f.v[2] = __uint_as_float(w.y << 16);
  f.v[3] = __uint_as_float(w.y & 0xFFFF0000u);
  f.v[4] = __uint_as_float(w.z << 16);
  f.v[5] = __uint_as_float(w.z & 0xFFFF0000u);
  f.v[6] = __uint_as_float(w.w << 16);
  f.v[7] = __uint_as_float(w.w & 0xFFFF0000u);
  return f;
}

// lane partial over 8 dims: (k1 * q1) first, then k0 q0, k2 q2, ... k7 q7 fused
__device__ __forceinline__ float lane(const F8& k, const F8& q) {
  float p = fma_rn(k.v[0], q.v[0], mul_rn(k.v[1], q.v[1]));
#pragma unroll
  for (int i = 2; i < 8; ++i) p = fma_rn(k.v[i], q.v[i], p);
  return p;
}

}  // namespace

// IQ [rows, HI, DI] bf16, POOLED [blocks, DI] bf16, POS0 the first row's position, SC [rows, NB] fp32. A CTA of 256
// threads takes 16 TR rows x 16 TB blocks (a thread TR rows x TB blocks); keys then queries in dynamic shared memory.
template <int TR, int TB, bool KF32>
__device__ __forceinline__ void scores_body(const uint16_t* __restrict__ IQ, const uint16_t* __restrict__ POOLED,
                                            const int* __restrict__ POS0, float* __restrict__ SC, int NB, int rows,
                                            int ratio, int top) {
  constexpr int RT = 16 * TR;
  constexpr int BT = 16 * TB;
  extern __shared__ __align__(16) uint16_t smem[];
  uint16_t* ks = smem;
  float* kf32 = reinterpret_cast<float*>(smem);
  uint16_t* qs = smem + BT * DI * (KF32 ? 2 : 1);
  const int tid = threadIdx.x;
  const int r0 = blockIdx.x * RT;
  const int j0 = blockIdx.y * BT;
  const int pos0 = __ldg(POS0);
  const int last = min(r0 + RT, rows) - 1;
  const int reach = (pos0 + last + 1) / ratio;  // the tile's longest row's complete blocks
  if (reach <= top || j0 >= reach) return;
  // stage the keys of blocks [j0, j0 + BT) below `reach` (zeros past it) and the rows' queries
  for (int i = tid; i < BT * (DI / 8); i += 256) {
    const int b = i / (DI / 8), c = i % (DI / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (j0 + b < reach) v = __ldg(reinterpret_cast<const uint4*>(POOLED + (size_t)(j0 + b) * DI) + c);
    if (KF32) {
      const F8 f = widen(v);
      float4* dst = reinterpret_cast<float4*>(kf32 + kslot(b, c));
      dst[0] = make_float4(f.v[0], f.v[1], f.v[2], f.v[3]);
      dst[1] = make_float4(f.v[4], f.v[5], f.v[6], f.v[7]);
    } else {
      *reinterpret_cast<uint4*>(ks + kslot(b, c)) = v;
    }
  }
  for (int i = tid; i < RT * HI * (DI / 8); i += 256) {
    const int rh = i / (DI / 8), c = i % (DI / 8);
    const int r = rh / HI;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r0 + r < rows) v = __ldg(reinterpret_cast<const uint4*>(IQ + (size_t)(r0 * HI + rh) * DI) + c);
    *reinterpret_cast<uint4*>(qs + rh * DI + c * 8) = v;
  }
  __syncthreads();
  const int tr = tid / 16;  // rows tr * TR ..
  const int tb = tid % 16;  // blocks tb + 16 * j
  float total[TR][TB];
  // the lanes in the tree's order: c_h = ((p_h + p_h+8) + (p_h+4 + p_h+12)) + ((p_h+2 + p_h+10) + (p_h+6 + p_h+14))
#pragma unroll 1
  for (int h = 0; h < HI; ++h) {
    float c0[TR][TB], cc[TR][TB], bb[TR][TB], aa[TR][TB];
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      // step -> lane: half (bit 3), q2 (bit 2), q4 (bit 1), q8 (bit 0)
      const int half = step >> 3, q2 = (step >> 2) & 1, q4 = (step >> 1) & 1, q8 = step & 1;
      const int l = half + 2 * q2 + 4 * q4 + 8 * q8;
      F8 qf[TR];
#pragma unroll
      for (int ir = 0; ir < TR; ++ir) qf[ir] = widen(*reinterpret_cast<const uint4*>(qs + ((tr * TR + ir) * HI + h) * DI + l * 8));
#pragma unroll
      for (int jb = 0; jb < TB; ++jb) {
        F8 kf;
        if (KF32) {
          const float4* src = reinterpret_cast<const float4*>(kf32 + kslot(tb + 16 * jb, l));
          const float4 x0 = src[0], x1 = src[1];
          kf.v[0] = x0.x; kf.v[1] = x0.y; kf.v[2] = x0.z; kf.v[3] = x0.w;
          kf.v[4] = x1.x; kf.v[5] = x1.y; kf.v[6] = x1.z; kf.v[7] = x1.w;
        } else {
          kf = widen(*reinterpret_cast<const uint4*>(ks + kslot(tb + 16 * jb, l)));
        }
#pragma unroll
        for (int ir = 0; ir < TR; ++ir) {
          const float p = lane(kf, qf[ir]);
          float& A = aa[ir][jb];
          float& B = bb[ir][jb];
          float& C = cc[ir][jb];
          if (q8 == 0) {
            A = p;
          } else {
            const float a = add_rn(A, p);  // a_l = p_l + p_{l+8}
            if (q4 == 0) {
              B = a;
            } else {
              const float bsum = add_rn(B, a);  // b_l = a_l + a_{l+4}
              if (q2 == 0) {
                C = bsum;
              } else {
                const float csum = add_rn(C, bsum);  // c_l = b_l + b_{l+2}
                if (half == 0) {
                  c0[ir][jb] = csum;
                } else {
                  const float S = max0(add_rn(c0[ir][jb], csum));  // S = c_0 + c_1
                  total[ir][jb] = h == 0 ? add_rn(S, 0.0f) : add_rn(total[ir][jb], S);
                }
              }
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int ir = 0; ir < TR; ++ir) {
    const int r = r0 + tr * TR + ir;
    if (r >= rows) continue;
    const int complete = (pos0 + r + 1) / ratio;
    if (complete <= top) continue;
#pragma unroll
    for (int jb = 0; jb < TB; ++jb) {
      const int bk = j0 + tb + 16 * jb;
      if (bk < complete) SC[(size_t)r * NB + bk] = div_sqrt(total[ir][jb]);
    }
  }
}

#define TF_QSA(NAME, TR, TB, KF)                                                                                     \
  extern "C" __global__ void __launch_bounds__(256)                                                            \
      NAME(const uint16_t* __restrict__ IQ, const uint16_t* __restrict__ POOLED, const int* __restrict__ POS0,    \
           float* __restrict__ SC, int NB, int rows, int ratio, int top) {                                     \
    scores_body<TR, TB, KF>(IQ, POOLED, POS0, SC, NB, rows, ratio, top);                                           \
  }
TF_QSA(fn_qsa_scores, 2, 4, false)
TF_QSA(fn_qsa_scores_r4b4, 4, 4, true)
TF_QSA(fn_qsa_scores_r2b8, 2, 8, false)
TF_QSA(fn_qsa_scores_r4b8, 4, 8, false)
