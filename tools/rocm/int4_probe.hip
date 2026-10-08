// Probes for the int4 port: the WMMA bf16 operand/result layout on gfx11 and a plain streaming read (achievable GB/s).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// a [16][16] bf16 row-major (m, k), b [16][16] bf16 (n, k); lane l feeds a row l%16 / b column l%16; d out [32][8].
extern "C" __global__ void probe_wmma(const short* a, const short* b, float* d) {
  const int l = threadIdx.x;
  v16s va, vb;
  for (int i = 0; i < 16; ++i) { va[i] = a[(l % 16) * 16 + i]; vb[i] = b[(l % 16) * 16 + i]; }
  v8f c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, c);
  for (int i = 0; i < 8; ++i) d[l * 8 + i] = c[i];
}

// Sum of uint4s, grid-stride; out one word a thread (keeps the loads live).
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
extern "C" __global__ void probe_stream(const u32x4* __restrict__ p, size_t n, uint32_t* out) {
  uint32_t s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const u32x4 v = __builtin_nontemporal_load(p + i);
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
