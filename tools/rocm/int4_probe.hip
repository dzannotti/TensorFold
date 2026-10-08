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

// probe_stream with U independent uint4 loads in flight a thread (memory-level parallelism).
template <int U>
__device__ void stream_u(const u32x4* __restrict__ p, size_t n, uint32_t* out) {
  uint32_t s = 0;
  const size_t T = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i + (U - 1) * T < n; i += U * T) {
    u32x4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load(p + i + u * T);
#pragma unroll
    for (int u = 0; u < U; ++u) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
extern "C" __global__ void probe_stream4(const u32x4* __restrict__ p, size_t n, uint32_t* out) { stream_u<4>(p, n, out); }
extern "C" __global__ void probe_stream8(const u32x4* __restrict__ p, size_t n, uint32_t* out) { stream_u<8>(p, n, out); }

// WMMA rate: CH independent accumulator chains a wave, `n` steps each (CH = 1: the dependent chain's latency).
template <int CH>
__device__ void wmma_rate(int n, float* out) {
  v16s a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (short)(threadIdx.x + i); b[i] = (short)(threadIdx.x * 3 + i); }
  v8f c[CH];
  for (int j = 0; j < CH; ++j) c[j] = v8f{};
  for (int s = 0; s < n; ++s)
#pragma unroll
    for (int j = 0; j < CH; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j]);
  float t = 0.f;
  for (int j = 0; j < CH; ++j) t += c[j][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = t;
}
extern "C" __global__ void probe_wmma1(int n, float* out) { wmma_rate<1>(n, out); }
extern "C" __global__ void probe_wmma4(int n, float* out) { wmma_rate<4>(n, out); }
extern "C" __global__ void probe_wmma8(int n, float* out) { wmma_rate<8>(n, out); }

// Our weights' access pattern: wave w streams its own region of `steps` KiB, a KiB a step (2 loads of 512 bytes),
// waves grid-strided over `regions`.
extern "C" __global__ void probe_regions(const u32x4* __restrict__ p, int regions, int steps, uint32_t* out) {
  const int lane = threadIdx.x & 31;
  uint32_t s = 0;
  for (int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; r < regions; r += gridDim.x * (blockDim.x / 32)) {
    const u32x4* q = p + (size_t)r * steps * 64;
    for (int i = 0; i < steps; i += 2) {
      u32x4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) v[u] = __builtin_nontemporal_load(q + i * 64 + u * 32 + lane);
#pragma unroll
      for (int u = 0; u < 4; ++u) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
