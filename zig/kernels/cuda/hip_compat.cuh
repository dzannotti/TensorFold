// HIP (gfx1151, wave32) spellings of the CUDA device and runtime API these kernels use. A CUDA build sees nothing.
// HIP builds force-include it (-include hip_compat.cuh) and resolve <cuda_runtime.h> and <cuda_bf16.h> to it
// through -I zig/kernels/cuda/hip, so the copies of the Python engine's device code stay byte-identical
// (zig/tests/cuda/copies.py). After it, kernels branch on __HIP_PLATFORM_AMD__ (hip_runtime.h defines it).
#pragma once

#if defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

// Round to nearest even; any NaN becomes 0x7FC0, as c10::BFloat16 rounds on ROCm (CUDA's intrinsic gives 0x7FFF;
// HIP's own conversion keeps the payload).
__host__ __device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float x) {
    return x != x ? __hip_bfloat16(__hip_bfloat16_raw{0x7FC0}) : __float2bfloat16(x);
}

// Warp intrinsics with CUDA's 32-lane semantics: every caller passes the full mask, so it is dropped; widths
// default to 32 (a wave on gfx1151); ballots and matches return HIP's 64-bit masks (high half 0 on wave32).
template <typename T>
__device__ __forceinline__ T tf_shfl(T v, int src, int width = 32) { return __shfl(v, src, width); }
template <typename T>
__device__ __forceinline__ T tf_shfl_up(T v, unsigned d, int width = 32) { return __shfl_up(v, d, width); }
template <typename T>
__device__ __forceinline__ T tf_shfl_down(T v, unsigned d, int width = 32) { return __shfl_down(v, d, width); }
template <typename T>
__device__ __forceinline__ T tf_shfl_xor(T v, int m, int width = 32) { return __shfl_xor(v, m, width); }
#define __shfl_sync(mask, ...) tf_shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) tf_shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) tf_shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) tf_shfl_xor(__VA_ARGS__)
#define __ballot_sync(mask, p) __ballot(p)
#define __any_sync(mask, p) __any(p)
#define __all_sync(mask, p) __all(p)
// lanes holding the same value: HIP's readfirstlane loop, the same mask whatever the lanes' order
#define __match_any_sync(mask, v) ((unsigned)__match_any(v))
#define __syncwarp(...) __syncwarp()

// ld.global.nc (read-only, no L1 allocate): a nontemporal load
__device__ __forceinline__ uint4 tf_ld_nc(const uint4* p) {
    uint4 r;
    r.x = __builtin_nontemporal_load(&p->x);
    r.y = __builtin_nontemporal_load(&p->y);
    r.z = __builtin_nontemporal_load(&p->z);
    r.w = __builtin_nontemporal_load(&p->w);
    return r;
}

// The runtime calls the C launchers (development parity checks) make.
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaMemsetAsync hipMemsetAsync

#endif
