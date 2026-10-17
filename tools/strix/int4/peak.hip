#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef __bf16 b16 __attribute__((ext_vector_type(16)));
// 0 bf16; 1 iu4 pure; 2 iu4 + fold per 4; 3 iu8 pure; 4 iu4 + fold (3 ops: cvt, fma, fma; scale product precomputed)
template <int MODE> __global__ void __launch_bounds__(256) peak(float* out, int iters, float s) {
    int l = threadIdx.x;
    v2i a[4], b[2]; v4i a8[4], b8[2]; b16 ab[4], bb[2];
    for (int j = 0; j < 4; ++j) { a[j] = v2i{l + j, l * 3 - j}; a8[j] = v4i{l, j, l, l * j}; for (int i = 0; i < 16; ++i) ab[j][i] = (__bf16)(float)(l + i + j); }
    for (int j = 0; j < 2; ++j) { b[j] = v2i{l * 5 + j, l * 7}; b8[j] = v4i{l, j, 1, 2}; for (int i = 0; i < 16; ++i) bb[j][i] = (__bf16)(float)(l - i + j); }
    f8 acc[8] = {}; v8i d[8] = {};
    float sx = s * l, sw = s + l, su = s - l, bw = s * 2;
    for (int it = 0; it < iters; ++it) {
        if constexpr (MODE == 2 || MODE == 4) for (int j = 0; j < 8; ++j) d[j] = v8i{0,0,0,0,0,0,0,0};
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 8; ++j) {
            if constexpr (MODE == 0) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ab[j & 3], bb[j >> 2], acc[j]);
            else if constexpr (MODE == 3) d[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a8[j & 3], false, b8[j >> 2], d[j], false);
            else d[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a[j & 3], false, b[j >> 2], d[j], false);
        }
        if constexpr (MODE == 2) { for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) { acc[j][i] = fmaf((float)d[j][i], sx * sw, acc[j][i]); acc[j][i] = fmaf(su, bw, acc[j][i]); } }
        if constexpr (MODE == 4) { for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) { acc[j][i] = fmaf((float)d[j][i], sx, acc[j][i]); acc[j][i] = fmaf(su, bw, acc[j][i]); } }
        if constexpr (MODE == 2 || MODE == 4) asm volatile("" : "+v"(sx), "+v"(su), "+v"(sw));
        asm volatile("" : "+v"(a[0]), "+v"(b[0]), "+v"(a8[0]), "+v"(b8[0]));
    }
    float r = 0; for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) r += acc[j][i] + d[j][i];
    out[blockIdx.x * 256 + l] = r;
}
int main() {
    float* o; (void)hipMalloc(&o, 4 << 20);
    int blocks = 40 * 8, iters = 4000;
    hipEvent_t e0, e1; (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
    auto run = [&](auto k, const char* n) { k<<<blocks, 256>>>(o, 10, 1.f); (void)hipDeviceSynchronize(); (void)hipEventRecord(e0); k<<<blocks, 256>>>(o, iters, 1.f); (void)hipEventRecord(e1); (void)hipEventSynchronize(e1); float ms; (void)hipEventElapsedTime(&ms, e0, e1);
        double ops = 2.0 * 16 * 16 * 16 * 32.0 * blocks * 8 * iters; printf("%-34s %.1f T(op)s\n", n, ops / ms / 1e9); };
    run(peak<0>, "bf16 wmma"); run(peak<3>, "iu8 wmma"); run(peak<1>, "iu4 wmma"); run(peak<2>, "iu4 + fold 4 VALU/elem/group"); run(peak<4>, "iu4 + fold 3 VALU/elem/group");
}
