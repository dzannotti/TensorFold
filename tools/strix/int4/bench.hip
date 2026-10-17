// microbench: W4A4 iu4 WMMA tile vs the repo's bf16 dequant prompt tile, on 27B shapes
#include "quant/mlx_tiles.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
using namespace tf::rocm;

template <int BM> __global__ void __launch_bounds__(256) dq(Affine a) { prompt_dq_tile<ActOut<WmmaBF16>, MlxDq4, BM, 32768 / BM>(a); }

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ inline int perm16(int r) { return (r & ~15) | ((r & 1) << 3) | ((r & 15) >> 1); }

// x bf16 [M][K] -> codes int4 (signed, low nibble first) [M][K/8], scale & exact sum [G][M] (rows permuted)
__global__ void quant_x(const uint16_t* x, uint32_t* q, float* sc, float* sm, int M, int K) {
    int G = K >> 6;
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= (long)M * G) return;
    int r = t / G, g = t % G;
    const u32x4* src = reinterpret_cast<const u32x4*>(x + (long)r * K + 64 * g);
    float v[64];
    for (int i = 0; i < 8; ++i) {
        u32x4 u = src[i];
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
        for (int j = 0; j < 4; ++j) { v[8 * i + 2 * j] = __uint_as_float(w[j] << 16); v[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
    }
    float amax = 0.f, s = 0.f;
    for (int i = 0; i < 64; ++i) { amax = fmaxf(amax, fabsf(v[i])); s += v[i]; }
    float d = amax / 7.f, inv = amax > 0.f ? 7.f / amax : 0.f;
    uint32_t* dst = q + (long)r * (K >> 3) + 8 * g;
    for (int i = 0; i < 8; ++i) {
        uint32_t w = 0;
        for (int j = 0; j < 8; ++j) { int c = __float2int_rn(v[8 * i + j] * inv); w |= (uint32_t)(c & 15) << (4 * j); }
        dst[i] = w;
    }
    int pr = perm16(r);
    sc[(long)g * M + pr] = d;
    sm[(long)g * M + pr] = s;
}

struct W4 { const uint32_t* xq; const float* xs; const float* xsum; const uint32_t* wq; const uint16_t* ws; const uint16_t* wb; uint16_t* out; int M, N, K; };

__device__ inline float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

// 128 x 128 a block, 8 waves of 32 x 64 (2 x 4 tiles); operands straight from global, one group ahead
__global__ void __launch_bounds__(256) w4a4(W4 a) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int nt_ = (a.N + 127) / 128;
    const int m0 = (blockIdx.x / nt_) * 128, n0 = (blockIdx.x % nt_) * 128;
    const int wm = wave & 3, wn = wave >> 2, rl = lane & 15, rh = lane >> 4;
    const int G = a.K >> 6, KW = a.K >> 3;
    const u32x4* xa[2]; const u32x4* wa[4];
    for (int mt = 0; mt < 2; ++mt) { int r = min(m0 + wm * 32 + mt * 16 + rl, a.M - 1); xa[mt] = reinterpret_cast<const u32x4*>(a.xq + (long)r * KW); }
    int cols[4];
    for (int nt = 0; nt < 4; ++nt) { cols[nt] = min(n0 + wn * 64 + nt * 16 + rl, a.N - 1); wa[nt] = reinterpret_cast<const u32x4*>(a.wq + (long)cols[nt] * KW); }
    const int rb = m0 + wm * 32 + rh * 8;  // permuted: lane's 8 rows of tile mt at rb + 16 mt
    f32x8 acc[2][4] = {};
    u32x4 X[2][2], W[4][2];
    auto load = [&](int g) {
        for (int mt = 0; mt < 2; ++mt) { X[mt][0] = xa[mt][2 * g]; X[mt][1] = xa[mt][2 * g + 1]; }
        for (int nt = 0; nt < 4; ++nt) { W[nt][0] = wa[nt][2 * g]; W[nt][1] = wa[nt][2 * g + 1]; }
    };
    load(0);
    for (int g = 0; g < G; ++g) {
        u32x4 Xc[2][2], Wc[4][2];
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) Xc[i][j] = X[i][j];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) Wc[i][j] = W[i][j];
        float sw[4], bw[4];
        for (int nt = 0; nt < 4; ++nt) { sw[nt] = bf(a.ws[(long)cols[nt] * G + g]); bw[nt] = bf(a.wb[(long)cols[nt] * G + g]); }
        float sx[2][8], su[2][8];
        for (int mt = 0; mt < 2; ++mt) {
            const float4* p = reinterpret_cast<const float4*>(a.xs + (long)g * a.M + rb + 16 * mt);
            const float4* q = reinterpret_cast<const float4*>(a.xsum + (long)g * a.M + rb + 16 * mt);
            float4 p0 = p[0], p1 = p[1], q0 = q[0], q1 = q[1];
            float t0[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w}, t1[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (int i = 0; i < 8; ++i) { sx[mt][i] = t0[i]; su[mt][i] = t1[i]; }
        }
        if (g + 1 < G) load(g + 1);
        for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 4; ++nt) {
            v8i d = {0, 0, 0, 0, 0, 0, 0, 0};
            const uint32_t xw[8] = {Xc[mt][0].x, Xc[mt][0].y, Xc[mt][0].z, Xc[mt][0].w, Xc[mt][1].x, Xc[mt][1].y, Xc[mt][1].z, Xc[mt][1].w};
            const uint32_t ww[8] = {Wc[nt][0].x, Wc[nt][0].y, Wc[nt][0].z, Wc[nt][0].w, Wc[nt][1].x, Wc[nt][1].y, Wc[nt][1].z, Wc[nt][1].w};
            for (int t = 0; t < 4; ++t) {
                v2i A = {(int)xw[2 * t], (int)xw[2 * t + 1]}, B = {(int)ww[2 * t], (int)ww[2 * t + 1]};
                d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, A, false, B, d, false);
            }
            for (int i = 0; i < 8; ++i) {
                acc[mt][nt][i] = fmaf((float)d[i], sx[mt][i] * sw[nt], acc[mt][nt][i]);
                acc[mt][nt][i] = fmaf(su[mt][i], bw[nt], acc[mt][nt][i]);
            }
        }
    }
    for (int mt = 0; mt < 2; ++mt) for (int i = 0; i < 8; ++i) {
        int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
        if (row >= a.M) continue;
        for (int nt = 0; nt < 4; ++nt) { int c = n0 + wn * 64 + nt * 16 + rl; if (c < a.N) { float v = acc[mt][nt][i]; a.out[(long)row * a.N + c] = (uint16_t)((__float_as_uint(v) + 0x8000u) >> 16); } }
    }
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); return (u + 0x8000u) >> 16; }
static float fbf(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 2048;
    int shapes[][2] = {{17408, 5120}, {5120, 17408}, {6144, 5120}, {5120, 6144}, {12288, 5120}};
    for (auto& sh : shapes) {
        int N = sh[0], K = sh[1], G = K / 64;
        std::vector<uint16_t> hx((size_t)M * K), hs((size_t)N * G), hb((size_t)N * G);
        std::vector<uint32_t> hw((size_t)N * K / 8);
        srand(1);
        for (auto& v : hx) v = tobf((rand() / (float)RAND_MAX - 0.5f) * 2.f);
        for (auto& v : hw) v = (uint32_t)rand() * 2654435761u ^ rand();
        for (size_t i = 0; i < hs.size(); ++i) { hs[i] = tobf(0.01f + 0.01f * (rand() / (float)RAND_MAX)); hb[i] = tobf(-8.f * fbf(hs[i])); }
        uint16_t *dx, *ds, *db, *dout; uint32_t *dw, *dxq; float *dxs, *dsum;
        CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&ds, hs.size() * 2)); CK(hipMalloc(&db, hb.size() * 2)); CK(hipMalloc(&dw, hw.size() * 4));
        CK(hipMalloc(&dout, (size_t)M * N * 2)); CK(hipMalloc(&dxq, (size_t)M * K / 2)); CK(hipMalloc(&dxs, (size_t)G * M * 4)); CK(hipMalloc(&dsum, (size_t)G * M * 4));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(ds, hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
        Affine A{}; A.x = dx; A.words = dw; A.scale = {ds, kScaleBF16}; A.bias = {db, kScaleBF16}; A.out = nullptr; A.m = M; A.n = N; A.k = K; A.bits = 4; A.group = 64; A.out16 = (__half*)dout;
        W4 w{dxq, dxs, dsum, dw, ds, db, dout, M, N, K};
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        double fl = 2.0 * M * N * K;
        auto timeit = [&](auto f) { f(); CK(hipDeviceSynchronize()); CK(hipEventRecord(e0)); for (int i = 0; i < 10; ++i) f(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); float ms; CK(hipEventElapsedTime(&ms, e0, e1)); return ms / 10; };
        bool tall = K >= 3 * N;
        int bm = tall ? 256 : 128;
        float tdq = timeit([&] { if (tall) dq<256><<<((N + 32768 / bm - 1) / (32768 / bm)) * ((M + bm - 1) / bm), 256>>>(A); else dq<128><<<((N + 255) / 256) * ((M + 127) / 128), 256>>>(A); });
        std::vector<uint16_t> ref((size_t)M * N); CK(hipMemcpy(ref.data(), dout, ref.size() * 2, hipMemcpyDeviceToHost));
        long qt = (long)M * G;
        float tq = timeit([&] { quant_x<<<(qt + 255) / 256, 256>>>(dx, dxq, dxs, dsum, M, K); });
        float t4 = timeit([&] { w4a4<<<((N + 127) / 128) * ((M + 127) / 128), 256>>>(w); });
        std::vector<uint16_t> o4((size_t)M * N); CK(hipMemcpy(o4.data(), dout, o4.size() * 2, hipMemcpyDeviceToHost));
        // scalar reference of the W4A4 product on a few outputs, from the device's quantized x
        std::vector<uint32_t> hq((size_t)M * K / 8); std::vector<float> hxs((size_t)G * M), hsum((size_t)G * M);
        CK(hipMemcpy(hq.data(), dxq, hq.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hxs.data(), dxs, hxs.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hsum.data(), dsum, hsum.size() * 4, hipMemcpyDeviceToHost));
        double maxe = 0, maxr = 0, errq = 0, nrm = 0;
        for (int s = 0; s < 400; ++s) {
            int r = (s * 7919) % M, c = (s * 104729) % N;
            int pr = (r & ~15) | ((r & 1) << 3) | ((r & 15) >> 1);
            double y = 0;
            for (int g = 0; g < G; ++g) {
                long dot = 0;
                for (int k = 64 * g; k < 64 * g + 64; ++k) {
                    int qx = (hq[(size_t)r * K / 8 + k / 8] >> (4 * (k % 8))) & 15; if (qx >= 8) qx -= 16;
                    int qw = (hw[(size_t)c * K / 8 + k / 8] >> (4 * (k % 8))) & 15;
                    dot += (long)qx * qw;
                }
                y += (double)hxs[(size_t)g * M + pr] * fbf(hs[(size_t)c * G + g]) * dot + (double)hsum[(size_t)g * M + pr] * fbf(hb[(size_t)c * G + g]);
            }
            double got = fbf(o4[(size_t)r * N + c]), bfr = fbf(ref[(size_t)r * N + c]);
            maxe = fmax(maxe, fabs(got - y) / (fabs(y) + 1e-2)); maxr = fmax(maxr, fabs(y));
            errq += (got - bfr) * (got - bfr); nrm += bfr * bfr;
        }
        printf("M=%d N=%5d K=%5d  dq %.3f ms %.1f TF | quant_x %.3f ms | w4a4 %.3f ms %.1f TOPS | w4a4+q %.1f TF-eq x%.2f | ref relerr %.2e  w4a4-vs-bf16 rel %.3f\n",
               M, N, K, tdq, fl / tdq / 1e9, tq, t4, fl / t4 / 1e9, fl / (t4 + tq) / 1e9, tdq / (t4 + tq), maxe, sqrt(errq / nrm));
        hipFree(dx); hipFree(ds); hipFree(db); hipFree(dw); hipFree(dout); hipFree(dxq); hipFree(dxs); hipFree(dsum);
    }
}
