#include "quant/mlx_tiles.hpp"
#include "tiles/matrix_int4.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
using namespace tf::rocm;
template <int BM> __global__ void __launch_bounds__(256) dq(Affine a) { prompt_dq_tile<ActOut<WmmaBF16>, MlxDq4, BM, 32768 / BM>(a); }
__global__ void __launch_bounds__(256) q4(Int4Prompt a) { int4_quant_x(a); }
__global__ void __launch_bounds__(256) t4(Int4Prompt a) { int4_prompt_tile(a); }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); return (u + 0x8000u) >> 16; }
static float fbf(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char** argv) {
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), G = K / 64, mp = (M + 127) / 128 * 128;
    std::vector<uint16_t> hx((size_t)M * K), hs((size_t)N * G), hb((size_t)N * G);
    std::vector<uint32_t> hw((size_t)N * K / 8);
    srand(1);
    for (auto& v : hx) { float g = 0; for (int i = 0; i < 4; ++i) g += rand() / (float)RAND_MAX - 0.5f; v = tobf(g); }
    for (int r = 0; r < M; ++r) hx[(size_t)r * K + 7] = tobf(30.f);  // an outlier channel
    for (auto& v : hw) v = (uint32_t)rand() * 2654435761u ^ rand();
    for (size_t i = 0; i < hs.size(); ++i) { hs[i] = tobf(0.01f + 0.01f * (rand() / (float)RAND_MAX)); hb[i] = tobf(-8.f * fbf(hs[i])); }
    void *dx, *ds, *db, *o1, *o2, *o3, *fake, *xq, *xs, *su, *dw;
    (void)hipMalloc(&dx, hx.size() * 2); (void)hipMalloc(&ds, hs.size() * 2); (void)hipMalloc(&db, hb.size() * 2); (void)hipMalloc(&dw, hw.size() * 4);
    for (void** p : {&o1, &o2, &o3}) (void)hipMalloc(p, (size_t)M * N * 2);
    (void)hipMalloc(&fake, (size_t)M * K * 2); (void)hipMalloc(&xq, (size_t)M * K / 2); (void)hipMalloc(&xs, (size_t)mp * G * 4); (void)hipMalloc(&su, (size_t)mp * G * 4);
    (void)hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice); (void)hipMemcpy(ds, hs.data(), hs.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice); (void)hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice);
    Affine A{}; A.x = dx; A.words = (uint32_t*)dw; A.scale = {ds, kScaleBF16}; A.bias = {db, kScaleBF16}; A.m = M; A.n = N; A.k = K; A.bits = 4; A.group = 64; A.out16 = (__half*)o1;
    dq<128><<<((N + 255) / 256) * ((M + 127) / 128), 256>>>(A);
    Int4Prompt I{dx, (uint32_t*)xq, (float*)xs, (float*)su, (uint32_t*)dw, (uint16_t*)ds, (uint16_t*)db, (uint16_t*)o2, M, N, K, mp, fake, argc > 4 ? atoi(argv[4]) : 0};
    q4<<<((long)M * G + 255) / 256, 256>>>(I);
    t4<<<((N + 127) / 128) * ((M + 127) / 128), 256>>>(I);
    A.x = fake; A.out16 = (__half*)o3;
    dq<128><<<((N + 255) / 256) * ((M + 127) / 128), 256>>>(A);
    (void)hipDeviceSynchronize();
    std::vector<uint16_t> r1((size_t)M * N), r2(r1.size()), r3(r1.size()), hf((size_t)M * K);
    (void)hipMemcpy(r1.data(), o1, r1.size() * 2, hipMemcpyDeviceToHost); (void)hipMemcpy(r2.data(), o2, r1.size() * 2, hipMemcpyDeviceToHost); (void)hipMemcpy(r3.data(), o3, r1.size() * 2, hipMemcpyDeviceToHost);
    (void)hipMemcpy(hf.data(), fake, hf.size() * 2, hipMemcpyDeviceToHost);
    double e21 = 0, e31 = 0, e23 = 0, n1 = 0, ex = 0, nx = 0; int nan = 0;
    for (size_t i = 0; i < r1.size(); ++i) { double a = fbf(r1[i]), b = fbf(r2[i]), c = fbf(r3[i]); if (std::isnan(b) || std::isnan(c)) ++nan; e21 += (b - a) * (b - a); e31 += (c - a) * (c - a); e23 += (b - c) * (b - c); n1 += a * a; }
    for (size_t i = 0; i < hx.size(); ++i) { double a = fbf(hx[i]), b = fbf(hf[i]); ex += (a - b) * (a - b); nx += a * a; }
    printf("M=%d N=%d K=%d nan %d | x fake-quant rel %.4f | int4 tile vs bf16 %.4f | fake+dq vs bf16 %.4f | int4 tile vs fake+dq %.5f\n", M, N, K, nan, sqrt(ex / nx), sqrt(e21 / n1), sqrt(e31 / n1), sqrt(e23 / n1));
}
