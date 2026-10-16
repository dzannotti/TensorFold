#pragma once

#include "common.hpp"

// Embedding rows, casts, sums, column copies and the draft projection.

namespace {

// An MLX code at index i of a packed row, including one that crosses two words (qwen_math._codes).
__device__ __forceinline__ uint32_t code_at(const uint32_t* row, int words, int bits, long long i) {
    long long bit = i * bits;
    int word = static_cast<int>(bit / 32), shift = static_cast<int>(bit % 32);
    uint64_t low = row[word];
    uint64_t high = (shift + bits > 32 && word + 1 < words) ? row[word + 1] : 0;
    uint64_t value = (low >> shift) | (high << ((32 - shift) & 31));
    return static_cast<uint32_t>(value & ((1u << bits) - 1u));
}

extern "C" __global__ void tf_embed_rows(const uint32_t* words, const void* scale, const void* bias, int scale_kind,
                                  const int* ids, int n, int bits, int group, int k, void* out, int out_kind) {
    long long i = gid();
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    long long id = ids[r];
    int words_row = k * bits / 32, groups = k / group;
    float code = static_cast<float>(code_at(words + id * words_row, words_row, bits, c));
    float s = load(scale, scale_kind, id * groups + c / group);
    float b = load(bias, scale_kind, id * groups + c / group);
    float y = code * s;
    y = y + b;
    store(out, out_kind, i, y);
}

// The float gather: rows of a float table of `table_kind` (an embedding kept unquantized), in the activation type.
extern "C" __global__ void tf_embed_dense(const void* table, int table_kind, const int* ids, int n, int k, void* out, int out_kind) {
    long long i = gid();
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    store(out, out_kind, i, load(table, table_kind, static_cast<long long>(ids[r]) * k + c));
}

extern "C" __global__ void tf_cast(const void* src, int skind, void* dst, int dkind, long long n) {
    long long i = gid();
    if (i < n) store(dst, dkind, i, load(src, skind, i));
}

extern "C" __global__ void tf_silu_mul(const void* gate, const void* up, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    float s = rounded(silu(load(gate, kind, i)), kind);
    store(out, kind, i, s * load(up, kind, i));
}

extern "C" __global__ void tf_add(const void* x, const void* y, void* out, int kind, long long n) {
    long long i = gid();
    if (i < n) store(out, kind, i, load(x, kind, i) + load(y, kind, i));
}

// rows of `cols` values from a source with row stride `stride` at column `offset`, kept in kind.
extern "C" __global__ void tf_copy_cols(const void* src, long long stride, int offset, void* dst, int kind, int rows,
                                 int cols) {
    long long i = gid();
    if (i >= static_cast<long long>(rows) * cols) return;
    long long r = i / cols;
    int c = static_cast<int>(i % cols);
    long long from = r * stride + offset + c;
    if (kind == 0) static_cast<float*>(dst)[i] = static_cast<const float*>(src)[from];
    else static_cast<uint16_t*>(dst)[i] = static_cast<const uint16_t*>(src)[from];
}

// y[r, n] = x[r, :] . w[n, :] over float weights of `w_kind` (MTP fc, an EXL3 pack's small projections), a block of 256
// a column and kDenseRows rows: each thread's strided terms, then the waves' sums and the block's in a fixed
// order, so a row's bits depend on K alone. A thread loads 8 strides' terms before adding them in order.
constexpr int kDenseRows = 4;
template <int XK, int WK>
__device__ __forceinline__ void dense_rows_body(const void* x, const void* w, void* out, int rows, int n, int k) {
    constexpr int R = kDenseRows, U = 8;
    constexpr int kind = XK, w_kind = WK;  // the kinds fixed: no branch between a thread's loads
    __shared__ float part[R][8];
    const int col = blockIdx.x, r0 = blockIdx.y * R;
    // rows past the call's read its last row: their sums are never stored
    const unsigned char* xr[R];
#pragma unroll
    for (int j = 0; j < R; ++j) xr[j] = static_cast<const unsigned char*>(x) + static_cast<long long>(min(r0 + j, rows - 1)) * k * (kind == 0 ? 4 : 2);
    const void* wr = static_cast<const unsigned char*>(w) + static_cast<long long>(col) * k * (w_kind == 0 ? 4 : 2);
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.0f;
    int i0 = threadIdx.x;
    for (; i0 + 256 * (U - 1) < k; i0 += 256 * U) {
        float wv[U], xv[R][U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            wv[u] = load(wr, w_kind, i0 + 256 * u);
#pragma unroll
            for (int j = 0; j < R; ++j) xv[j][u] = load(xr[j], kind, i0 + 256 * u);
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int j = 0; j < R; ++j) acc[j] += xv[j][u] * wv[u];
    }
    for (; i0 < k; i0 += 256) {
        const float wv = load(wr, w_kind, i0);
#pragma unroll
        for (int j = 0; j < R; ++j) acc[j] += load(xr[j], kind, i0) * wv;
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_xor(acc[j], off, 32);
        if ((threadIdx.x & 31) == 0) part[j][threadIdx.x >> 5] = acc[j];
    }
    __syncthreads();
    if (threadIdx.x < R && r0 + static_cast<int>(threadIdx.x) < rows) {
        const int j = threadIdx.x;
        float sum = part[j][0];
        for (int q = 1; q < 8; ++q) sum += part[j][q];
        store(out, kind, static_cast<long long>(r0 + j) * n + col, sum);
    }
}

// y[r, n] = x[r, :] . w[n, :] over bf16 x and weights, K a multiple of 256 (a ciru_q8 checkpoint's projections): a wave
// a column, lane l's k at 8 l + 256 s by 16-byte loads, four strides in flight, its products added in k order into
// one fp32 a row, then the wave's butterfly. A row's bits depend on K alone, never on R or the rows beside it.
__device__ __forceinline__ float dense_dot8(uint4 x, uint4 w, float acc) {
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc = fmaf(__uint_as_float(xs[i] << 16), __uint_as_float(ws[i] << 16), acc);
        acc = fmaf(__uint_as_float(xs[i] & 0xffff0000u), __uint_as_float(ws[i] & 0xffff0000u), acc);
    }
    return acc;
}

template <int R>
__device__ __forceinline__ void dense_wave_body(const uint16_t* x, const uint16_t* w, uint16_t* out, int rows, int n, int k) {
    constexpr int U = R >= 8 ? 1 : 4;  // strides in flight: many rows bring their own loads
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (col >= n) return;
    const int r0 = blockIdx.y * R;
    const uint4* wr = reinterpret_cast<const uint4*>(w + static_cast<long long>(col) * k) + lane;
    const uint4* xr[R];
#pragma unroll
    for (int j = 0; j < R; ++j) xr[j] = reinterpret_cast<const uint4*>(x + static_cast<long long>(min(r0 + j, rows - 1)) * k) + lane;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.f;
    const int steps = k >> 8;
    int s = 0;
    for (; s + U <= steps; s += U) {
        uint4 wv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) wv[u] = wr[32 * (s + u)];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int j = 0; j < R; ++j) acc[j] = dense_dot8(xr[j][32 * (s + u)], wv[u], acc[j]);
    }
    for (; s < steps; ++s) {
        const uint4 wv = wr[32 * s];
#pragma unroll
        for (int j = 0; j < R; ++j) acc[j] = dense_dot8(xr[j][32 * s], wv, acc[j]);
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_xor(acc[j], off, 32);
        if (lane == j && r0 + j < rows) out[static_cast<long long>(r0 + j) * n + col] = float_to_bf16(acc[j]);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_dense_wave1(const uint16_t* x, const uint16_t* w, uint16_t* out, int rows, int n, int k) {
    dense_wave_body<1>(x, w, out, rows, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_dense_wave4(const uint16_t* x, const uint16_t* w, uint16_t* out, int rows, int n, int k) {
    dense_wave_body<4>(x, w, out, rows, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_dense_wave16(const uint16_t* x, const uint16_t* w, uint16_t* out, int rows, int n, int k) {
    dense_wave_body<16>(x, w, out, rows, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_dense_rows(const void* x, int kind, const void* w, int w_kind, void* out, int rows, int n, int k) {
#define TF_DENSE_W(XK)                                                    \
    if (w_kind == 0) dense_rows_body<XK, 0>(x, w, out, rows, n, k);       \
    else if (w_kind == 1) dense_rows_body<XK, 1>(x, w, out, rows, n, k);  \
    else dense_rows_body<XK, 2>(x, w, out, rows, n, k);
    if (kind == 0) {
        TF_DENSE_W(0)
    } else if (kind == 1) {
        TF_DENSE_W(1)
    } else {
        TF_DENSE_W(2)
    }
#undef TF_DENSE_W
}

}  // namespace
