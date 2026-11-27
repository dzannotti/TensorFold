#pragma once

#include "moe.hpp"

// tf_moe_route's plan over many blocks, with its bytes: each span's counts, one block's scan and items, then each
// span's stable scatter (a pair's place among its span's earlier pairs of the same expert).

namespace {

constexpr int kRouteSpan = 512;

// Block b: hist[b][e] = the pairs of span b (kRouteSpan pairs from b x kRouteSpan) that pick expert e.
extern "C" __global__ void __launch_bounds__(256) tf_moe_route_count(const int* picks, int pairs, int experts,
                                                                     int* hist) {
    extern __shared__ int counts[];
    const int t = threadIdx.x, p0 = blockIdx.x * kRouteSpan;
    for (int e = t; e < experts; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    for (int i = t; i < kRouteSpan && p0 + i < pairs; i += blockDim.x) atomicAdd(&counts[picks[p0 + i]], 1);
    __syncthreads();
    for (int e = t; e < experts; e += blockDim.x) hist[static_cast<long long>(blockIdx.x) * experts + e] = counts[e];
}

// One block: hist[b][e] becomes span b's first member place for expert e; the items as tf_moe_route cuts them.
extern "C" __global__ void __launch_bounds__(1024) tf_moe_route_scan(int* hist, int spans, int experts, int tile,
                                                                     int shared_expert, int shared_tile, int* items,
                                                                     int capacity) {
    const auto cut = [=](int e) { return e == shared_expert ? shared_tile : tile; };
    extern __shared__ int shared[];
    int* counts = shared;
    int* starts = shared + experts;
    int* ends = shared + 2 * experts;
    const int t = threadIdx.x;
    for (int e = t; e < experts; e += blockDim.x) {
        int run = 0;
        for (int b = 0; b < spans; ++b) {
            const int c = hist[static_cast<long long>(b) * experts + e];
            hist[static_cast<long long>(b) * experts + e] = run;
            run += c;
        }
        counts[e] = run;
    }
    __syncthreads();
    route_ranges(counts, experts, cut, starts, ends, shared + 3 * experts);
    __syncthreads();
    for (int e = t; e < experts; e += blockDim.x)
        for (int b = 0; b < spans; ++b) hist[static_cast<long long>(b) * experts + e] += starts[e];
    route_items(counts, starts, ends, experts, cut, items, capacity);
}

// Block b: each pair of span b to members[hist[b][its expert] + its rank among the span's earlier same-expert pairs].
extern "C" __global__ void __launch_bounds__(256) tf_moe_route_scatter(const int* picks, int pairs, int experts,
                                                                       const int* hist, int* members) {
    __shared__ int span[kRouteSpan];
    const int t = threadIdx.x, p0 = blockIdx.x * kRouteSpan;
    const int n = pairs - p0 < kRouteSpan ? pairs - p0 : kRouteSpan;
    for (int i = t; i < n; i += blockDim.x) span[i] = picks[p0 + i];
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x) {
        const int e = span[i];
        int rank = 0;
        for (int j = 0; j < i; ++j) rank += span[j] == e;
        members[hist[static_cast<long long>(blockIdx.x) * experts + e] + rank] = p0 + i;
    }
}

}  // namespace
