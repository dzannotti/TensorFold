// A lane round's attention walk, cache writes and keep, read from the round's plan on the device (plan.hpp).

#include "ops/common.hpp"
#include "decode/plan.hpp"

namespace {

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

// Row r's keys (already rotated, fp32) and values (the activation kind) into its slot's pages at its cache slot.
__device__ void plan_kv_write_body(const float* keys, const void* values, int kind, int rows, int kv_heads, int d,
                                     PlanArgs plan, int layer) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    const unsigned long long* desc = plan_desc(plan, plan.slot[r]);
    const long long to = page_at(plan_pages(plan, desc), plan.pool, h, plan_at(plan, r), d) + j;
    store(reinterpret_cast<void*>(desc[plan_first(layer)]), kind, to, keys[i]);
    static_cast<uint16_t*>(reinterpret_cast<void*>(desc[plan_second(layer)]))[to] = static_cast<const uint16_t*>(values)[i];
}

// Query heads a block takes together: they share a KV head, so its keys and values are read once for all of them.
constexpr int kMaxGroup = 16;

// Keys a warp scores in turn (tf_plan_score_*): the block's staged queries serve 4 warps times this many keys.
constexpr int kScoreKeys = 4;

// score_keys over a plan's rows: a warp scores a key for each of the block's heads (one KV head's), in its order.
template <int KIND>
__device__ void plan_score_key(const float* __restrict__ qs, float* __restrict__ scores, int span, int heads,
                                 int kv_heads, int d, float scale, const PlanArgs& plan, int layer, int batch, int group,
                                 int head0, int visible, int key) {
    const int lane = threadIdx.x & 31;
    if (key >= visible) {  // a key past the row's: -inf for every head, as its sums would store
        if (lane == 0 && key < span)
            for (int g = 0; g < group; ++g) scores[(static_cast<long long>(batch) * heads + head0 + g) * span + key] = -INFINITY;
        return;
    }
    float kreg[8] = {};
    const int slot = plan.slot[batch];
    const unsigned long long* desc = plan_desc(plan, slot);
    const void* keys = reinterpret_cast<const void*>(desc[plan_first(layer)]);
    // a tree row's keys past the stream's are its ancestors' slots, in the order its serial step sees them
    const int at = plan_key(plan.anc[batch], plan_len(plan, slot), key);
    const long long base = page_at(plan_pages(plan, desc), plan.pool, head0 / (heads / kv_heads), at, d);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = lane + i * 32;
        if (idx < d) kreg[i] = load(keys, KIND, base + idx);
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
        if (g < group) {  // the same for the whole block, so every lane reaches wave_sum
            float partial = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int idx = lane + i * 32;
                if (idx < d) partial += qs[g * d + idx] * kreg[i];
            }
            const float dot = wave_sum(partial);
            if (lane == 0) scores[(static_cast<long long>(batch) * heads + head0 + g) * span + key] = dot * scale;
        }
    }
}

// A block's 4 warps score kScoreKeys keys each (warp w keys w, w + 4, ...); a block past the row's keys stages no query.
template <int KIND>
__device__ void plan_score_body(const float* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                  int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    extern __shared__ float qs[];  // the block's query heads, group * d
    const int batch = blockIdx.z;
    const int group = heads / gridDim.y;
    const int head0 = blockIdx.y * group;
    const int visible = min(plan.pos[batch] + 1, span);
    const int warps = blockDim.x >> 5;
    const int key0 = blockIdx.x * warps * kScoreKeys + (threadIdx.x >> 5);
    if (blockIdx.x * warps * kScoreKeys < visible) {
        const float* qrows = q + (static_cast<long long>(batch) * heads + head0) * d;
        for (int i = threadIdx.x; i < group * d; i += blockDim.x) qs[i] = qrows[i];
        __syncthreads();
    }
    for (int j = 0; j < kScoreKeys; ++j)
        plan_score_key<KIND>(qs, scores, span, heads, kv_heads, d, scale, plan, layer, batch, group, head0, visible, key0 + j * warps);
}

// apply_values over a plan's rows: a 128-key tile of the block's heads, weights worked once a block, in its order.
template <int KIND>
__device__ void plan_apply_body(const float* __restrict__ scores, const float* __restrict__ stats,
                                  float* __restrict__ partials, int span, int heads, int kv_heads, int d,
                                  PlanArgs plan, int layer) {
    __shared__ float weights[kMaxGroup * 128];
    const int batch = blockIdx.z;
    const int visible = min(plan.pos[batch] + 1, span);
    const int tile = blockIdx.x;
    const int group = heads / gridDim.y;
    const int head0 = blockIdx.y * group;
    const long long first = static_cast<long long>(batch) * heads + head0;
    const int lane = threadIdx.x;
    const int key0 = tile * 128;
    if (key0 >= visible) {  // a tile past the keys in use adds zero
        if (lane < d) {
            for (int g = 0; g < group; ++g) partials[((first + g) * gridDim.x + tile) * d + lane] = 0.f;
        }
        return;
    }
    const int stop = min(key0 + 128, visible);
    for (int i = threadIdx.x; i < group * 128; i += blockDim.x) {
        const long long h = first + i / 128;
        const int key = key0 + i % 128;
        if (key < stop) weights[i] = expf(scores[h * span + key] - stats[h * 2]) / stats[h * 2 + 1];
    }
    __syncthreads();
    if (lane >= d) return;
    const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
    const void* values = reinterpret_cast<const void*>(desc[plan_second(layer)]);
    const unsigned* table = plan_pages(plan, desc);
    const int kv = head0 / (heads / kv_heads);
    float acc[kMaxGroup] = {};
    // a tree row's keys past the stream's are its ancestors' slots: summed in the same order, a key at a time
    const unsigned anc = plan.anc[batch];
    const int len = plan_len(plan, plan.slot[batch]);
    const int split = anc == 0 ? stop : min(max(len, key0), stop);
    // a page at a time: its table entry is read once, and the keys are summed in order
    int key = key0;
    while (key < split) {
        const int edge = min((key / kPageTokens + 1) * kPageTokens, split);
        const long long slot0 = page_row(table[key / kPageTokens], plan.pool, kv, 0, d) + lane;
        for (; key < edge; ++key) {
            const float value = load(values, KIND, slot0 + static_cast<long long>(key % kPageTokens) * d);
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g < group) acc[g] += weights[g * 128 + key - key0] * value;
            }
        }
    }
    for (; key < stop; ++key) {
        const float value = load(values, KIND, page_at(table, plan.pool, kv, plan_key(anc, len, key), d) + lane);
#pragma unroll
        for (int g = 0; g < kMaxGroup; ++g) {
            if (g < group) acc[g] += weights[g * 128 + key - key0] * value;
        }
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
        if (g < group) partials[((first + g) * gridDim.x + tile) * d + lane] = acc[g];
    }
}

// ---- split attention over a plan's rows (flash decoding): a block a (256-key split, KV head, row) ----
// A thread scores one key for the block's query heads (16-byte key loads, queries from LDS), each head's split max and
// exp-sum go through LDS in key order, then a thread a dimension adds the split's values in key order. The finish adds
// the splits in order: a row's bits depend on its own keys alone, never on the round's rows or span.
constexpr int kFlashKeys = 256;

template <int KIND>
__device__ __forceinline__ void flash_widen8(uint4 w, float* v) {
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (KIND == 2) {
            v[2 * i] = __uint_as_float(u[i] << 16);
            v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        } else {
            v[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(u[i] & 0xffffu)));
            v[2 * i + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(u[i] >> 16)));
        }
    }
}

// Rows a block takes together: a chain's rows of one slot read each key and value once for all of them.
constexpr int kFlashRows = 4;
// Query vectors a block holds (rows times query heads a KV head).
constexpr int kFlashQueries = 32;

// Row r leads a block when it starts a run of chain rows (no ancestors) of its slot, at most kFlashRows long; a tree row
// or a row outside its slot's range leads alone. Returns the run's length (0: another row leads it).
__device__ __forceinline__ int flash_run(const PlanArgs& plan, int r, int rows, int per) {
    const int slot = plan.slot[r];
    const int first = plan.first[slot];
    const int end = first + plan.count[slot];
    if (plan.anc[r] != 0 || r < first || r >= end) return 1;
    if ((r - first) % per != 0 && plan.anc[r - 1] == 0) return 0;
    int n = 1;
    while (n < per && r + n < end && r + n < rows && plan.anc[r + n] == 0) ++n;
    return n;
}

template <int KIND>
__device__ void plan_flash_body(const float* __restrict__ q, float* __restrict__ part, float* __restrict__ ml, int span,
                                int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer, int rows) {
    // the queries, then (once every score is in registers) the split's weights over them
    __shared__ __attribute__((aligned(16))) float lds[kFlashQueries * 256];
    float* const qs = lds;
    float (*const ps)[kFlashQueries] = reinterpret_cast<float (*)[kFlashQueries]>(lds);
    static_assert(kFlashKeys * kFlashQueries <= kFlashQueries * 256, "the weights fit where the queries were");
    __shared__ long long vrow[kFlashKeys];
    __shared__ float stat[kFlashQueries][2];
    __shared__ int visl[kFlashRows];
    const int t = threadIdx.x;
    const int split = blockIdx.x, kv = blockIdx.y, r0 = blockIdx.z;
    const int splits = gridDim.x;
    const int group = heads / kv_heads;
    // a short walk (few splits) gives every row its own block: the queries' LDS reads bind a block, so spreading them
    // over more CUs beats reading each key once for four rows (a query's sums are the same either way)
    const int per = splits * kv_heads * rows < 640 ? 1 : min(kFlashRows, kFlashQueries / group);
    const int n = flash_run(plan, r0, rows, per);
    if (n == 0) return;
    const int head0 = kv * group;
    const int nq = n * group;  // query j: row r0 + j / group, head head0 + j % group
    const int key0 = split * kFlashKeys;
    int widest = 0;
    for (int i = 0; i < n; ++i) widest = max(widest, min(plan.pos[r0 + i] + 1, span));
    if (t < kFlashRows) visl[t] = t < n ? min(plan.pos[r0 + t] + 1, span) : 0;
    if (key0 >= widest) {  // a split past every row's keys: nothing to add
        if (t < nq) ml[((static_cast<long long>(r0 + t / group) * heads + head0 + t % group) * splits + split) * 2 + 1] = 0.f;
        return;
    }
    for (int i = t; i < nq * d; i += blockDim.x) {
        const int j = i / d;
        qs[i] = q[(static_cast<long long>(r0 + j / group) * heads + head0 + j % group) * d + i % d];
    }
    const int slot = plan.slot[r0];
    const unsigned long long* desc = plan_desc(plan, slot);
    const unsigned* table = plan_pages(plan, desc);
    const int key = key0 + t;
    const bool live = key < widest;
    const long long base = live ? page_at(table, plan.pool, kv, plan_key(plan.anc[r0], plan_len(plan, slot), key), d) : 0;
    vrow[t] = base;
    __syncthreads();
    float s[kFlashQueries];
#pragma unroll
    for (int j = 0; j < kFlashQueries; ++j) s[j] = 0.f;
    if (live) {
        const uint4* kp = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(reinterpret_cast<const void*>(desc[plan_first(layer)])) + base);
        // 64 dims of the key in flight at once (d a multiple of 64), added in dimension order
        for (int c0 = 0; c0 < d / 8; c0 += 8) {
            uint4 kk[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) kk[u] = kp[c0 + u];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int c = c0 + u;
                float k8[8];
                flash_widen8<KIND>(kk[u], k8);
#pragma unroll
                for (int j = 0; j < kFlashQueries; ++j) {
                    if (j < nq) {
                        const float4 a = *reinterpret_cast<const float4*>(qs + j * d + 8 * c);
                        const float4 b = *reinterpret_cast<const float4*>(qs + j * d + 8 * c + 4);
                        s[j] += a.x * k8[0];
                        s[j] += a.y * k8[1];
                        s[j] += a.z * k8[2];
                        s[j] += a.w * k8[3];
                        s[j] += b.x * k8[4];
                        s[j] += b.y * k8[5];
                        s[j] += b.z * k8[6];
                        s[j] += b.w * k8[7];
                    }
                }
            }
        }
    }
    __syncthreads();  // every query read: their LDS takes the weights
#pragma unroll
    for (int j = 0; j < kFlashQueries; ++j)
        if (j < nq) ps[t][j] = key < visl[j / group] ? s[j] * scale : -INFINITY;
    __syncthreads();
    // a wave a query: the split's max, then exp and sum, each in key order
    const int lane = t & 31, wave = t >> 5;
    for (int j = wave; j < nq; j += blockDim.x >> 5) {
        float m = -INFINITY;
        for (int i = lane; i < kFlashKeys; i += 32) m = fmaxf(m, ps[i][j]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 32));
        float sum = 0.f;
        for (int i = lane; i < kFlashKeys; i += 32) {
            // a row with no key in this split weighs every key zero (its sum 0 drops the split in the finish)
            const float e = m == -INFINITY ? 0.f : expf(ps[i][j] - m);
            ps[i][j] = e;
            sum += e;
        }
        sum = wave_sum(sum);
        if (lane == 0) stat[j][0] = m, stat[j][1] = sum;
    }
    __syncthreads();
    if (t < nq) {
        const long long at = ((static_cast<long long>(r0 + t / group) * heads + head0 + t % group) * splits + split) * 2;
        ml[at] = stat[t][0];
        ml[at + 1] = stat[t][1];
    }
    if (t >= d) return;
    const uint16_t* values = static_cast<const uint16_t*>(reinterpret_cast<const void*>(desc[plan_second(layer)]));
    const int stop = min(kFlashKeys, widest - key0);
    float acc[kFlashQueries];
#pragma unroll
    for (int j = 0; j < kFlashQueries; ++j) acc[j] = 0.f;
    constexpr int U = 8;
    int i = 0;
    // the next U keys' values load while these U are summed (the sums' order unchanged)
    uint16_t next[U];
    if (U <= stop) {
#pragma unroll
        for (int u = 0; u < U; ++u) next[u] = values[vrow[u] + t];
    }
    for (; i + U <= stop; i += U) {
        float v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint16_t h = next[u];
            v[u] = KIND == 2 ? __uint_as_float(static_cast<uint32_t>(h) << 16) : __half2float(__ushort_as_half(h));
        }
        if (i + 2 * U <= stop) {
#pragma unroll
            for (int u = 0; u < U; ++u) next[u] = values[vrow[i + U + u] + t];
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int j4 = 0; j4 < kFlashQueries; j4 += 4) {
                if (j4 < nq) {
                    const float4 p = *reinterpret_cast<const float4*>(&ps[i + u][j4]);
                    acc[j4] += p.x * v[u];
                    acc[j4 + 1] += p.y * v[u];
                    acc[j4 + 2] += p.z * v[u];
                    acc[j4 + 3] += p.w * v[u];
                }
            }
    }
    for (; i < stop; ++i) {
        const uint16_t h = values[vrow[i] + t];
        const float v = KIND == 2 ? __uint_as_float(static_cast<uint32_t>(h) << 16) : __half2float(__ushort_as_half(h));
#pragma unroll
        for (int j4 = 0; j4 < kFlashQueries; j4 += 4) {
            if (j4 < nq) {
                const float4 p = *reinterpret_cast<const float4*>(&ps[i][j4]);
                acc[j4] += p.x * v;
                acc[j4 + 1] += p.y * v;
                acc[j4 + 2] += p.z * v;
                acc[j4 + 3] += p.w * v;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kFlashQueries; ++j)
        if (j < nq) part[((static_cast<long long>(r0 + j / group) * heads + head0 + j % group) * splits + split) * d + t] = acc[j];
}

// The words of a row copied by a block row of threads, four at a time.
__device__ __forceinline__ void copy_words(unsigned int* to, const unsigned int* from, int words) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < words; i += stride * 4) {
        *reinterpret_cast<uint4*>(to + i) = *reinterpret_cast<const uint4*>(from + i);
    }
}

// The words of a row copied by one block, four at a time.
__device__ __forceinline__ void block_words(unsigned int* to, const unsigned int* from, int words) {
    for (int i = threadIdx.x * 4; i < words; i += blockDim.x * 4) {
        *reinterpret_cast<uint4*>(to + i) = *reinterpret_cast<const uint4*>(from + i);
    }
}

}  // namespace

extern "C" {

__global__ void __launch_bounds__(256)
tf_plan_kv_write(const float* keys, const void* values, int kind, int rows, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_kv_write_body(keys, values, kind, rows, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_f16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<1>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_bf16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<2>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_f16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<1>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_bf16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<2>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_flash_f16(const float* q, float* part, float* ml, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer, int rows) {
    plan_flash_body<1>(q, part, ml, span, heads, kv_heads, d, scale, plan, layer, rows);
}

__global__ void __launch_bounds__(256)
tf_plan_flash_bf16(const float* q, float* part, float* ml, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer, int rows) {
    plan_flash_body<2>(q, part, ml, span, heads, kv_heads, d, scale, plan, layer, rows);
}

// The splits of a (head, row) added in order: out = sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s.
__global__ void __launch_bounds__(256)
tf_plan_flash_sum(const float* part, const float* ml, float* out, int splits, int d) {
    const int head = blockIdx.x, batch = blockIdx.y, t = threadIdx.x;
    if (t >= d) return;
    const long long hr = static_cast<long long>(batch) * gridDim.x + head;
    const float* m = ml + hr * splits * 2;
    float top = -INFINITY;
    for (int s = 0; s < splits; ++s)
        if (m[2 * s + 1] > 0.f) top = fmaxf(top, m[2 * s]);
    float acc = 0.f, total = 0.f;
    for (int s = 0; s < splits; ++s) {
        if (!(m[2 * s + 1] > 0.f)) continue;
        const float w = expf(m[2 * s] - top);
        total += w * m[2 * s + 1];
        acc += w * part[(hr * splits + s) * d + t];
    }
    out[hr * d + t] = acc / total;
}

// A prompt pass's rotated, rounded keys and values (16-bit) into positions pos0 .. pos0 + len of a stream's pages.
__global__ void __launch_bounds__(256)
tf_page_write(const uint16_t* src, uint16_t* pool, const unsigned* table, int len, int kv_heads, int d, long long s_head,
              long long s_row, int pos0, int count) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    pool[page_at(table, count, h, pos0 + r, d) + j] = src[h * s_head + r * s_row + j];
}

// The first `len` positions of a stream's pages as (kv_heads, len, d), the flat layout other attention kernels read.
__global__ void __launch_bounds__(256)
tf_page_gather(const uint16_t* pool, const unsigned* table, uint16_t* dst, int len, int kv_heads, int d, int count) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int r = static_cast<int>((i / d) % len);
    const int h = static_cast<int>(i / (static_cast<long long>(d) * len));
    dst[i] = pool[page_at(table, count, h, r, d) + j];
}

// Row r of `dst` is `words` words from the address srcs[r]: a launch whose rows come from anywhere keeps its arguments.
__global__ void __launch_bounds__(256) tf_plan_gather(const unsigned long long* srcs, unsigned int* dst, int words) {
    const unsigned int* from = reinterpret_cast<const unsigned int*>(srcs[blockIdx.y]);
    unsigned int* to = dst + static_cast<long long>(blockIdx.y) * words;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) to[i] = from[i];
}

// Each slot's conv window and last final row after the path to its row keep[slot] - 1 (negative: none): a chain's
// first keep[slot] rows, a tree's that row's ancestors, their keys and values moved down; states: tf_gdn_replay.
__global__ void __launch_bounds__(256)
tf_plan_keep(PlanArgs plan, const int* keep, const unsigned int* hidden, int hidden_words, int channels, int taps, int layers,
             int kv_heads, int head_dim) {
    const int slot = blockIdx.z;
    const int last = keep[slot];
    if (last < 0) return;
    const unsigned long long* desc = plan_desc(plan, slot);
    if (blockIdx.y == layers) {
        const int row = plan.first[slot] + last - 1;
        if (desc[1] != 0) copy_words(reinterpret_cast<unsigned int*>(desc[1]), hidden + static_cast<long long>(row) * hidden_words, hidden_words);
        return;
    }
    const int layer = blockIdx.y;
    if (plan.count[slot] <= 1) return;
    const unsigned path = plan.anc[plan.first[slot] + last - 1];
    const int n = path == 0 ? last : __popc(path);
    if (plan.snaps[2 * layer] == 0) {
        // an attention layer: a tree path's keys and values to the slots after the stream's, in order (sources lie above)
        if (path == 0 || path + 1 == 1u << n) return;
        uint16_t* keys = reinterpret_cast<uint16_t*>(desc[plan_first(layer)]);
        uint16_t* values = reinterpret_cast<uint16_t*>(desc[plan_second(layer)]);
        const unsigned* table = plan_pages(plan, desc);
        const int len = plan_len(plan, slot);
        for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < kv_heads * head_dim; e += gridDim.x * blockDim.x) {
            const int h = e / head_dim;
            const int j = e % head_dim;
            unsigned rest = path;
            for (int i = 0; i < n; ++i) {
                const int from = __ffs(rest) - 1;
                rest &= rest - 1;
                if (from == i) continue;
                const long long src = page_at(table, plan.pool, h, len + from, head_dim) + j;
                const long long dst = page_at(table, plan.pool, h, len + i, head_dim) + j;
                keys[dst] = keys[src];
                values[dst] = values[src];
            }
        }
        return;
    }
    float* window = reinterpret_cast<float*>(desc[plan_first(layer)]);
    const float* samples = reinterpret_cast<const float*>(plan.snaps[2 * layer]) + static_cast<long long>(plan.first[slot]) * channels;
    // tap t takes the path's position n - taps + t after the round's first window: ascending taps read ahead
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        for (int t = 0; t < taps; ++t) {
            const int at = n - taps + t;
            const int row = path == 0 ? at : nth_bit(path, at < 0 ? 0 : at);
            window[static_cast<long long>(t) * channels + c] = at < 0 ? window[static_cast<long long>(taps + at) * channels + c] : samples[static_cast<long long>(row) * channels + c];
        }
    }
}

// A tapped layer's residual rows (16-bit) into each row's slot's ring (none: skipped) at its cache slot, a block a row.
__global__ void __launch_bounds__(256)
tf_plan_tap(const uint16_t* x, PlanArgs plan, const unsigned long long* rings, int width, int tap, int taps, int ring_rows) {
    const int r = blockIdx.x;
    const unsigned long long ring = rings[plan.slot[r]];
    if (ring == 0) return;
    const long long at = (static_cast<long long>(plan_at(plan, r) % ring_rows) * taps + tap) * width;
    block_words(reinterpret_cast<unsigned int*>(reinterpret_cast<uint16_t*>(ring) + at), reinterpret_cast<const unsigned int*>(x + static_cast<long long>(r) * width), width / 2);
}

// A prompt pass's tapped residual rows from row `from` on into a stream's ring, the row at position p at p % ring_rows.
__global__ void __launch_bounds__(256)
tf_tap_rows(const uint16_t* x, uint16_t* ring, int from, long long pos0, int width, int tap, int taps, int ring_rows) {
    const long long r = from + blockIdx.x;
    const long long at = (((pos0 + r) % ring_rows) * taps + tap) * width;
    block_words(reinterpret_cast<unsigned int*>(ring + at), reinterpret_cast<const unsigned int*>(x + r * width), width / 2);
}

}  // extern "C"
