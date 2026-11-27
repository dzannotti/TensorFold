// Gated DeltaNet: a wave per (batch, value head, value row) walks the sequence, lane sums as the Python reference.

#include "decode/plan.hpp"

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

__device__ float lane0_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int DK>
__global__ void __launch_bounds__(32)
gated_delta_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                   const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                   float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                   float* __restrict__ states) {
    constexpr int SPAN = (DK + 31) / 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (row >= dv) return;
    const int key_head = head / (value_heads / key_heads);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row =
        ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t = 0; t < length; ++t) {
        const size_t step = (static_cast<size_t>(batch) * length + t) * value_heads + head;
        const float decay = gate[step];
        const float step_beta = beta[step];
        const float value = v[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
                              static_cast<size_t>(head) * dv + row];
        const size_t key =
            ((static_cast<size_t>(batch) * length + t) * key_heads + key_head) * DK;

        float remembered = 0.f;
        float ks[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) {
                ks[i] = k[key + idx];
                s[i] = s[i] * decay;
                remembered = remembered + s[i] * ks[i];
            }
        }
        const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) s[i] = s[i] + ks[i] * delta;
        }
        if (states != nullptr) {
            const size_t snap = (((static_cast<size_t>(batch) * length + t) * value_heads + head) * dv + row) * DK;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) states[snap + idx] = s[i];
            }
        }

        float readout = 0.f;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) readout = readout + s[i] * q[key + idx];
        }
        // Every lane has to enter the shuffle. Lane 0 owns the stored sum.
        readout = lane0_sum(readout);
        if (lane == 0) {
            y[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
              static_cast<size_t>(head) * dv + row] = readout;
        }
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

// Value rows share a staged tile of k and q, in the wave kernel's order (-ffp-contract=off keeps it).
template <int DK, int ROWS, int STEPS>
__device__ __forceinline__ void
gated_delta_tile_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                      const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                      float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                      float* __restrict__ states, int batch, bool write_back = true) {
    constexpr int SPAN = (DK + 31) / 32;
    constexpr int WIDTH = 32 * ROWS;
    const int lane = threadIdx.x;
    const int local = threadIdx.y;
    const int row = blockIdx.x * ROWS + local;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const int tid = local * 32 + lane;

    __shared__ float ks[STEPS][DK];
    __shared__ float qs[STEPS][DK];
    __shared__ float gs[STEPS];
    __shared__ float bs[STEPS];
    __shared__ float vs[STEPS][ROWS];

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row = ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t0 = 0; t0 < length; t0 += STEPS) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
        for (int i = tid; i < nvec; i += WIDTH) {
            const int elem = i * 4;
            const int step = elem / DK;
            const int col = elem - step * DK;
            const size_t at =
                ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + col;
            const float4 key = *reinterpret_cast<const float4*>(k + at);
            ks[step][col] = key.x;
            ks[step][col + 1] = key.y;
            ks[step][col + 2] = key.z;
            ks[step][col + 3] = key.w;
            // a replay (no q) advances the state alone
            if (q == nullptr) continue;
            const float4 query = *reinterpret_cast<const float4*>(q + at);
            qs[step][col] = query.x;
            qs[step][col + 1] = query.y;
            qs[step][col + 2] = query.z;
            qs[step][col + 3] = query.w;
        }
        for (int i = tid; i < n; i += WIDTH) {
            const size_t step = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[step];
            bs[i] = beta[step];
        }
        for (int i = tid; i < n * ROWS; i += WIDTH) {
            const int step = i / ROWS;
            const int which = i - step * ROWS;
            vs[step][which] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * dv +
                                blockIdx.x * ROWS + which];
        }
        __syncthreads();

        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt];
            const float step_beta = bs[tt];
            const float value = vs[tt][local];
            float remembered = 0.f;
            float held[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) {
                    held[i] = ks[tt][idx];
                    s[i] = s[i] * decay;
                    remembered = remembered + s[i] * held[i];
                }
            }
            const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) s[i] = s[i] + held[i] * delta;
            }
            if (states != nullptr) {
                const size_t snap =
                    (((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row) * DK;
#pragma unroll
                for (int i = 0; i < SPAN; ++i) {
                    const int idx = lane + i * 32;
                    if (idx < DK) states[snap + idx] = s[i];
                }
            }
            if (q == nullptr) continue;
            float readout = 0.f;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) readout = readout + s[i] * qs[tt][idx];
            }
            readout = lane0_sum(readout);
            if (lane == 0) {
                y[((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row] = readout;
            }
        }
        __syncthreads();
    }

    if (!write_back) return;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

template <int DK, int ROWS, int STEPS>
__global__ void __launch_bounds__(32 * ROWS)
gated_delta_tile(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                 const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                 float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                 float* __restrict__ states) {
    gated_delta_tile_body<DK, ROWS, STEPS>(q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv, states,
                                           blockIdx.z);
}

// A round's windows in one launch, a grid plane a slot; rows, state and snapshots come from the plan, same scan.
template <int DK>
__device__ __forceinline__ void
gated_delta_plan_body(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                      int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    const int slot = blockIdx.z;
    const int length = plan.count[slot];
    if (length == 0) return;
    const long long row0 = plan.first[slot];
    const unsigned long long* desc = plan_desc(plan, slot);
    float* state = reinterpret_cast<float*>(desc[plan_second(layer)]);
    // a window of rows leaves the state as the round found it, for its keep to replay (padding rows keep nothing)
    const bool replayed = length > 1 && slot + 1 < static_cast<int>(gridDim.z);
    gated_delta_tile_body<DK, 8, 16>(q + row0 * key_heads * DK, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                     gate + row0 * value_heads, beta + row0 * value_heads, state,
                                     y + row0 * value_heads * dv, length, key_heads, value_heads, dv, nullptr, 0,
                                     !replayed);
}

// A keep: each slot's state run from the round's start through its kept rows, by the same scan on the inputs kept.
template <int DK>
__device__ __forceinline__ void gated_delta_replay_body(const int* keep, int rows, int key_heads, int value_heads, int dv,
                                                        PlanArgs plan, int slots) {
    const int slot = blockIdx.z % slots;
    const int layer = blockIdx.z / slots;
    const int n = keep[slot];
    if (n <= 0 || plan.count[slot] <= 1 || plan.snaps[2 * layer + 1] == 0) return;
    const long long row0 = plan.first[slot];
    const float* k = reinterpret_cast<const float*>(plan.snaps[2 * layer + 1]);
    const float* v = k + static_cast<long long>(rows) * key_heads * DK;
    const float* gate = v + static_cast<long long>(rows) * value_heads * dv;
    const float* beta = gate + static_cast<long long>(rows) * value_heads;
    float* state = reinterpret_cast<float*>(plan_desc(plan, slot)[plan_second(layer)]);
    gated_delta_tile_body<DK, 8, 16>(nullptr, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                     gate + row0 * value_heads, beta + row0 * value_heads, state, nullptr, n, key_heads,
                                     value_heads, dv, nullptr, 0);
}

// A prompt's recurrence for DK = DV = 128, 4 lanes a row and 2 rows a lane; lane0_sum's sums in its order, same bits.
constexpr int kPromptSteps = 16;

__device__ __forceinline__ float quad_sum(const float (&part)[8]) {
    // xor 16, 8, 4 of lane0_sum: virtual lanes j ^ 4, j ^ 2, j ^ 1 in registers (each pair's add commutes)
    const float a0 = part[0] + part[4], a1 = part[1] + part[5], a2 = part[2] + part[6], a3 = part[3] + part[7];
    const float b0 = a0 + a2, b1 = a1 + a3;
    float c = b0 + b1;
    // xor 2 and 1: the other lanes of the row
    c = c + __shfl_xor(c, 2, 32);
    c = c + __shfl_xor(c, 1, 32);
    return c;
}

template <int WAVES>
__device__ __forceinline__ void
gated_delta_prompt_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                        const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                        float* __restrict__ y, int length, int key_heads, int value_heads) {
    constexpr int DK = 128, DV = 128, T = kPromptSteps, THREADS = 32 * WAVES, ROWS = 16 * WAVES;
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int tid = wave * 32 + lane;
    const int p = lane & 3;
    const int first = blockIdx.x * ROWS;  // the block's first Dv row
    const int row0 = first + wave * 16 + 2 * (lane >> 2);  // this lane's rows: row0, row0 + 1
    const int head = blockIdx.y, batch = blockIdx.z;
    const int key_head = head / (value_heads / key_heads);

    __shared__ __attribute__((aligned(16))) float ks[T][DK];  // transposed: [t][p * 32 + m] = k[t][p + 4 m]
    __shared__ __attribute__((aligned(16))) float qs[T][DK];
    __shared__ float vs[T][ROWS];
    __shared__ float gs[T];
    __shared__ float bs[T];

    float s[2][32];
    for (int r = 0; r < 2; ++r) {
        const float* at = state + ((static_cast<size_t>(batch) * value_heads + head) * DV + row0 + r) * DK;
#pragma unroll
        for (int m = 0; m < 32; ++m) s[r][m] = at[p + 4 * m];
    }

    for (int t0 = 0; t0 < length; t0 += T) {
        const int n = min(T, length - t0);
        for (int i = tid; i < n * DK; i += THREADS) {
            const int step = i / DK, e = i - step * DK;
            const size_t at = ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + e;
            const int to = (e & 3) * 32 + (e >> 2);
            ks[step][to] = k[at];
            qs[step][to] = q[at];
        }
        for (int i = tid; i < n * ROWS; i += THREADS) {
            const int step = i / ROWS, e = i - step * ROWS;
            vs[step][e] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * DV + first + e];
        }
        for (int i = tid; i < n; i += THREADS) {
            const size_t at = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[at];
            bs[i] = beta[at];
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt], step_beta = bs[tt];
            float kv[32], qv[32];
            const float4* k4 = reinterpret_cast<const float4*>(&ks[tt][p * 32]);
            const float4* q4 = reinterpret_cast<const float4*>(&qs[tt][p * 32]);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float4 a = k4[c], b = q4[c];
                kv[4 * c] = a.x, kv[4 * c + 1] = a.y, kv[4 * c + 2] = a.z, kv[4 * c + 3] = a.w;
                qv[4 * c] = b.x, qv[4 * c + 1] = b.y, qv[4 * c + 2] = b.z, qv[4 * c + 3] = b.w;
            }
            float out[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float part[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) part[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = j + 8 * i;
                        s[r][m] = s[r][m] * decay;
                        part[j] = part[j] + s[r][m] * kv[m];
                    }
                }
                const float delta = (vs[tt][row0 - first + r] - quad_sum(part)) * step_beta;
#pragma unroll
                for (int j = 0; j < 8; ++j) part[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = j + 8 * i;
                        s[r][m] = s[r][m] + kv[m] * delta;
                        part[j] = part[j] + s[r][m] * qv[m];
                    }
                }
                out[r] = quad_sum(part);
            }
            if (p == 0) {
                float* yo = y + ((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * DV + row0;
                yo[0] = out[0];
                yo[1] = out[1];
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < 2; ++r) {
        float* at = state + ((static_cast<size_t>(batch) * value_heads + head) * DV + row0 + r) * DK;
#pragma unroll
        for (int m = 0; m < 32; ++m) at[p + 4 * m] = s[r][m];
    }
}


// ---- the CUDA family's DeltaNet lanes (flashnext_gdn_lanes.cuh) on wave32: front, chain with a pending-row log, roll ----

// Per slot, ints [4 * slot]: pending rows, the log's first row, kept rows (-1: none), and the slot's rows' first.
struct LaneArgs {
    const int* first;
    const int* count;
    const int* ctl;
    const unsigned long long* tab;  // per slot: its table, five addresses a layer: conv, state, state_out, log, samples
};

constexpr int kLaneWords = 5;

__device__ __forceinline__ const unsigned long long* lane_layer(const LaneArgs& t, int slot, int layer) {
    return reinterpret_cast<const unsigned long long*>(t.tab[slot]) + kLaneWords * layer;
}

__device__ __forceinline__ float lane_load(const void* p, int kind, long long i) {
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return __uint_as_float(static_cast<unsigned>(static_cast<const uint16_t*>(p)[i]) << 16);
}

struct LaneFront {
    const void* x;
    int kind;
    const float* weight;
    const float* qw;
    const float* kw_w;
    float eps;
    float* qn;
    const void* ga;
    const void* gb;
    const float* a_log;
    const float* dt_bias;
    int channels, kernel, kw, vw, heads, layer, log_rows;
    LaneArgs lanes;
};

// fn_gdn_lanes_front: conv, silu, q/k norm; q to the round's rows, k, v and the gates into the slot's log, the window
// untouched (its input rows go to samples for the roll). Grid (conv blocks + gate blocks, slots), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_lanes_front(LaneFront c) {
    __shared__ float sq[128];
    __shared__ float inv_s;
    const int slot = blockIdx.y;
    const int rows = c.lanes.count[slot];
    if (rows == 0) return;
    const long long row0 = c.lanes.first[slot];
    const unsigned long long* tab = lane_layer(c.lanes, slot, c.layer);
    float* window = reinterpret_cast<float*>(tab[0]);
    float* log = reinterpret_cast<float*>(tab[3]);
    float* samples = reinterpret_cast<float*>(tab[4]);
    const int at0 = c.lanes.ctl[4 * slot + 1] + c.lanes.ctl[4 * slot];
    float* klog = log;
    float* vlog = log + static_cast<long long>(c.log_rows) * c.kw;
    float* glog = vlog + static_cast<long long>(c.log_rows) * c.vw;
    const int conv_blocks = (c.channels + 127) / 128;
    if (blockIdx.x >= conv_blocks) {
        const int i = (blockIdx.x - conv_blocks) * 128 + threadIdx.x;
        if (i >= rows * c.heads) return;
        const int r = i / c.heads, h = i % c.heads;
        const long long g = row0 * c.heads + i;
        const long long to = (static_cast<long long>((at0 + r) % c.log_rows) * c.heads + h) * 2;
        const float beta = 1.f / (1.f + expf(-lane_load(c.gb, c.kind, g)));
        const float xg = lane_load(c.ga, c.kind, g) + c.dt_bias[h];
        const float soft = xg > 20.f ? xg : log1pf(expf(xg));
        glog[to] = expf(-expf(c.a_log[h]) * soft);
        glog[to + 1] = beta;
        return;
    }
    const int channel = blockIdx.x * 128 + threadIdx.x;
    const bool live = channel < c.channels;
    const bool head = static_cast<int>(blockIdx.x) * 128 < 2 * c.kw;
    const float* taps = c.weight + static_cast<long long>(live ? channel : 0) * c.kernel;
    const int kept = c.kernel - 1;
    float history[7];
    if (live) for (int tap = 0; tap < kept; ++tap) history[tap] = window[static_cast<long long>(tap) * c.channels + channel];
    for (int r = 0; r < rows; ++r) {
        float y = 0.f, sample = 0.f;
        if (live) {
            float acc = 0.f;
            for (int tap = 0; tap < kept; ++tap) acc += history[tap] * taps[tap];
            sample = lane_load(c.x, c.kind, (row0 + r) * c.channels + channel);
            acc += sample * taps[kept];
            y = acc * (1.f / (1.f + expf(-acc)));
        }
        if (head) {
            __syncthreads();
            sq[threadIdx.x] = y * y;
            __syncthreads();
            if (threadIdx.x < 32) {
                float sum = 0.f;
                for (int j = 0; j < 4; ++j) sum += sq[threadIdx.x + 32 * j];
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
                if (threadIdx.x == 0) inv_s = rsqrtf(sum / 128.f + c.eps);
            }
            __syncthreads();
            y = y * inv_s * (channel < c.kw ? c.qw : c.kw_w)[threadIdx.x];
        }
        if (live) {
            const long long kr = (at0 + r) % c.log_rows;
            if (channel < c.kw) c.qn[(row0 + r) * c.kw + channel] = y;
            else if (channel < 2 * c.kw) klog[kr * c.kw + channel - c.kw] = y;
            else vlog[kr * c.vw + channel - 2 * c.kw] = y;
            for (int tap = 0; tap < kept - 1; ++tap) history[tap] = history[tap + 1];
            if (kept > 0) history[kept - 1] = sample;
            samples[static_cast<long long>(r) * c.channels + channel] = sample;
        }
    }
}

// fn_gdn_lanes_chain (gdn_rows_late): the slot's pending log rows on its state, then its window's rows read out; the
// end goes to the spare. Past store_above pending rows the state they reach is stored in place first.
// Grid (value heads, dv / 16, slots), 128 threads: a wave 4 state rows, a lane 4 key columns.
extern "C" __global__ void __launch_bounds__(128)
tf_gdn_lanes_chain(const float* __restrict__ q, float* __restrict__ y, LaneArgs t, int layer, int log_rows,
                   int store_above, int key_heads, int value_heads) {
    constexpr int DK = 128, DV = 128;
    const int slot = blockIdx.z;
    const int rows = t.count[slot];
    const unsigned np = t.ctl[4 * slot], first = t.ctl[4 * slot + 1];
    if (rows == 0 && np == 0) return;
    const long long row_first = t.first[slot];
    const unsigned long long* tab = lane_layer(t, slot, layer);
    float* state_in = reinterpret_cast<float*>(tab[1]);
    float* state_out = reinterpret_cast<float*>(tab[2]);
    const float* log = reinterpret_cast<const float*>(tab[3]);
    const int hv = blockIdx.x, hk = hv / (value_heads / key_heads), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * 16 + warp * 4;
    const size_t sbase = static_cast<size_t>(hv) * DV * DK;
    const float* vl = log + static_cast<size_t>(log_rows) * key_heads * DK;
    const float* gl = vl + static_cast<size_t>(log_rows) * value_heads * DV;
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v4 = *reinterpret_cast<const float4*>(state_in + sbase + (size_t)(row0 + j) * DK + lane * 4);
        s[j][0] = v4.x; s[j][1] = v4.y; s[j][2] = v4.z; s[j][3] = v4.w;
    }
    const unsigned total = np + static_cast<unsigned>(rows);
    for (unsigned p = 0; p < total; ++p) {
        if (p == np && static_cast<int>(np) > store_above) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<float4*>(state_in + sbase + (size_t)(row0 + j) * DK + lane * 4) =
                    make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        }
        const size_t lr = (first + p) % static_cast<unsigned>(log_rows);
        const float4 k4 = *reinterpret_cast<const float4*>(log + (lr * key_heads + hk) * DK + lane * 4);
        const float4 v4 = *reinterpret_cast<const float4*>(vl + lr * value_heads * DV + hv * DV + row0);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
        const float g = gl[(lr * value_heads + hv) * 2], beta = gl[(lr * value_heads + hv) * 2 + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) kv += __shfl_xor(kv, mask, 32);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        if (p < np) continue;
        const int r = static_cast<int>(p - np);
        const float4 q4 = *reinterpret_cast<const float4*>(q + ((row_first + r) * key_heads + hk) * DK + lane * 4);
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        float o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) o += __shfl_xor(o, mask, 32);
            o4[j] = o;
        }
        if (lane == 0)
            *reinterpret_cast<float4*>(y + ((row_first + r) * value_heads + hv) * DV + row0) =
                make_float4(o4[0], o4[1], o4[2], o4[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(state_out + sbase + (size_t)(row0 + j) * DK + lane * 4) =
            make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

// fn_late_roll for the conv windows: each kept slot's window moved on by its kept rows, every linear layer in one launch.
// Grid (channels / 256, layers, slots), 256 threads; layers without a window skip.
extern "C" __global__ void __launch_bounds__(256) tf_gdn_lanes_roll(LaneArgs t, int channels, int taps) {
    const int slot = blockIdx.z, layer = blockIdx.y;
    const int n = t.ctl[4 * slot + 2];
    if (n < 0 || t.count[slot] == 0) return;
    const unsigned long long* tab = lane_layer(t, slot, layer);
    float* window = reinterpret_cast<float*>(tab[0]);
    const float* samples = reinterpret_cast<const float*>(tab[4]);
    if (window == nullptr) return;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int tp = 0; tp < taps; ++tp) {
        const int at = n - taps + tp;
        window[static_cast<long long>(tp) * channels + c] =
            at < 0 ? window[static_cast<long long>(taps + at) * channels + c] : samples[static_cast<long long>(at) * channels + c];
    }
}

}  // namespace rocm
}  // namespace tf

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_128(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<128>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_128(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<128>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_16(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<16>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_16(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
               int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<16>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

// The template kernels launches.zig resolves.
template __global__ void tf::rocm::gated_delta_tile<128, 8, 16>(const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
    const float* __restrict__ beta, float* __restrict__ state, float* __restrict__ y, int length, int key_heads,
    int value_heads, int dv, float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_tile<16, 8, 16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<128>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);

// Half a value head a block (4 waves, 64 rows; grid 2 x value heads), the fastest of the three block sizes on gfx1151.
extern "C" __global__ void __launch_bounds__(128)
tf_gdn_prompt_128(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* state,
                  float* y, int length, int key_heads, int value_heads) {
    tf::rocm::gated_delta_prompt_body<4>(q, k, v, gate, beta, state, y, length, key_heads, value_heads);
}
