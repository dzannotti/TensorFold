// probe.cu for HIP: the same runtime test kernels without the cluster and PDL ones (PTX asm; AMD has neither),
// plus a streaming read for bandwidth. Nothing here serves a model.

#include <hip/hip_runtime.h>
#include <cstdint>

struct ProbeView {
    const float* src;
    int n;
    float scale;
};

__device__ int tf_probe_table[4];

extern "C" __global__ void tf_probe_fill(float* out, float base, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base + static_cast<float>(i);
}

extern "C" __global__ void tf_probe_axpy(float* y, const float* x, float a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// One thread adds to a counter, so a chain of these is a chain of dependent launches.
extern "C" __global__ void tf_probe_step(unsigned long long* counter, unsigned long long add) {
    *counter += add;
}

// By-value struct, bool, int8, double and int64 parameters, echoed so the host can check its packing.
extern "C" __global__ void tf_probe_args(float* out, ProbeView view, bool flag, int8_t small, double wide, long long big) {
    const int i = threadIdx.x;
    if (i < view.n) out[i] = view.src[i] * view.scale;
    if (i == 0) {
        out[view.n] = flag ? 1.0f : 0.0f;
        out[view.n + 1] = static_cast<float>(small);
        out[view.n + 2] = static_cast<float>(wide);
        out[view.n + 3] = static_cast<float>(big);
    }
}

extern "C" __global__ void tf_probe_read_table(int* out) {
    out[threadIdx.x] = tf_probe_table[threadIdx.x] * 2;
}

// Reads `n` 16-byte words grid-stride and folds them, so the loads cannot be dropped; one word out a block.
extern "C" __global__ void tf_probe_read(const uint4* src, unsigned long long n, unsigned* out) {
    unsigned acc = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;
}
