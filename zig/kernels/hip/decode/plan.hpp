#pragma once

#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

// A lane round's plan in device memory: no stream state is a launch argument, so one launch or graph serves any.
struct PlanArgs {
    const int* pos;                   // per row: its position
    const int* slot;                  // per row: its slot
    const int* first;                 // per slot: its first row
    const int* count;                 // per slot: its rows
    const unsigned long long* desc;   // per slot: the address of its descriptor
    const unsigned long long* snaps;  // per layer: the round's conv and DeltaNet snapshots, a row each
    const unsigned* anc;              // per row: a tree row's ancestors and itself, bits of its slot's rows (0: a chain)
    unsigned pages;                   // words of a descriptor before its page table
    unsigned pool;                    // pages a head of a layer's pool holds
};

// Slot descriptor: [0] cached positions, [1] last kept final row, two cache addresses a layer, then the page table.
__device__ __forceinline__ const unsigned long long* plan_desc(const PlanArgs& p, int slot) {
    return reinterpret_cast<const unsigned long long*>(p.desc[slot]);
}

__device__ __forceinline__ int plan_first(int layer) { return 2 + 2 * layer; }

__device__ __forceinline__ int plan_second(int layer) { return 3 + 2 * layer; }

__device__ __forceinline__ const unsigned* plan_pages(const PlanArgs& p, const unsigned long long* desc) {
    return reinterpret_cast<const unsigned*>(desc + p.pages);
}

// The cached positions of a slot: its first row's position.
__device__ __forceinline__ int plan_len(const PlanArgs& p, int slot) { return p.pos[p.first[slot]]; }

// The cache slot row r writes: its slot's cached positions plus its index among the slot's rows (a chain's position).
__device__ __forceinline__ int plan_at(const PlanArgs& p, int r) {
    const int first = p.first[p.slot[r]];
    return p.pos[first] + r - first;
}

// The `n`th set bit of `bits`, from the lowest.
__device__ __forceinline__ int nth_bit(unsigned bits, int n) {
    for (int j = 0; j < n; ++j) bits &= bits - 1;
    return __ffs(bits) - 1;
}

// The cache slot of a row's logical key `key`: below `len` the stream's own, past it the row's nth ancestor's.
__device__ __forceinline__ int plan_key(unsigned anc, int len, int key) {
    return anc == 0 || key < len ? key : len + nth_bit(anc, key - len);
}
