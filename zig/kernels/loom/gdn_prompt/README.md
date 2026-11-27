# Prompt Gated DeltaNet recurrence (Loom, gfx11, fp32 VALU)

The delta rule of a Gated DeltaNet layer over a prompt's rows (Qwen3.8 Flash Next's linear-attention layers,
128 x 128 heads), in Loom's C++ frontend (hrx-loom 0.3.0) for gfx1151, wave32. One kernel with a plain buffer ABI:

- `gdn_recur`: per value head h and row t, `S = g S`, `u = S k`, `d = (v - u) beta`, `S = S + d k^T`, `y = S q`,
  the state carried through every row in place.

Its outputs and final state are the same bytes as TensorFold's HIP kernels (`tf_gdn_prompt_128` from 64 rows,
`gated_delta_tile<128, 8, 16>` and `gated_delta_kernel<128>` below; hipcc `-ffp-contract=off`), at any row count.

## Files

| file | job |
|---|---|
| `recur.h` | the scan: lane geometry, lane0_sum's tree, the staged rows |
| `gdn_recur.cxx` | the kernel and its launch configuration |
| `oracle.h` | input generators and the reference scan (a wave a state row), for the checks |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: y and the state bitwise against the oracle |

## ABI

`gdn_recur(q, k, v, gate, beta, state, y, length, key_heads, value_heads)`

- `q`, `k`: fp32 `[length][key_heads][128]` (normalized, q scaled); `v`, `y`: fp32 `[length][value_heads][128]`;
  `gate`, `beta`: fp32 `[length][value_heads]`; `state`: fp32 `[value_heads][128 value rows][128]`, in place.
- Value head h reads key head `h / (value_heads / key_heads)`. `length <= 0` writes nothing.
- Launch `(2, value_heads)` workgroups of 128 work-items: half a value head (64 state rows) a workgroup.

Rows are a sequence: a prompt cut into chunks gives the same bytes when each chunk continues the state.

## Sum order (the bits)

Every product and sum is rounded (no FMA contraction), in the reference's order. For state row r of head h and row t:

- `s_i = s_i g` for every element, then `u` = lane0_sum over 32 virtual lanes: lane l holds elements
  `l + 32 i`, `i = 0 .. 3`, summed from +0 in i order (`part = part + s_i k_i`), then the xor butterfly over masks
  16, 8, 4, 2, 1.
- `d = (v_r - u) beta`; `s_i = s_i + k_i d`; `y_r` = lane0_sum of `s_i q_i` the same way.

## Geometry

Four work-items a state row hold 32 elements each (virtual lanes `p + 4 j`, j = 0 .. 7): the butterfly's levels
16, 8, 4 run in registers, levels 2 and 1 across the four lanes by DPP. Each work-item holds two rows, a wave 16 rows,
four waves half a head. A stage is 8 rows of k and q (staged in LDS so a row's four lanes read adjacent 16 bytes),
the block's v values, gates and betas; each work-item fetches its share of the next stage into registers while the
current stage runs, so the global loads never stall the scan. 216 VGPRs, 10,304 bytes of LDS, no scratch.

The register prefetch of the next window and the bank-conflict-free q/k staging follow gufo's
`GdnPrefillKernel` (MIT, `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`); the arithmetic is the
TensorFold HIP reference's.

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_rows_4096_x17 --measure=case_end_to_end
```

`bench_rows_4096` and `bench_rows_4096_x17` time 1 and 17 launches over the same inputs (with their generators);
the difference over 16 is one launch.
