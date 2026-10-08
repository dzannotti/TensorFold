# rocm-misc: plan, GDN, n-gram and torch-op kernels on gfx1151

All commands run in `docker/rocm-dev/run.sh` from the worktree root (torch 2.11+rocm7.13, hipcc 7.15).

## HIP build contract (for the -Dgpu=hip build)
- Flags: `-x hip -O3 -std=c++20 -ffp-contract=off -fno-gpu-flush-denormals-to-zero --offload-arch=gfx1151
  -include<abs>/zig/kernels/cuda/hip_compat.cuh -I zig/kernels/cuda/hip -I zig/kernels/cuda --genco`
  (`tools/rocm/hipcc_kernels.sh`). Give `-include` its path joined: hipcc reorders a separate one.
- Mangled names: the CUDA names with `13__nv_bfloat16` -> `14__hip_bfloat16` (checked for fn_gdn_io, fn_gdn_prefill,
  fn_gdn_tree; experts plan, extern "C" torch ops and fn_pack are unchanged).
- New entry point: `tf_topk_f32_sort_kernel` (torch_ops/topk.cu), launched only under HIP.
- Under HIP, `experts` holds the plan kernels only; `tf_fn_fp4_serial_kernel` is absent (fn_ops.cu #if block).
- Compile under HIP: fn_pack, fn_gdn_io, fn_gdn_tree, fn_gdn_prefill, fn_gdn, experts (plan), gdn, scan_rows and the
  torch ops. Do not compile: qmm_group (cooperative_groups.h, qmm_frag.cuh mma/cp.async asm), prefill_attention
  (qmm_frag.cuh) -- Nemotron's, load lazily or skip.

## Checks and results
| command | result |
|---|---|
| `python tools/rocm/check_plan.py --out D` | 270/270 exact vs host reference, repeatable (P 1..16389, E 8..1024, tile 16/64) |
| `python tools/rocm/check_pack.py --out D` | 4/4 byte-equal to torch indexing |
| `python tools/rocm/check_gdn.py --out D` | 121/121: byte-equal to the Python extensions built on ROCm; fp64; invariance |
| `cd tools/zig; python check_flashnext_ops.py --source ../../zig/kernels/cuda/torch_ops --out D --config C --lse` | 359/359 byte-equal to ATen-ROCm |
| `python tools/rocm/check_sampler_ops.py --source zig/kernels/cuda/torch_ops --out D --config C` | 311/311 (argmax, argmax_f64, topk; small-width topk shapes skipped) |
| `cd tools/zig; PYTHONPATH=../../src python check_flashnext_nucleus.py --source ../../zig/kernels/cuda/torch_ops --out D --fixture D/f.json` | mass 36/36 byte-equal |

C = /home/dzannotti/models/qwen38fn-int4-autoround/config.json.

## ATen-ROCm orders that differ from ATen-CUDA (torch 2.11+rocm7.13)
- Reduce.cuh (USE_ROCM): lane shuffles walk offsets 1, 2, 4..; CTA split uses 128 values a thread and at least 64
  CTAs; `multiProcessorCount` is 20 on gfx1151 (WGPs), `maxThreadsPerMultiProcessor` 2048.
- topk: one row of >= 10000 goes through a stable descending sort (TensorTopK should_use_sort); multi-row shapes
  past should_use_multiblock's bounds keep column order; smaller slices use single-block kernels whose ROCm order
  we do not reproduce (refused under HIP; no Flash Next width is one).
- c10::BFloat16 rounds any NaN to 0x7FC0 (compat's `__float2bfloat16_rn` does the same).
