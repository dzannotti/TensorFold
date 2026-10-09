# Fewer decode launches on gfx1151 (perf-fuse)

Base `rocm-cand` 5074bde (set aot-final = aot-hc7). New set: aot-final + 12 `_hc_up_mix` BM 16 entries
(`/home/dzannotti/tf-fuse-scratch/aot-fuse4`, built by `flashnext_aot.py build --target hip` as usual; hip_entries adds
them). Every fusion keeps the replaced kernels' bytes and has an env switch, default on, HIP only (CUDA build unchanged).

## Kept (each measured faster in a graph, 96 instances, best of 7x10 replays, view GPU shared: relative)

| fusion | before -> after | switch (=0 off) | graph bench, 96 at 1 / 4 / 16 rows (ms) |
|---|---|---|---|
| read-out up + mix at 16 rows | `_b16mm` + `_hc_mix` -> `_hc_up_mix` BM 16 BD 16 | TF_FLASHNEXT_DEC_UPMIX (DEC_BD 16/32) | whole read-out 2.90->2.83, 2.96->2.88, 3.71->3.37 |
| projection bf16 columns (DeltaNet b\|a, indexer) | `_reduce` + `tf_strided_copy` -> `tf_fn_reduce_ld` (fn_ops) | TF_FLASHNEXT_REDUCE_LD | 0.82->0.64, 0.80->0.64, 0.81->0.66 |
| router top-k + plan | `_topk_rows` + `plan_kernel` -> `tf_fn_topk_plan` (hip/topk_plan.hip, <= 1024 pairs) | TF_FLASHNEXT_TOPK_PLAN | 0.74->0.61, 0.76->0.62, 0.82->0.72 |
| shared expert gate/up + SwiGLU | `qmmf` + `tf_fn_shared_swiglu` -> `qmmf_swiglu_kernel` (16/32 rows; gu rows interleaved 32/32 at load; >= 33 rows: `qmmf` + fn_ops' interleaved SwiGLU) | TF_FLASHNEXT_SHARED_SWIGLU | 2.82->2.63, 2.86->2.65, 3.06->2.84 |
| shared expert down into its moe_y slot | `qmmf` + `tf_strided_copy` -> `qmmf_ld` | TF_FLASHNEXT_SHARED_LD | (one launch less, same kernel) |
| finish write-back from h into the streams' copy | `copyBuffer` + `_hc_writeback` -> `_hc_writeback` (HOUT != H) | (DEC_FUSE) | |

TF_FLASHNEXT_DEC_FUSE=0 turns off UPMIX, REDUCE_LD, TOPK_PLAN and the finish change. Removing one launch saves only
~1-2 us in a graph here (not the 3 us idle plus the kernel), so a fusion must not add tail work.

Bits: `_hc_up_mix` was already byte-equal to `_b16mm` + `_hc_mix` (hc.md), now also checked at 16-row tiles.
reduce_ld adds the slices in order in fp32 and rounds as Triton does (RNE, NaN -> 0x7FFF, from `_reduce`'s code
object). topk_plan copies `_topk_rows`' gfx1151 instruction sequence (exp = v_exp_f32(fma(x, log2e, 64 if
x log2e < -126)) x 2^-64 then; the total contracted as fma(e, scale, total) as LLVM did; IEEE divides; Triton's bf16
rounding), then experts.cu's plan_kernel on the picks in LDS. SwiGLU epilogue: gate/up rounded to bf16 as the store
does, then fn_ops' expression uncontracted (`#pragma clang fp contract(off)`); per-column scales make the row
interleave free. The qmmf body moved into a device function shared with the SwiGLU kernel: existing kernels' code is
the same up to instruction order (fp8-check byte-equal).

## Tried and dropped (slower in the same bench, though byte-equal)

- `_hc_wbn` (write-back + norm, a (row, stream) a program): 96 read-outs 2.98 -> 3.90 ms at 1 row (the 10 chunks run
  serially in one program).
- down projection + `_hc_act_sk` in each tile's last program (atomic ticket, `.cv` reads): +5 us a read-out
  (occupancy: 112 -> 167 VGPRs, and a serial tail).
- normed made as the down / up+mix kernels load (no `_hc_normed`): slower, `_hc_up_mix_n` spills.
- `_b16mm` with the slices' `_reduce` in each tile's last program (`_b16mm_tk`): equal at 1-4 rows, worse at 16.

## Verification (aot-fuse4, 1-layer views fn1mtp (DeltaNet) and fn1amtp (layer 3, attention, --mtp), depth 15)

- glue-check PASS: every new kernel byte-equal to the kernels it replaces at rows 1..128 (normal, wide, edge, special
  fills; topk_plan top 10 and 5 incl. ties and exp's small-input path; SwiGLU n 2560 and 1280); the Triton parts too.
- fp8-check PASS, mm-check PASS, hc_bits 111/0, triton_parity hash 1112/1112, tiles 516/0, run all equal.
- CLI sky drafted == `--no-drafts` and base == new sha on both views, INT4AR_FAST 1 and 0, served rule and
  CONFIDENCE=0, fp8 KV: fn1mtp 95c2c0d8d8a2 (fast) / fae9f59e8e75, fn1amtp b90d8de4b50a / 442795719c0a, kv8
  315654b60ab4; all fusions off gives the same.
- gate-many --streams 8 --against-solo: 24/24 in all 8 (view x fast x rule) cases, every stream's line equal to base.
- `zig build test -Dgpu=hip -Daot-set=aot-fuse4` 21/21, 105 tests; CUDA build (`zig build`) compiles, tests pass.

## Launches (rocprofv3 graph-mode traces of drafted sky runs on the views, FAST=0; tools/rocm/launch_count.py)

| | base | new |
|---|---:|---:|
| DeltaNet layer | 29 | 23 |
| attention layer | 33 | 27 |
| main forward outside the layer (embed, finish, head, draws) | 16.4 | 14.4 |
| an MTP forward (view) | 62.1 / 56.8 | 57.1 / 51.8 |

Full model (36 + 12 layers, ~3.8 MTP forwards a prose x1 round): main 1,456 -> 1,166, MTP ~236 -> ~217, ~1,690 ->
~1,385 launches a round (-18%). Estimated saving ~1.5 us a launch removed: ~0.45 ms a prose x1 round (~1%), to be
measured in the next window (A/B with TF_FLASHNEXT_DEC_FUSE=0 TF_FLASHNEXT_SHARED_SWIGLU=0 TF_FLASHNEXT_SHARED_LD=0).

## Not done

MTP absorb on the last row only (attention is being rewritten elsewhere); GDN front/back folds (the tree kernel splits
a head's value rows over blocks); `_fp4mm` + `_reduce` + slot copy in the MTP shared expert (side stream).
