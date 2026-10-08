# gfx1151 Triton retune (rocm-tritonperf)

## What changed
- `src/tensorfold/cuda/hip_tune.py`: the one table of gfx1151 launch options. `moe.router` and `bf16.matmul` launch
  with it on ROCm; `flashnext_aot.hip_options` builds the set with it (prompt_mm kernels have no Python wrapper).
  Only warps / stages change. Every new config is bit-equal to the CUDA config's JIT on the bench inputs ("same"
  in triton-tune-bench.txt): these kernels reduce K only inside the dot's MMA chain, so warps and stages change
  parallelism, not order.
- The set holds every entry twice: plain, and `tt.pointer_range 32` (`range32` in aot.json, buffer ops). 830 hsaco.
- `aot.zig`: a launch takes the range32 variant when every pointer's allocation ends within 2 GiB of it
  (`hipMemGetAddressRange`; VMM memory never counts, since the driver reports one mapped chunk, not the
  reservation). The KV caches are VMM, so `_chunks*` and `_attn_prep*` take the plain form.

| kernel | constexprs | CUDA config (old) | gfx1151 |
|---|---|---|---|
| _router | BM 16 | w4 s4 | w2 s1 |
| _router | BM 32/64 | w4 s3 | w8 s2 |
| _router | BM 128 | w4 s3 | w4 s1 |
| _b16mm | BM 16 | w4 s3 | w2 s1 |
| _b16mm | BM 128 | w4 s3 | w4 s1 |
| _b16mm_ks | all | w4 s2 (HIP default) | w8 s1 |
| _hc_up_mix | BM 32 / 64 | w4 s2 | w2 s1 / w4 s1 |

## Root cause of the wrong BM-128 `_b16mm` / `_hc_up_mix` rows (commit 3fcaebe)
This is not a launcher or ABI fault. gfx1151 code that spills heavily around masked global loads drops the mask on
a partial row tile: some rows come out wrong or NaN, and it can fault (seen at N 8240). The plain
`_b16mm` w4 s3 (290 spills) fails at M 129/200/257. BM-16 w4 s1 (88 spills) fails at M 17. `_hc_up_mix`
w4 s2 / w4 s1 / w8 s1 fail when plain (517-2089 spills). The same configs with buffer ops are right, and so are the
spill-free plain builds. The old set also failed with SK > 1 (N 48/96/8240), which the integration workaround's
`sk == 1` guard missed. Now every `_b16mm` / `_b16mm_ks` / `_router` variant spills nothing in either form.
`_hc_up_mix` has no spill-free config in both forms (BM 64: 53 / 32 spills, BM 32: 59 / 46), but it passes the
tiles check. `triton_parity.py tiles` launches every row-tiled variant in both forms on a partial last tile against
whole tiles. Old set: 18 differ. New set: 312/312 equal. Tri.b16mm's HIP split is removed. `TF_FLASHNEXT_GLUE_FUSE`
(cuda_engine.zig) can go back to default-on for correctness, but `_hc_up_mix` is only 0.95x the unfused pair now.

## Checks (final set)
hash 908/908. run 748/748 equal (17 no variant: M 1 on BM-128-only kernels). tiles 312/312. glue / mm / qsa / fp4 /
int4 / experts-check all PASS with TENSORFOLD_CUDA_KERNELS = the set. With the plain rows deleted from aot.json,
glue-check and mm-check still pass, which shows Zig picks range32. `zig build test` (hip, with
TF_FLASHNEXT_AOT_SET; and cuda) 104/104.

## Not changed (attention)
`_chunks8` (fp8 KV) has 612-678 spills in every config. There is no single config that wins across rows: w8 is
better at 1-3 rows and w2 at 16, and it is 4-5x bf16 `_chunks` (fp8 -> bf16 is software on gfx1151). It needs a
kernel change, not a config. `_chunks` with buffer ops is 30% faster (184 -> 128 us), but it cannot take them while
the caches are VMM: Zig would need each region's reservation size.
