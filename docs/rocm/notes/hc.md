# Hyper-connection mixes and bf16 matmuls on gfx1151 (perf-hc)

Base `rocm-next` (7ec5235). Every change keeps the bits of the kernels it replaces (checked bit for bit, below), so no
fp64 comparison was needed: today's error is unchanged.

## What changed

| where | change | why |
|---|---|---|
| `prompt_mm._hc_up_mix` | the 4 streams' BD weight rows are one [4 BD, BK] tile a K step (one `tl.dot`, N = 4 BD), split back per stream (`tl.split`) and added in stream order | 4 sequential 5-step dot chains -> one wide chain; spill-free (BM 64 w8: 224/193 vgpr, 0 spills; the old kernel had no spill-free config) |
| `hip_tune.tile()` + `flashnext_aot.hip_entries` | gfx1151 tiles beside warps/stages: the HIP set adds the tile entry next to each spec entry; Zig (`cuda_prompt.ks_tile`, `cuda_triton.b16Tile`) and `bf16.matmul` launch it on HIP | tiles change no bits: an output is its own WMMA chain over K |
| `_b16mm_ks` | 64 x 128 tile, w4 s1 (was 128 x 64, w8 s1) | N 324: 96 programs on 40 CUs, a sixth of the work on a 4-column tile |
| `_b16mm` decode (BM 16), K slices >= 1024 | BK 256, w4 s1 (was BK 64, w2 s1) | 512-byte runs of each row a step instead of 128 (MTP's 13952 / 16480 / 10240 x 2560, 2560 x 6144) |
| hc down weights (HIP) | stored slice-major [32, N, 320] (`bf16.slice_major`, `W.Rows.slices`), read by `_b16mm_sm` / `_b16mm_ks_sm` | row-major, a (16-row, slice) program reads 16 runs of 640 B 20 KB apart; slice-major it is one 10 KB block |
| `glue._hc_act_sk` (HIP decode) | `_reduce` (fp32, slice order) + `_hc_act` in one launch | one launch less a read-out (96 a forward) |

## Bits

`tools/rocm/hc_bits.py` (ROCm JIT): RESULT_BITS

`triton_parity.py` on the new set (aot-hc5): RESULT_PARITY

1-layer views (`fn1` linear, `fn1a` attention; prompts 17 / 129 / 2300 rows, 24 decoded tokens; default, prompt mm
off, glue fusions off), rocm-next binary + aot-hip3 vs this branch + aot-hc5, tokens and prefill digests: RESULT_E2E

## Speed (microbenchmarks; see "Measurement")

RESULT_SPEED

## Measurement

The performance lead ran the full model the whole night, so every number here is taken next to it (min over
interleaved rounds, A and B alternated; relative only). Decode kernels rotate through 160 MB of weight copies so each
launch reads its weight from DRAM (a 6.5 MB weight otherwise sits in the 32 MB MALL: rocm-next's "router 73 -> 11 us"
is that hot number).

## Not done / open

- `_hc_wb_norm` (prompt, ~1 ms a 2048-row chunk launch): it moves ~250 MB (MoE mode reads the 11 expert outputs) at
  ~230 GB/s already; only removing traffic (not writing `normed`) would help.
- Router: no bit-preserving change found (17-33 programs on one serial K chain); cold 18-19 us a launch vs an 11 us
  floor.
- A hand-written HIP WMMA matvec gives Triton's bits exactly (verified) but was no faster: the decode shapes are bound
  by the weight layout / access pattern, not by Triton's code.
