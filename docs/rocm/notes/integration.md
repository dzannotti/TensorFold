# Integration on `rocm` (2026-10-08, before the first full-model window)

All on gfx1151 inside tf-rocm-dev, production sharing the GPU (timings relative only). Kernel set:
`/home/dzannotti/tf-triton-scratch/aot-hip` (415 hsaco). No full model was loaded; engine runs used 1-layer views
(`tools/rocm/layer_view.py`, ~2.8 GiB of weights, no MTP).

## What the HIP build changes by default (all speed-only or work-arounds; CUDA unchanged)

| knob | HIP default | why |
|---|---|---|
| fn_qmm / fn_qmm_prefill / fn_qmm_cluster | optional (not built) | the draft vocabulary's 4-bit head drafts over the full head instead; TF_FLASHNEXT_MTP_Q4 ignored |
| QsaScores tiles | 0 and 2 only | r4b4 / r4b8 need 96 KiB LDS (64 KiB here); a TF_FLASHNEXT_QSA_TILE that does not fit -> tile 0 |
| TF_FLASHNEXT_QSA_FAST | off (opt-in) | fn_qsa_scores.hip 0.33-0.94x of `_scores` below 1M keys (same bits) |
| TF_FLASHNEXT_GLUE_FUSE | off (opt-in) | AOT `_hc_up_mix` wrong on a partial row tile (glue-check rows 17, 129) and 0.41x |
| `_b16mm` BM 128, M % 16 != 0, one K slice | last M % 16 rows on BM 16 | those AOT builds differ from the ROCm JIT (below) |
| prompt experts (fn_experts_prompt) | 64-pair items, T 2/4, from 1024 rows | experts-check: 974 rows 1.33x / 1.14x; 256-512 rows down 0.67-0.81x |
| fp4_serial rows | as CUDA (128 at N 640, 64 at N 1280) | N 1280 at 113-128 rows: 0.95x |
| sequence budget | min(MemAvailable, device free) | MemAvailable over-reports GTT by ~6.8 GiB |

## Checks (commands in WINDOW.md section 2)

| check | result |
|---|---|
| host tests, `-Dgpu=hip` and cuda | pass (fixture test skips the b16 cases HIP splits on purpose) |
| fixture launches vs the built set (`TF_FLASHNEXT_AOT_SET`) | every launch finds a variant; an empty set fails 5 tests |
| tf-cuda-test info/smoke/graph/symbols/occupancy/vmm | PASS; occupancy is per WGP (32 x 64 threads), so occupancy x 20 fills the GPU |
| oracle triton fixtures (`_swiglu`, `_add_rmsnorm`) through the Zig hsaco loader | BITEXACT |
| int4-check (merged rocm-int4) | PASS |
| fp8-check (ROCm cases: layouts only) | PASS, one-row invariance at every row count incl. 161/200/300/700/1100 |
| fp4-check, experts-check, qsa-check | PASS |
| glue-check | `_hc_up_mix` DIFFER (known, off), `_hc_wb_norm` 100/100 EQUAL (2.06x) |
| mm-check | DIFFER: its reference is `_b16mm` BM 128 (the broken builds); `_b16mm_ks` == ROCm JIT (checked in Python) |
| 1-layer `run`: determinism, 2300 rows one chunk vs 2048+252, prompt experts/mm/fp4 serial/qsa fast/eager on-off | identical streams digests and tokens (linear and attention views) |
| 1-layer server, Mia's flags minus drafts: contracts.py b at 4 and 8, c greedy, d | PASS (18/18; full run 36 pass, 0 fail, 24 unchecked = drafts + sampled resume) |

## The broken AOT `_b16mm` builds (for the Triton set's owner)

`tools/rocm/triton_parity.py run` only replays rows (1, 3, 16). At prompt rows (triton_parity's interceptor around
`bf16.matmul`): every BM 128 variant whose `M` parameter is not div16-specialized differs from the JIT for N 10240 K 320
(hc_up), N 10240 K 2560 and N 8240 K 2560 SK 2 (N 7296 equal); the div16 builds equal it at 128, 144, 256, 384, 400,
2048, 2064 rows. They spill ~290 VGPRs (928 private bytes). The JIT itself is row and BM invariant (BM 16 rows ==
BM 128 rows). In the engine they made NaN logits for prompts of 17 + 16k + r rows (all 2560 columns of the last row
after the first hyper-connection). `_hc_up_mix` (2089 spills) is broken the same way. Rebuild these with the JIT's
options (or compare AOT vs JIT at M in {129, 161, 2049}) before re-enabling.

## Still open

- No full-model run: weights packing of all 48 layers, MTP head (NVFP4 experts), drafts, and memory at 262144 x 8
  are first exercised in the window.
- `mm-check`'s reference path and the profile's `b16Slices` still launch the broken BM 128 builds (prompt_mm on
  keeps the serving path off them; TF_FLASHNEXT_PROMPT_MM=0 would put split-K shapes back on them).
- Nemotron's modules (qmm_group, prefill_attention, experts_prefill, qmm_prefill) and fn_roce do not build under HIP.
