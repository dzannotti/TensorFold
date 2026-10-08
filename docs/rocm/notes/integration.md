# Integration on `rocm` (2026-10-08, before the first full-model window)

All on gfx1151 inside tf-rocm-dev, production sharing the GPU (timings relative only). Kernel set:
`/home/dzannotti/tf-triton-scratch/aot-hip2` (buffer ops; the first set, aot-hip, is superseded: below). No full model was loaded; engine runs used 1-layer views
(`tools/rocm/layer_view.py`, ~2.8 GiB of weights, no MTP).

## What the HIP build changes by default (all speed-only or work-arounds; CUDA unchanged)

| knob | HIP default | why |
|---|---|---|
| fn_qmm / fn_qmm_prefill / fn_qmm_cluster | optional (not built) | the draft vocabulary's 4-bit head drafts over the full head instead; TF_FLASHNEXT_MTP_Q4 ignored |
| QsaScores tiles | 0 and 2 only | r4b4 / r4b8 need 96 KiB LDS (64 KiB here); a TF_FLASHNEXT_QSA_TILE that does not fit -> tile 0 |
| TF_FLASHNEXT_QSA_FAST | off (opt-in) | fn_qsa_scores.hip 0.33-0.94x of `_scores` below 1M keys (same bits) |
| Triton argument extents | refused at load past 2 GiB | buffer ops (Engine.tritonSpans; largest at 262144 x 8: the embedding, 1.18 GiB; a chunk's indexer scores pass 2 GiB from 8192 prompt rows) |
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
| glue-check (aot-hip2) | PASS; `_hc_up_mix` 1.10x, `_hc_wb_norm` 2.02x |
| mm-check (aot-hip2) | PASS |
| 1-layer `run`: determinism, 2300 rows one chunk vs 2048+252, prompt experts/mm/fp4 serial/qsa fast/eager on-off | identical streams digests and tokens (linear and attention views) |
| 1-layer server, Mia's flags minus drafts: contracts.py b at 4 and 8, c greedy, d | PASS (18/18; full run 36 pass, 0 fail, 24 unchecked = drafts + sampled resume) |

## The first set's broken builds (fixed in aot-hip2)

The first set (aot-hip) was built without AMD buffer ops; Triton 3.6's other path miscompiles partial 128-row tiles
(BM 128 `_b16mm` with M unspecialized, `_hc_up_mix`). In the engine that made NaN logits for prompts of 17 + 16k + r
rows (found on the 1-layer views by the first dumped NaN: hc_up after the first hyper-connection). aot-hip2 (rocm-triton
7eafb51) has buffer ops on every pointer, as the JIT does for tensors under 2 GiB; the engine's work-arounds for the
first set are gone again and the 1-layer runs give the same tokens as with them.

## Still open

- No full-model run: weights packing of all 48 layers, MTP head (NVFP4 experts), drafts, and memory at 262144 x 8
  are first exercised in the window.
- Nemotron's modules (qmm_group, prefill_attention, experts_prefill, qmm_prefill) and fn_roce do not build under HIP.
