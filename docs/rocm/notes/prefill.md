# Prefill matmuls on gfx1151 (branch perf-prefill, base rocm a6b3077)

Microbenchmarks taken 09:20-09:50 BST 2026-10-09 with production down. Best-of-N, A/B alternated in one process,
weights rotated past the 32 MiB MALL and, where noted, inputs as well ("cold"). The numbers are relative.

## Re-reading the window-2 profile (w2/p2/rp-pf, 8 chunks of 2048 rows)

- The 610 ms "FP8 at ~16 TF" adds up kernel durations that overlap. The shared expert's `qmmw<128,64>` runs on the
  side stream at the same time as the int4 experts, and its duration grows from 0.8 to 5.2 ms. Measured alone at 2048
  rows, every block-FP8 shape already runs at 30-38 TF (table below). The real FLOP count is 12.9 TFLOP a chunk,
  not 10 (GDN layers: qkvz 16384x2560, out 2560x6144 and shared 2560x2560 + 2560x1280; attention layers: qkv
  13312x2560). That is about 373 ms of serial FP8 work.
- Each kernel's share of the wall time, with overlaps split evenly, in ms a chunk: FP8 465, hyper-connection kernels
  300 (`_b16mm_ks_sm` 114, `_hc_wb_norm` 105, `_hc_up_mix` 81), GDN 248 (back 96, chain 84, front 68), attention
  ~205 (`_chunks8` 179), int4 experts 172.
- The int4 prompt experts are close to the DRAM floor. 2048 rows x 5 picks reach all 512 experts, so a chunk streams
  every layer's 1.26 GB of expert weights: 60 GB a chunk, 249 ms at 242 GB/s. Today's kernels take ~330 ms serial,
  so 30 TF is not reachable at 2048-row chunks.

## What changed (launch geometry only: no output bit changes)

1. `cuda_fp8.zig wideGroup`: wide-tile L2 bands of 4 row tiles instead of l2Group's ~12 MB band (16 row tiles for
   K 2560). Tile order only. The mirror is `fp8_check.py plan`.
2. `cuda_int4.zig launch`: HIP prompt passes (MT 2) run on twice the resident grid. The old grid was occupancy x 20
   WGPs = 60 blocks on 40 CUs (1.5 a CU); the new one is 120 blocks (3 a CU). Units stride the grid, so no bits
   change. The mirror is `int4_check.py launch`.

| shape (2048 rows, cold x and weights) | before | after | change |
|---|---:|---:|---:|
| FP8 gdn_qkvz 16384x2560 (wide 128x128) | 4884 us, 35.2 TF | 4527 us, 37.9 TF | -357 us x 36 layers |
| FP8 attn_qkv 13312x2560 | 3736 us, 37.4 TF | 3668 us, 38.1 TF | -68 us x 12 |
| FP8 out_proj 2560x6144 (128x64, 4 K slices) | 2156 us, 29.9 TF | 2156 us, 29.9 TF | 0 |
| FP8 shared_gu 2560x2560 | 835 us, 32.1 TF | 796 us, 33.7 TF | -39 us x 48 |
| FP8 shared_down 2560x1280 | 432 us, 31.1 TF | 416 us, 32.2 TF | -16 us x 48 |
| int4 experts gate/up + down, 512 experts | 4465 + 2387 us, 14.7 TF | 4030 + 2277 us, 15.9 TF | -545 us x 48 |
| int4, 1024 / 512 rows | 6483 / 6166 us | 5969 / 5401 us | -8% / -12% |

Estimated saving: FP8 ~16 ms + int4 ~26 ms = ~42 ms of a 1,484 ms chunk (~3%, about 1,390 -> 1,430 rows/s CLI).
The shared-expert part overlaps the int4 experts, so its share is smaller. To be confirmed in a full-model window.

With the input hot (one x reused), out_proj ran fastest in a single band (34 TF). With cold inputs that gain went
away, so the band rule does not special-case it.

## Probes that did not ship (scratch builds, gdn_qkvz / out_proj at 2048 rows)

| qmmw variant | gdn_qkvz | out_proj |
|---|---:|---:|
| shipped | 35.6-35.9 TF | 29.9-30.0 TF |
| no scale fma (wrong bits; probe only) | 36.1 | 27.8 |
| no fp8 -> bf16 conversion in staging (probe) | 30.6 | 30.1 |
| half the A-fragment LDS reads (wrong bits; probe) | **44.4** | 29.7 |
| A fragments via one ds_read_b128 + 4 v_permlanex16 (same bits) | 31.3 | 25.7 |
| A and B that way | 29.4 | 24.2 |

- The 128x128 tile is limited by LDS fragment reads. RDNA3 WMMA needs A and B copied into both lane halves, so each
  fragment costs two b128 reads a lane. Swapping halves through VALU permutes costs more than it saves. The only
  real fix is larger per-wave tiles, and the bit contract blocks that: each 64-input group must run its own 4-step
  WMMA chain from zero, then an fma into the slice sum. A 64x64 wave tile would need acc (128 VGPRs), live chains
  and B fragments together, more than 256 VGPRs.
- out_proj (128x64, K slices) did not respond to any probe. It re-reads x (25 MB) from the MALL for each of its 40
  column tiles. A 128x128 split tile needs `tot` and `acc` both in registers (128 VGPRs) on top of the fragments.
- hc down `_b16mm_ks_sm` (N 324, K 10240, SK 32): a sweep of 54 tile/warp/stage configs gave the same bits for
  every config. The best was BM 32 x BLOCK_N 128, w4 s2: 685 us vs the shipped 64x128 w4 s1 at 737 us (-7%, ~5 ms a
  chunk). It needs a `hip_tune` row, the Zig `ks_tile` and an AOT set rebuild, so it is left as a follow-up.
  `_hc_wb_norm` moves ~170-225 MB a launch at 200-250 GB/s, which is the memory floor.
- GDN chain: the 64-row blocks (96 blocks) are slower than the 128-row ones (48 blocks): 2493 vs 2316 us, same bits.
  The kernel is latency-bound per step, not occupancy-bound.

## Checks

`fp8_check.py` (full: exact codes, 7 shapes x 22 row counts with row invariance, determinism, ld and strided) PASS.
`int4_check.py` (full, with checkpoint slices) PASS, and `int4_zig.sh` PASS. `zig build test -Dgpu=hip` 21/21 steps,
104/104 tests. The engine builds with `-Dkernel-set=aot-final`. The int4 A/B compared bits on the gate/up and down
outputs at every grid size: all equal.

## Where 2.3k tok/s would have to come from

The matmul tiles have little left: FP8 ~10% with the bit contract as it is, int4 ~20% to the DRAM floor.
- Longer prompt chunks (4096 rows) halve the int4 weight streaming per token (~-125 ms per 2048 tokens). This moves
  chunk boundaries (`cuda_state.prefill_rows`, Python `PREFILL_ROWS`), and with them the stored prompt states and
  the chunk-invariance references. It is a product decision.
- Fusing `_hc_wb_norm` with its neighbours (memory-bound, 105 ms a chunk), the attention `_chunks8` (179 ms) and the
  GDN back/front kernels (164 ms) are each larger than what remains in the matmuls.
