# AMD Strix Halo (gfx1151), native HIP engine

This page covers the native engine's HIP build on a Ryzen AI MAX+ 395: a Radeon 8060S with 40 CUs, gfx1151, wave32,
128 GB unified memory and ~238 GB/s of measured read bandwidth. It lists the checkpoints that serve today, how they
were measured, and what is still missing. The work lives on the `strix/*` branches of this fork and lands on `main`.

## Build

The HIP build needs zig 0.17 and a ROCm 7 SDK with gfx1151 (TheRock nightly builds have it):

```sh
zig build native-hip -Dhip-arch=gfx1151
zig-out/native-hip/bin/tensorfold-native serve <model dir> --drafter <drafter dir> --parallel 8 --context 32768
```

GPU memory is GTT. The default `ttm.pages_limit` caps it well below 128 GB. This box runs with 104 GiB, set with
`ttm.pages_limit=27262976` on the kernel command line. Keep other GPU users stopped while serving the larger models.

## Measured (single stream, C1)

All numbers use Mia's sparkDash method (`tools/strix/sparkbench.py`: greedy, 400 tokens, the first token excluded,
the hash-map prose prompt and the per-stream code tasks). Each was taken under an exclusive GPU lock, with nothing
else building or running. The Spark column is MiaAI-Lab's published TensorFold numbers on one DGX Spark.

| Model | Checkpoint, drafter | Prose C1 | Code C1 | Prefill 8k | Spark prose / code / prefill |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | EXL3 3.5 bpw, DFlash2 (block 16) | 57.6 | 128.8 | 380 | 58.3 / 136.9 / 1,793 |
| Qwen3.8-27B | MLX 4-bit, DFlash2 (block 16) | 43.6 | 112.7 | 562 | same |
| Qwen3.6-35B-A3B | MLX 4-bit, MTP 3 | 112.0 | 147.6 | 2,394 | ~110 / ~130 / 7,353 |
| Qwen3.8 Flash Next | MLX 3-bit, MTP (warm) | 49.4 | 64.2 | ~400 | 64.4 / 57.5 / 2,606 (INT4-AR) |

Every row passes the exactness check: one stream alone, four streams at once and `draft:false` all give the same
bytes. The 27B's prefill cannot reach the Spark's. At ~54 GFLOP a token, 1,793 tok/s needs ~97 TFLOPS against this
GPU's ~57 TFLOPS bf16 peak.

Quality, from `tools/strix/quality.py` (Qwen3.8-27B against its bf16 weights):

| Checkpoint | PPL (bf16 3.0337) | KL | Top-1 |
| --- | --- | --- | --- |
| MLX 4-bit | 3.1853 | 0.1115 | 91.19% |
| EXL3 3.5 bpw | 3.2141 | 0.1162 | 91.87% |

## Reproducing

```sh
source tools/strix/gpu-lock.sh
work zig build native-hip -Dhip-arch=gfx1151
bench tools/strix/bench.sh exl3 ~/models/Qwen3.8-27B-EXL3-3.5bpw ~/models/Qwen3.8-27B-DFlash2
bench tools/strix/bench.sh a3b ~/models/Qwen3.6-35B-A3B-MLX-4bit-MTP -
work python tools/strix/quality.py --model ~/models/Qwen3.8-27B-MLX-4bit   # needs quality_ref.py's bf16 logits
```

`bench` takes the lock exclusively and `work` takes it shared, so a timing never overlaps a build or a test.

## Switches

| Variable | Effect |
| --- | --- |
| `TF_PROMPT_DQ=1` | The 4-bit prompt tile dequantizes to bf16 first: ~13% faster prefill, KL vs bf16 0.1122 instead of 0.1115. Off by default |
| `TF_DRAFT_BLOCK=N` | The DFlash2 drafter's block (default 16) |
| `TF_DRAFT_PROFILE=1` | Prints the drafter hold's per-launch breakdown at load |
| `TF_LANES_DEBUG=1` | Prints each finished stream's rounds, planner choices and accepted lanes |
| `TF_FN_TIMES=1` | Prints a Flash Next round's host phases |

## Not done yet

- Flash Next INT4-AutoRound and NVFP4 checkpoints are being ported. The 3-bit MLX checkpoint is a stopgap.
- The DFlash v1 drafter (the 35B-A3B's) does not load. HIP reads DFlash2 only.
- MiniCPM-4.7 (ciru_q8) is being ported, text only first.
- C8 (eight streams) is not tuned yet.
- A feature-by-feature parity audit against CUDA and Metal is in progress.
