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

## Measured

All numbers use Mia's sparkDash method (`tools/strix/sparkbench.py`: greedy, 400 tokens, the first token excluded,
the hash-map prose prompt and the per-stream code tasks; C8 is eight requests at once, aggregate tokens over the
first-to-last token window). Each was taken on `main` (2026-10-18) under an exclusive GPU lock, with nothing else
building or running. The Spark column is MiaAI-Lab's published TensorFold numbers on one DGX Spark; MiniCPM's rival
is Ciru's vLLM build on this same hardware.

| Model | Checkpoint, drafter | C1 prose / code | C8 prose / code | Prefill 8k / 32k | Rival C1 / C8 / prefill 8k |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | EXL3 3.5 bpw, DFlash2 | 62 / 137.5 | 204 / 221 | 675 / 592 | Spark 58.3 / 136.9, 220.6 / 323.2, 1,793 |
| Qwen3.8-27B | MLX 4-bit, DFlash2 | 43 / 116.7 | 150 / 159 | 578 / 514 | same |
| Qwen3.8-27B | GGUF UD-Q4_K_XL (Unsloth), DFlash2 | 41.9 / 98.0 | 125.5 / 158.5 | 677 / 577 | same |
| Qwen3.6-35B-A3B | MLX 4-bit, MTP 3 | 116.8 / 153 | ~255 / ~245 | 2,394 | ~110 / ~130, ~380 / ~383-415, 7,353 |
| Qwen3.8 Flash Next | INT4-AutoRound, MTP (warm) | ~53 / ~69 | not measured | 458 | Spark 64.4 / 57.5, 200.9 / 189.3, 2,606 |
| MiniCPM-V 4.7 (text) | ciru_q8, DSpark | 66.7 / 82.9 | ~131 / ~150 | 2,050 | Ciru: see below |

MiniCPM against Ciru on Ciru's own probes (`tools/strix/longctx.py`, a ~63,870-token prompt): decode 42.4 tok/s
against 40.45; prefill ~1,500 against 1,968.6; eight concurrent requests ~108 against 122.78.

Every row passes the exactness check: one stream alone, four streams at once and `draft:false` all give the same
bytes. At ~54 GFLOP a token the 27B's bf16 prefill ceiling is ~1,000 tok/s; the Spark's 1,793 uses FP8 activations.
During prefill the GPU runs at ~2.75 GHz and ~156 W (clocks are left dynamic), and the quantized products, ~77% of
prefill time, run at ~40-42 TF on real activations.

Quality, from `tools/strix/quality.py` against the fp32 reference (Qwen3.8-27B, 8 sequences of 4,096 tokens):

| Checkpoint | PPL (bf16 3.0359) | KL | Top-1 |
| --- | --- | --- | --- |
| GGUF UD-Q4_K_XL | 3.0656 | 0.0290 | 96.67% |
| MLX 4-bit | 3.1853 | 0.1122 | 91.14% |
| EXL3 3.5 bpw | 3.2193 | 0.1165 | 91.87% |

The GGUF also has a same-weights check: against an fp32 forward of its own dequantized weights the engine scores KL
0.0004 and top-1 99.4% ([gguf.md](gguf.md)). The other checkpoints have no same-weights check yet.

A C8 round of eight prose streams (`TF_ROUND_PROFILE=1`, EXL3) takes ~110 ms against a ~69 ms floor: the verify of
~30 rows 88 ms (EXL3 products ~65 ms against a 48.6 ms weight read; the round tile is issue-bound on the trellis
decode), the drafter's hold beside the keep's DeltaNet replay 22 ms, ~4.5 ms of GPU gaps. Host work is ~0.1 ms.

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
| `TF_DRAFT_SMALL=N` | A shared round's DFlash2 hold runs only blocks of N (default 4 and 8; 0: always the full block) |
| `TF_ROUND_PROFILE=1` | Prints a lane round's GPU verify, keep and hold times and the gaps between them |
| `TF_FILL_ONE=1` | One prompt chunk a round, even for short prompts arriving together |
| `TF_FN_TIMES=1` | Prints a Flash Next round's host phases |

## Not done yet

- **Correctness audit (blocks calling any family done):** a same-weights check for the 27B EXL3/MLX4, quality gates
  with fp32 references for the 35B-A3B and MiniCPM, a statistical test of the top-k/top-p sampler, and YaRN or a
  hard limit past the native context.
- **C8 code** (27B 221 against 323): a row past 32 costs ~1.4 ms of verify, so code streams get ~5 tokens a round at
  C8 against ~10 alone, and shared rounds cap each stream at 7 drafts. Next: a round product that dequantizes each
  weight tile once for 64-128 rows (the prompt tile's arithmetic, so the bits must match it at every round size).
- **Prefill:** the 27B is at ~68% of its bf16 ceiling; the 35B-A3B and Flash Next are far below theirs (expert
  routing). 4-bit activations were measured and rejected on quality (KL 0.112 to 0.277).
- **C8 for the MoE models** (35B-A3B ~65% of its rival; MiniCPM below Ciru): rounds are bound by the experts the rows
  touch.
- **GGUF:** a round tile sharing x across rows (C8 runs at 45-55 GB/s at 128 rows), iq4_nl, zero-copy mmap
  registration; only the qwen3_5 dense family reads GGUF.
- Flash Next: the WMMA prompt-attention branch (`strix/fn-perf`) is unmerged pending its quality check; NVFP4 is not
  ported; C8 is not measured.
- The DFlash v1 drafter (the 35B-A3B's) does not load (`BadDraftConfig`). HIP reads DFlash2 and DSpark.
- MiniCPM: vision is not ported (text only).
- The DFlash2 drafter keeps a 16-row block and bf16-rounded conv sums where vLLM and llama.cpp use drafts + 1 and
  fp32: both were measured, the references' choices were no faster (block 8 was much slower).
- Feature parity against CUDA and Metal: [the matrix](strix-halo-parity.md).
