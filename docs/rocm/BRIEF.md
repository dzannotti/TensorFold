# ROCm/HIP port of the Flash Next engine: shared brief for agents

## Goal
Serve Qwen3.8 Flash Next with TensorFold's Zig engine (`tensorfold-native`) on AMD Strix Halo, matching
MiaAI-Lab's DGX Spark recipe as closely as the hardware allows (their GB10 numbers: 64 tok/s prose / 57.5 code
at 1 request, 201 tok/s aggregate at 8, prefill ~2.5k tok/s at 4k-64k). Checkpoint:
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` @1464274120d3 (GPTQ int4 g128 sym routed experts + lm_head,
block-FP8 128x128 dense linears, FP8 n-gram table, bf16 MTP), local copy `/home/dzannotti/models/qwen38fn-int4-autoround`.
Served flags: `--context 262144 --parallel 8 --kv-dtype fp8 --thinking`, `TF_FLASHNEXT_DEPTH=15`, TP=1. Vision and
two-rank (RoCE/NCCL) are out of scope.

## Tree
- Branch `rocm`, worktree `/home/dzannotti/tf-rocm` = TensorFold db28187 + MiaAI-Lab patches 0001-0009 + homelab
  0010-0011 (commit 05d8390). This is exactly the engine thorim (a GB10 Spark) serves.
- Agents working in parallel use their own worktree (`git -C /home/dzannotti/tf-rocm worktree add -b <branch>
  /home/dzannotti/tf-<name> rocm`) and commit there; the orchestrator merges. Stay inside your assigned files.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Author: Daniele Zannotti
  <d.zannotti@gmail.com>. Never push.
- Keep the CUDA build working: HIP is selected with a build option (`-Dgpu=hip`); shared `.cu` sources use
  `#if defined(__HIP_PLATFORM_AMD__)` branches or the compat header, not forks, unless a kernel is a rewrite.
- Repo rules (CONTRIBUTING.md): one job per module, files <= 600 lines where practical, one-line comments.

## Hardware: gfx1151 (Radeon 8060S, RDNA 3.5)
40 CUs (20 WGPs), wave32 (warpSize 32), 64 KiB LDS per workgroup, WMMA 16x16x16 (f16/bf16 -> f32, iu8/iu4 -> i32;
RDNA3 operand layout: A/B fragments replicated across lane halves), NO hardware FP8/FP4, 2 MiB L2 + 32 MiB MALL,
LPDDR5X-8000 256-bit unified memory (~256 GB/s peak). GPU memory is GTT (system RAM), ~104 GiB limit.

## Toolchain
`/home/dzannotti/tf-rocm/docker/rocm-dev/run.sh [cmd...]` runs image `tf-rocm-dev:latest` with GPU access, mounts
/home/dzannotti rw (models ro), workdir the rocm worktree (pass `-w`-style `cd` in your command for other worktrees).
Inside: zig 0.17.0, /opt/rocm HIP 7.15 (hipcc, amdclang++, hipblaslt, rocprofv3, hipify-*), python venv with torch
2.11+rocm7.13 and triton 3.6 (gfx1151, warp 32). torch processes load torch's bundled HIP 7.13 runtime.
`/opt/venv` is read-only: `pip install --user` if needed.

## The machine is shared with PRODUCTION
Prod (ornith vLLM, ComfyUI, embedding, reranker) holds ~80 GiB of GTT and often keeps the GPU busy. Until a test
window: GPU work must stay small (a few GiB, short runs). Never stop/restart/modify containers or images you did not
create; never load the full model; do not touch /srv, /homelab or other services. Timings taken now are noisy:
report them as relative (A/B alternated) and note it. CPU builds: `nice`, <= 12 jobs.

## Correctness contract on ROCm
CUDA-bit-equality is impossible (different MMA order, libm, Triton codegen). What must hold:
1. Row/batch invariance: a row's output bits never depend on how many other rows, streams or drafts share the
   launch, nor on its position (this is what makes drafted == plain and stream == solo hold). Split-K or tile
   choices may depend on shapes known per *model* but never on M (rows), or every M must reduce in the same order.
2. Determinism: same input -> same bits, run to run (no float atomics in reductions).
3. Kernels with a Python/Triton counterpart: byte-equal to that counterpart run on ROCm torch/Triton (same source,
   same options). Torch-op replacements: byte-equal to ROCm ATen.
4. Kernels without a counterpart (int4, fp8 block, etc.): compare to a host fp64 reference; error no worse than one
   rounding of the output dtype plus the documented accumulation; report worst element and per-column worst.
5. End to end (later): drafted == `"draft": false`, concurrent == solo, resumed == fresh, chunk invariance, and
   teacher-forced top-1 agreement vs the CUDA engine on thorim.

## Reporting
Final report: what changed (files, commits), what was verified and how (commands + results), what is left, open
risks. Keep it under ~800 words. Put long logs in files under docs/rocm/notes/ or the scratchpad, not the report.
