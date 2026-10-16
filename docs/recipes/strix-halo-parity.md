# HIP feature parity: strix/audit on gfx1151

- Date: 2026-10-16.
- Branch: `strix/audit`, rebased on strix/int 3de37350. It adds 4 commits, listed below.
- Smoke tests: `smoke.py` in the session scratchpad (audit/). Each test is one short request with a sanity check, run against a live server, one load per model, under `bench`.
- Exactness: `bench_concurrent --levels 1,4 --tokens 128 --alone --serial` was all equal on every model it ran on: 27B MLX4 + DFlash2, 27B EXL3 + DFlash2, 27B MLX4 plain/copy, 35B-A3B MTP, and Flash Next MLX3 MTP.
- Reference: CUDA native serves only Nemotron. Metal native serves Nemotron, Flash Next, GLM, the 27B and the 2B. Several rows below are therefore gaps in every native backend, not ROCm-only gaps. They are marked `n.a.-native`, and the Python 0.6.6 engine had them. The old ROCm branch is `origin/rocm` in /tmp/claude-1000/TensorFold-old (34994858). Its code is reference only and was not verified.

Legend: **works** = smoke passed · **broken** = present but fails · **missing** = absent on HIP · **n.a.** = does not apply · **fixed** = fixed on strix/audit and re-tested.

Columns:

- **27B-M4**: Qwen3.8-27B MLX 4-bit.
- **27B-X**: Qwen3.8-27B EXL3 3.5 bpw.
- **A3B**: Qwen3.6-35B-A3B MLX 4-bit + MTP.
- **FN3**: Flash Next MLX 3-bit + MTP.
- **FN-INT4 / FN-NVFP4**: Flash Next INT4-AutoRound / NVFP4, being ported by another agent.

## Matrix

| Feature | 27B-M4 | 27B-X | A3B | FN3 | FN-INT4 / FN-NVFP4 |
|---|---|---|---|---|---|
| Load / serve | works: 31 s load, 4 streams at 128k | works: 33 s | works: 30 s | works: 27 s | **missing**: `UnsupportedAffineMode` (another agent is porting it) |
| /health, /v1/models, /metrics | works | works | works | works | – |
| Greedy chat, repeatable | works | works | works | works | – |
| temperature / top_k / top_p / min_p / seed | works: same seed gives the same text, another seed differs | works | works | works | – |
| Malformed sampling gets 400 | works | works | works | works | – |
| presence / frequency / repetition penalty | missing: silently ignored (n.a.-native: no native backend has them) | missing (same) | missing (same) | missing (same) | – |
| logit_bias | missing: silently ignored (n.a.-native) | missing | missing | missing | – |
| logprobs / top_logprobs (chat + completions) | **missing**: 400 "logprobs are not supported by this model or backend" (CUDA Nemotron has them) | **missing** | **missing** | **missing** | – |
| prompt_logprobs | **missing**: 400 | **missing** | **missing** | **missing** | – |
| /v1/completions, raw prompt | works | works | works | works | – |
| json_schema / response_format | **fixed**: was 400 "cannot load libtfgrammar.so"; works with the lib installed (one test reply was cut by max_tokens 80) | works | works | works | – |
| guided_regex / guided_choice / guided_grammar | **fixed**: lib, plus the DFlash2 wrapper's `held` hook (it failed with 500 `HeldDraftsUnreadable`) | works (+DFlash2 after the fix) | works | works | – |
| Tool calls (OpenAI), parsed | works | works | works | works | – |
| tool_choice required / named | **fixed**: same two causes | works | works | works | – |
| Thinking on/off, reasoning_effort | works | works | works | works | – |
| thinking_budget | works: 33 reasoning tokens at budget 32 | works | works | works | – |
| Stop strings, ignore_eos | works | works | works | works | – |
| Streaming + include_usage, same text as unstreamed | works | works | works | works | – |
| /v1/responses (+ GET by id) | works | works | works | works | – |
| Anthropic /v1/messages + count_tokens | works | works | works | works | – |
| Anthropic tools | works | works | works | broken?: the 3-bit model answered in text (end_turn); the OpenAI tool path works, so this is likely the model's choice | – |
| /tokenize, /detokenize | works | works | works | works | – |
| /v1/embeddings | n.a.: no route in any native backend (404) | n.a. | n.a. | n.a. | – |
| /v1/decisions | missing (n.a.-native): "decision scoring is not supported by this engine yet". No native engine implements `score` | missing | missing | missing | – |
| n > 1 refused | works: 400, as upstream | works | works | works | – |
| Prompt cache / radix reuse | works: 1152 of 1220 tokens cached on the second turn | works | works | works | – |
| `--prompt-cache-gib`, `--checkpoint-slots` | works: the startup line shows the plan | works | works | works | – |
| Learned states on disk (`--learn`) | **fixed → refused**: it was silently ignored; Metal-only upstream | refused | refused | refused | – |
| Spill to disk (`--spill-gib`) | missing: "not served by the native engine" (Python-only upstream). Old branch: Flash Next `spill.zig` (CUDA-hipified engine) | missing | missing | missing (old branch had it) | – |
| `--decode-share` | missing (Python-only upstream; the native lane host interleaves chunks itself). Old branch: Python only | missing | missing | missing | – |
| `--prefill-fp8` | missing (Python-only upstream). Old branch: Python, plus Flash Next CUDA-hipified FP8 tiles | missing | missing | missing (old branch had it) | – |
| `--parallel` N, concurrent == solo | works | works | works | works | – |
| MTP drafts | n.a.: this checkpoint has no MTP weights (copy drafts run: accepted 2/30) | works: accepted 47/91, 23/27 | works | works | – |
| DFlash2 drafter, bf16 (`--drafter-bits 4`) | works: C1 code 47 / chat 31 tok/s, exact | works: C1 code 58.7 / chat 44.9 tok/s, exact | n.a.: no A3B DFlash2 | n.a. | – |
| DFlash2 drafter, EXL3 5.0 bpw | n.a. | **missing**: `does not load (MissingTensor)`, "no tensor fc.weight" (EXL3 trellis tensors) | n.a. | n.a. | – |
| DFlash v1 drafter | n.a. | n.a. | **missing**: `does not load (BadDraftConfig)`: HIP reads DFlash2 configs only | n.a. | – |
| Copy / n-gram drafts | works: lane core copy index | works | works | works | – |
| Draft trees | works: DFlash2 trees, planner chains vs trees | works | n.a. (MTP chains) | n.a. | – |
| `"draft": false` equals drafted | works | works | works | works | – |
| Long context, 100k-token needle | works: 100,035 tokens in 262 s (≈380 tok/s), correct answer | works (slow): over 300 s, client timeout, answer not checked | works: 76 s, correct | works: 288 s, correct | – |
| Context past the native 262,144 tokens (YaRN 1M) | **missing**: `--context 1048576 exceeds this model's 262144-token window`. No native code reads `rope_scaling`. Old branch: Flash Next host YaRN (`cuda_rope.zig`) | **missing**: the checkpoint ships `config.yarn-1m.json` | **missing** | **missing** (old branch had it) | – |
| KV dtype fp8 / q8 (`--kv-dtype`) | **missing**: `invalid choice: 'fp8'`; KV pages are bf16. Old branch: `--kv-dtype fp8` native for the Flash Next CUDA-hipified engine, plus kv8 attention notes and tests | **missing** | **missing** | **missing** (old branch had it) | – |
| Pinned KV pool | n.a. on unified memory: the paged KV pool is preallocated in GTT; no host-pinned tier | n.a. | n.a. | n.a. | – |
| Vision: image | **missing**: 400 "accepts text parts only". `--vision` is refused (n.a.-native). Old branch: `server/vision.zig` + `tensorfold.vision.native_helper` (Python tower helper, Flash Next) | **missing** | **missing** | **missing** (old branch had it) | – |
| Vision: 50 images / video | **missing** (same) | **missing** | **missing** | **missing** (old branch: `--vision-max-videos`) | – |
| Tensor parallel (`--tp` 2/4/8 over RCCL) | untested: code present (`native/hip.zig`, `hip/comm`). Flag checks work. Needs a second box | untested | untested | untested: Flash Next has `tp.zig` | – |
| Sliding / Living Weights (`--slide`) | **fixed → refused**: it was silently ignored. Upstream: Nemotron only (CUDA/Metal), so n.a. for Qwen | refused | refused | refused | – |
| Checkpoint formats | MLX affine q2–q8 g64 | EXL3: works but is missing from the `capabilities` formats list | MLX q4 (q8 gates) | MLX q3/q4/q6 g32. EXL3 Flash Next is in code but untested (no checkpoint) | GPTQ/INT4-AR, NVFP4: missing (porting) |
| bf16 checkpoint (Qwen3.8-27B reference) | **missing**: `cannot load … (UnsupportedQuantization)`. The KL quality reference cannot be served on HIP | – | – | – | – |
| FP8 checkpoint | missing: no native family reads FP8 weights (n.a.-native) | – | – | – | – |
| `--chat-template`, `--alias`, `--api-key`, `--dashboard` | works: shared server code, not smoke-tested per model | ” | ” | ” | – |
| `serve --help` lists the HIP flags | **fixed**: it now prints `--tp --policy --mtp-drafts …` | | | | |
| Desktop app on Linux | **missing**: Electron app with only macOS packaging (`pack:mac`/`dist:mac`, darwin onnxruntime); node 22 is pinned (box has 26). The old branch has no app changes, only Docker deploy (`docker/rocm-serve`, PM4 from source), which was **not run** per the no-docker rule | | | | |

## Fixes on strix/audit (4 commits)

1. `65114ae6`: HIP refuses `--learn`, `--slide` and `--segments` at startup instead of silently ignoring them.
2. `520511f7`: `serve --help` lists the backend's own flags.
3. `d6977a69`: `lanes/drafted.zig` gets a `held` hook. Before it, every structured-output request failed with `--drafter` (500 `HeldDraftsUnreadable`). Re-tested on 27B MLX4 + DFlash2: regex, choice, grammar and tool_choice all work, and bench_concurrent is all equal.
4. `ce46346c`: `zig build native-hip -Dxgrammar=DIR` also installs `libtfgrammar.so` into `native-hip/lib`, where the server looks. Without it, structured output on HIP is a 400. **Deploys must pass -Dxgrammar** (xgrammar v0.2.8 checkout).

## Prioritized work items (non-trivial)

1. **logprobs / top_logprobs / prompt_logprobs on HIP**, for qwen3_5, qwen3_5_moe and Flash Next. The user asked for it explicitly. Implement `Backend.first_row`, `Verified.rows` and `prompt_rows` from the verify and prefill logits: a log-softmax and top-20 per row in `hip_serve/draw.zig`. Wire it in `qwen3_5/backend/hip/engine/hip_lanes.zig` and `flashnext/hip/lanes.zig`, and set `Info.logprobs` / `prompt_logprobs` in `native/hip.zig`. The model to copy is CUDA Nemotron (`families/nemotron/cuda_lanes.zig`). Rows must be the raw target distribution, equal with drafts on or off. **Est. 3–5 days.**
2. **FP8 KV cache (`--kv-dtype fp8`)**, a prerequisite for useful 1M context and for more streams. Files: `qwen3_5/backend/hip/forward/pages.zig` (pool dtype), the decode/prefill attention kernels (`zig/kernels/hip/attention/*`), the Flash Next attention, the admission byte counts (`plan.zig` already takes a `kv_dtype` byte count), and the cli `native_values`. Old-branch kv8 notes and kernels are a reference. Needs a KL check against bf16. **Est. 1–1.5 weeks.**
3. **YaRN / 1M context.** Read `rope_scaling` (yarn, factor 4, original 262144) in `qwen3_5/config.zig` and `flashnext/config.zig`. Compute YaRN inv_freq and the attention factor on the host (port the old branch's `cuda_rope.zig`, which has Python bits). Let `hip/admission.zig` grant the scaled window. Validate with a needle at 300k+. Depends on item 2 for memory: 1M bf16 KV is 64 GB per 27B stream. **Est. 2–3 days**, plus item 2.
4. **Vision: images, 50 images, video.** The server must accept media parts, render them through the template, run a vision tower and inject the embeddings plus 3-D mrope positions into the prompt rows (`api.Media`). Fastest route: port the old branch's `server/vision.zig` Python-helper design to qwen3_5. A native tower comes later. Files: `server/fields.zig`/`messages.zig` (the 400 today), `server/cli.zig` (`--vision*` native), the qwen3_5 HIP prefill (embedding injection, mrope sections). **Est. 1 week with the helper; 2–3 weeks native.**
5. **EXL3 DFlash2 drafter** (Mia's 5.0 bpw). The drafter loader expects `fc.weight` and similar; the EXL3 file stores trellis/suh/svh/mul1. Reuse `quant/exl3.hip` products in `qwen3_5/backend/hip/draft/weights.zig` and `draft.zig`. **Est. 2–4 days.**
6. **bf16 27B checkpoint** (`UnsupportedQuantization`): needed to measure KL on HIP itself. A bf16 product exists for the drafter, so extend the qwen3_5 weight table (`weights/table.zig`) to bf16 linears. **Est. 1–2 days.**
7. **Flash Next INT4-AR and NVFP4** (`UnsupportedAffineMode`): already owned by another agent. Tracked here only.
8. **DFlash v1 drafter** (Qwen3.6-35B-A3B-DFlash, `BadDraftConfig`): a v1 config/architecture reader and its forward beside DFlash2 in `qwen3_5/backend/hip/draft/`. **Est. 3–5 days.**
9. **Tensor parallel across two Strix boxes**: the code exists (RCCL, `--tp`/`--master`). It needs a two-host run (27B, A3B, Flash Next), exactness against one rank, and link throughput. **Est. 1–2 days** once a second box is available.
10. **Long-prompt prefill on the 27B**: 100k tokens takes 262 s on MLX4 and over 300 s on EXL3, against 76 s on the A3B. Not a parity gap; listed for the tuning queue.
11. **Desktop app on Linux**: add electron-builder linux targets (AppImage/deb), a linux-hip engine fetch in `scripts/fetch-engine.mjs` and a linux onnxruntime. Node 22 is required. **Est. 1–2 days.**
12. **On-disk learned prompt states (`--learn`) on HIP**: port `core/prompt_imprint.zig` usage, as `native/metal.zig` does, to `native/hip.zig` and the HIP prefix caches. It is now refused rather than ignored. **Est. 2–3 days.**
13. **Spill to disk, `--prefill-fp8`, `--decode-share`**: these are Python-0.6.6 features with no native implementation upstream. The old branch has Flash Next spill and FP8 prefill for its CUDA-hipified engine. Port them if wanted; `--prefill-fp8` is really a performance item. **Est. 2–4 days each.**
14. **Penalties, logit_bias, /v1/decisions (Engine.score)**: missing in every native backend upstream. Penalties and logit_bias are currently ignored *silently*. The cheapest honest step is to reject them with a 400, a small server change; the full version goes in `hip_serve/sample.zig` and must keep draft equality. **Est. 1–2 days.**
15. **Sliding / Living Weights on HIP**: upstream has it for Nemotron only (CUDA/Metal trainer). It does not apply to the Qwen families until upstream extends it. Now refused.
16. Small items:
    - Add `exl3` to the qwen3_5 `formats` list in `capabilities --json`.
    - The HIP help/usage still says `--backend {auto,mlx,cuda}`.
    - The `capabilities` env list names CUDA variables, not `TENSORFOLD_HIP_MEMORY_LIMIT_GB` / `TF_POLICY`.
