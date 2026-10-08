# Handover from the resumed session (2026-10-09 ~00:20 BST)

The user asked this session (the `claude -c` / remote-control continuation of the same conversation) to stop and hand
everything back to the original session (pid 568461, "tensorfold-5c"). Nothing of mine is still running.

## Window state
- User approved starting early: at 00:06 I stopped `tf-window-down.timer` and ran `~/tf-window/down.sh` by hand.
  `stopped.list` = trellis comfyui ornith reranker embedding. `tf-window-up.timer` still fires 10:30 (stops tfdev-*,
  restarts that list). Do NOT re-run down.sh (it would overwrite stopped.list with an empty list).
- My window-operator agent ran WINDOW.md §2-§6a on `rocm` @ f3696cd with set `/home/dzannotti/tf-triton-scratch/aot-hip2`,
  then I stopped it and container `tf-window` (00:2x). `GPU_EXCLUSIVE` removed. Raw results copied to
  `~/tf-window/results-0009/`.

## Results so far (rocm @ f3696cd, full model, prod down)
- Pre-flight: tf-cuda-test smoke/graph/occupancy/vmm and int4/qsa/fp4/experts/glue/mm checks all PASS.
- CLI load 48 s, 65.63 GiB weights (1198 buffers), low-water MemAvailable ~49 GiB. Serve loaded in 79.7 s, context 262144.
- CLI sky (102 tokens, thorim prompt): drafted == plain (sha b64935d893d5), 59/71 drafts accepted.
  **Plain decode 63 ms/token, drafted 35 ms/token** — ~3x over the single-row bandwidth bound: the top perf question.
- contracts quick (a,b; 4 concurrent): 8/8 PASS. Full contracts: 54 PASS, 0 FAIL, 6 UNCHECKED (resume incl. shared
  system prompt, 18k-token chunk crossings, chat resumed from cache).
- Agreement vs thorim (teacher-forced, 20 prompts): 3076/3101 positions = 99.19% (PASS, min 98%); 7/20 free-run identical.
- NOT yet run: bench.py (Mia-method speed table), rocprofv3 profile.

## Pending candidate (not merged into rocm)
- Branch `rocm-next`, worktree `/home/dzannotti/tf-next` = rocm + `rocm-tritonperf` (gfx1151 Triton retune: router
  73->11 us, prefill b16mm/hc_up_mix 2-7x, fixes a spill miscompile of masked loads on partial tiles; both pointer
  forms, Zig picks buffer ops by allocation) + `rocm-fp8perf` (LDS-staged wide FP8 tile from 33 rows, prefill 25->29-35
  TF, same bits). Built: `tf-next/zig-out/{bin/tensorfold,native/bin/tensorfold-native}`; kernel set
  `/home/dzannotti/tf-triton-scratch/aot-hip3` (830 entries, = tf-tritonperf-scratch/aot-new). Host tests pass.
  Suggested: correctness (CLI sky, contracts a,b,c, agreement) then the same bench A/B vs rocm, then profile.

## Other notes
- `rocm-drafthead` merged (8411967) + engine switch 22e6261: HIP drafts over an int4 slice of the 79,591 draft
  columns (0.50 ms vs 1.45 ms full head, logits bit-equal). `layer_view.py --mtp` keeps the MTP layer (needs
  TF_FLASHNEXT_INT4AR_FAST=1 for its MTP tensors; the 1-layer view's MTP gives NaN drafts — full model is fine).
- FP8 agent: the earlier "16-row = 2x 8-row" anomaly was prod noise, not a kernel issue.
- Budget model (estimates, not measurements): prefill ~1.3-1.4k tok/s now, ~2.0-2.2k with fixes vs Mia 2.5k;
  1-stream decode ~46 -> ~56 vs Mia 64.4.
