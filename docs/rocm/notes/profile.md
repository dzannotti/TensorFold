# Profile: Flash Next on gfx1151, rocm @ b768ba8 (runtime ROCm 10.0 / HIP 7.15, Triton set aot-hip2)

Window 2026-10-09, prod down, GPU exclusive. Raw logs: `~/tf-window/perf/`. All decode numbers: depth 15, fp8 KV,
context 262144, served stop rule (`--served`: running product -0.4 at <= 2 streams, 0.5 above), greedy, 256 tokens,
EOS ignored, sparkDash prompts (`bench-many --kind dash-prose,dash-code`, the same prompts bench.py sends).

## Tooling notes (read first)
- `rocprofv3 --kernel-trace` HANGS the engine in graph mode (server warm-up and CLI bench-many alike): the first
  replayed forward never completes (`queue_interposition.cpp: Async signal handler still waiting on signal`), 1,520
  dispatches queued with start == end. Not fixed by GPU_MAX_HW_QUEUES=1. `rocprofv3 --attach` fails (ptrace, scope 1).
  Works with `--eager`, so kernel-level numbers below are eager (kernel durations are the same; gaps are not).
- The engine's own part profiler (`TF_FLASHNEXT_PROFILE=1`, eager, events between parts) gives per-part GPU timeline.
- Server and CLI agree: server prose 1-req 49.3 tok/s, 100 rounds, 155/192 accepted; CLI bench-many 49.7 tok/s,
  100 rounds. So the CLI is a faithful stand-in.
- Our env matches Mia's served env except vision: engine defaults are TF_FLASHNEXT_PRODUCT_STREAMS=2 and
  TF_FLASHNEXT_PREFILL_TAIL=512 (cuda_engine.zig), DEPTH=15 set. Differences: TENSORFOLD_MEMORY_RESERVE_GIB 12 (engine
  default) vs Mia 10 (only sizes the pool), no TF_FLASHNEXT_YARN (=0, same).
- Our build warns `no groups-of-32 4-bit kernels in this build: MTP drafts use the full head` (aot-hip2 lacks the
  draft-head slice kernels): drafts read the full 0.31 GiB int4 head, ~1.3 ms a draft level.

## 1a. Single-stream decode, graphed (CLI bench-many, steady reps)

| | prose x1 | code x1 | prose x8 | code x8 |
|---|---:|---:|---:|---:|
| tok/s (aggregate) | 49.7 | 91.7 | 164.6 | 205.8 |
| rounds / 256 tok a stream | 100 | 38 | 109 | 62 |
| tokens a stream-round | 2.55 | 6.71 | 2.65 | 5.37 |
| round wall (ms) | 51.3 | 72.9 | 113.9 | 160.1 |
| verify (ms; host enqueue) | 44.2 (4.7) | 54.7 (34.4) | 93.0 (8.3) | 121.2 (15.3) |
| drafting, MTP (ms) | 6.9 | 18.1 | 19.6 | 37.7 |
| draws+commits (ms) | 0.2 | 0.4 | 1.1 | 1.0 |

Graphs save only ~2.6 ms a round over eager (prose verify 44.2 graphed vs 46.8 eager), i.e. host/launch overhead is
NOT the main cost: the GPU is busy. Thorim (same engine, GB10) prose 65.2 tok/s at 2.55 tok/round => ~39 ms a round;
we are 51 ms. Code thorim 122.7 => ~54.7 ms a round (if same acceptance) vs our 72.9.

## 1b. Where a decode round goes (TF_FLASHNEXT_PROFILE=1, eager, ms a round, steady rep)

Main-model parts (48 layers; 36 GDN + 12 full attention):

| part | prose x1 | code x1 | prose x8 | code x8 | weights read (fp8 unless noted) | bandwidth floor @242 GB/s | achieved x1 |
|---|---:|---:|---:|---:|---|---:|---:|
| gdn_proj (in_proj qkv,z,a,b) | 8.78 | 8.80 | 9.15 | 10.52 | 1.43 GiB | 6.3 ms | 175 GB/s |
| hc_down (2x mix down/layer) | 4.75 | 4.72 | 7.54 | 9.29 | 0.59 GiB | 2.6 ms | 133 GB/s |
| hc_up (2x mix up/layer) | 3.54 | 3.55 | 4.37 | 4.20 | 0.59 GiB | 2.6 ms | 178 GB/s |
| out_proj (GDN out + attn o) | 4.08 | 4.15 | 5.59 | 6.11 | 0.70 GiB | 3.1 ms | 185 GB/s |
| attn_proj (q,k,v,indexer) | 2.58 | 2.60 | 2.58 | 3.33 | 0.43 GiB | 1.9 ms | 180 GB/s |
| router (bf16 gate) | 2.37 | 2.39 | 2.37 | 2.51 | 0.117 GiB | 0.52 ms | **53 GB/s** |
| topk_plan | 0.51 | 0.52 | 0.54 | 0.57 | - | - | |
| experts_gate_up (int4 + shared) | 6.03 | 10.24 | 17.74 | 26.98 | per distinct expert 1.7 MB/layer | | |
| experts_down | 3.38 | 5.56 | 9.31 | 14.42 | | | |
| attention (12 layers + QSA) | 3.00 | 3.91 | **19.29** | **21.86** | KV tiny (prompt ~50 tokens) | ~0 | |
| gdn_chain | 1.51 | 2.03 | 6.18 | 6.93 | state 36 x 48 x 128 x 128 f32 | | |
| head (int4 lm_head) | 1.42 | 1.43 | 2.61 | 3.92 | 0.31 GiB | 1.36 ms | 230 GB/s |
| ple + hc_* glue + reduce + rest | ~4.6 | ~4.7 | ~6.5 | ~8 | | | |
| **main total** | **47.2** | **55.2** | **95.2** | **119.4** | ~4.6 GiB dense + experts | ~27 ms (x1 prose) | |
| MTP head (drafting) | 6.97 | 18.2 | 20.2 | 38.7 | | | |

MTP draft levels (prose x1): absorb 2.66, L1 2.14, L2 1.13, L3 0.56 ... ms a round; each draft level costs ~1.3 ms of
full int4 head (mtp.head 1.29 ms/round prose, 3.34 code) + attn_proj/out_proj (bf16 MTP layer).

Key decode findings (priority order):
1. **Dense fp8 matvecs run at 133-185 GB/s, not 242.** gdn_proj + hc_down + hc_up + out_proj + attn_proj = 23.7 ms a
   prose round vs a 16.5 ms floor. hc_down is worst (133 GB/s). Getting all to ~220 GB/s saves ~6 ms/round (12%).
2. **Router: 2.37 ms for 0.117 GiB = 53 GB/s** (~1.85 ms lost a round). rocm-next's retune claims router 73 -> 11 us
   per launch (48 launches: 3.5 -> 0.5 ms), which matches this gap.
3. **8-stream attention: 19.3 ms a round (3.0 at 1 stream)** with ~50-token contexts: this is launch/occupancy, not
   bytes. Plus gdn_chain 6.2 vs 1.5 and mtp.attention 4.2 vs 0.6. At 8 streams this is ~20% of the round.
4. MTP drafting 7 ms (prose) / 18 ms (code) a round: the full int4 head per draft level (draft-head slice kernels
   missing from aot-hip2), and the bf16 MTP layer's attn/out projections ~2.4 ms a round prose.


## 1c. Kernel-level decode (rocprofv3 --kernel-trace, CLI bench-many --eager, steady rep)

Eager under rocprofv3 runs at the graphed speed (prose x1 49.1 tok/s vs 49.7 graphed; code x1 94.8 vs 91.7: graphs
buy nothing on HIP here, code is even slightly faster eager). Windows are steady reps cut from the trace; "busy
(union)" is the union of kernel intervals; kernel-sum > busy because consecutive dispatches overlap in the trace
(tails/queueing), so per-kernel ms/round are slight over-counts: use them for ranking.

Per round: prose x1 GPU busy 47.5 of 52.0 ms (91%; ~4.5 ms idle a round = host gaps between verify, draws and draft
levels); code x1 64.5 of 70.8 (91%); prose x8 104.9 of 114.9 (91%); code x8 145.6 of 162.5 (90%). Launches a round:
~1,740 (prose x1) to ~3,000 (code x8).

Kernel -> role: `tf_fn_qmmf::qmmf_kernel<3,R,..>` block-FP8 dense matvec/matmul (R = row tile 16/32/64: out_proj,
attn o/k/v, shared expert, ...); `tf_fn_qmmf_ld` the LDS-staged FP8 variant, 1 launch a layer = the big fused input
projection (GDN in_proj qkv+z / attention q); `_b16mm` Triton bf16 matmul = hyper-connection mix down/up (bf16, 6.5 MB
a weight, 4 a layer) + bf16 MTP-layer projections; `tf_int4_128_1_1_2_2` / `_1_1_1_3` GPTQ int4 routed experts
gate+up / down (cuda_int4); `_chunks8` Triton attention (12 full-attention layers + MTP); `_router` Triton router
(bf16 gate, 512 experts), then `_topk_rows` + `tf_experts::plan_kernel`; `tf_fn_gdn_tree::tree_kernel` GDN chain
(recurrent state); `tf_fn_gdn_io::front/back` GDN conv/gating; `_fp4mm`, `tf_fn_nvfp4_*`, `tf_fn_fp4_serial` NVFP4
MTP-layer experts / shared paths; `tf_argmax_rows_i32`, `tf_fn_lse_max` draws; `_hc_*`, `_reduce`,
`tf_strided_copy` glue (1.5-4 us each, ~600 launches a round, ~2 ms).

### prose1: window 4.50s = 87 rounds at 52.0 ms; GPU busy (union) 91.4% ; kernel-sum 108.0% ; dispatches/round 1741; busy/round 47.52 ms; kernel-sum/round 56.18 ms
| kernel | ms/round | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `tf_fn_qmmf::qmmf_kernel<3, 16, 64, 1, 4, 4, false, false, ` | 11.83 | 21.1 | 144.0 | 82.1 |
| `_b16mm` | 11.75 | 20.9 | 272.4 | 43.1 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 16, 64, 1, 4, 4, false, fals` | 10.33 | 18.4 | 48.0 | 215.1 |
| `tf_int4_128_1_1_1_3` | 5.81 | 10.3 | 51.9 | 112.1 |
| `tf_int4_128_1_1_2_2` | 5.56 | 9.9 | 48.0 | 115.8 |
| `_chunks8` | 3.20 | 5.7 | 14.8 | 215.4 |
| `_router` | 2.29 | 4.1 | 50.8 | 45.1 |
| `tf_fn_gdn_tree::tree_kernel<float, 0, 8, 4, true>` | 1.33 | 2.4 | 36.0 | 36.9 |
| `_reduce` | 0.45 | 0.8 | 165.9 | 2.7 |
| `_fp4mm` | 0.42 | 0.8 | 5.7 | 74.7 |
| `tf_strided_copy_kernel` | 0.40 | 0.7 | 98.9 | 4.1 |
| `_hc_writeback` | 0.35 | 0.6 | 106.5 | 3.3 |
| `tf_argmax_rows_i32_kernel` | 0.28 | 0.5 | 3.8 | 72.0 |
| `_hc_mix` | 0.26 | 0.5 | 105.5 | 2.5 |
| `tf_fn_nvfp4_shape::expert_nt_kernel<2, 2, 1, 2, 1>` | 0.21 | 0.4 | 2.0 | 105.6 |
| `_hc_normed` | 0.17 | 0.3 | 105.5 | 1.6 |
| `_hc_act` | 0.16 | 0.3 | 105.5 | 1.5 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>` | 0.14 | 0.3 | 2.8 | 51.0 |
| `tf_fn_shared_swiglu_kernel` | 0.14 | 0.3 | 50.9 | 2.8 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>` | 0.14 | 0.2 | 0.9 | 163.6 |
| `_topk_rows` | 0.14 | 0.2 | 50.8 | 2.7 |
| `tf_fn_gdn_io::front_kernel<16, 48>` | 0.13 | 0.2 | 36.0 | 3.7 |
| `tf_experts::plan_kernel` | 0.12 | 0.2 | 50.8 | 2.4 |
| `tf_fn_lse_max_kernel` | 0.10 | 0.2 | 2.8 | 35.8 |
| `_attn_prep8` | 0.10 | 0.2 | 14.8 | 6.6 |

### prose8: window 11.00s = 96 rounds at 114.9 ms; GPU busy (union) 91.3% ; kernel-sum 113.6% ; dispatches/round 2661; busy/round 104.88 ms; kernel-sum/round 130.47 ms
| kernel | ms/round | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `tf_fn_qmmf::qmmf_kernel<3, 32, 64, 1, 4, 4, false, false, ` | 24.32 | 18.6 | 119.4 | 203.6 |
| `_chunks8` | 20.40 | 15.6 | 109.2 | 186.8 |
| `_b16mm` | 18.54 | 14.2 | 277.6 | 66.8 |
| `tf_int4_128_1_1_2_2` | 17.54 | 13.4 | 44.8 | 391.4 |
| `tf_int4_128_1_1_1_3` | 14.22 | 10.9 | 50.8 | 279.9 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 32, 64, 1, 4, 4, false, fals` | 8.91 | 6.8 | 39.8 | 223.8 |
| `tf_fn_gdn_tree::tree_kernel<float, 0, 8, 4, true>` | 6.08 | 4.7 | 33.2 | 183.0 |
| `tf_fn_qmmf::qmmf_kernel<3, 64, 64, 1, 4, 4, false, false, ` | 3.17 | 2.4 | 10.5 | 303.3 |
| `_router` | 2.28 | 1.7 | 49.9 | 45.7 |
| `tf_argmax_rows_i32_kernel` | 1.96 | 1.5 | 27.8 | 70.4 |
| `tf_strided_copy_kernel` | 1.75 | 1.3 | 98.5 | 17.8 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>` | 1.10 | 0.8 | 3.8 | 289.2 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 64, 64, 1, 4, 4, false, fals` | 0.97 | 0.7 | 3.5 | 279.4 |
| `_fp4mm` | 0.86 | 0.7 | 9.2 | 93.2 |
| `tf_fn_lse_max_kernel` | 0.73 | 0.6 | 20.5 | 35.5 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>` | 0.64 | 0.5 | 5.1 | 125.3 |
| `tf_fn_qmmf::qmmf_kernel<3, 16, 64, 1, 4, 4, false, false, ` | 0.59 | 0.5 | 4.5 | 130.9 |
| `tf_fn_fp4_serial_kernel` | 0.59 | 0.4 | 0.9 | 629.8 |
| `_reduce` | 0.54 | 0.4 | 171.8 | 3.1 |
| `_hc_writeback` | 0.50 | 0.4 | 105.8 | 4.8 |
| `tf_fn_gdn_io::front_kernel<16, 48>` | 0.48 | 0.4 | 33.6 | 14.2 |
| `_attn_prep8` | 0.46 | 0.4 | 109.2 | 4.2 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 16, 64, 1, 4, 4, false, fals` | 0.32 | 0.2 | 1.5 | 213.6 |
| `_hc_mix` | 0.31 | 0.2 | 104.8 | 3.0 |
| `_hc_normed` | 0.31 | 0.2 | 104.8 | 3.0 |

### code1: window 3.30s = 47 rounds at 70.8 ms; GPU busy (union) 91.1% ; kernel-sum 112.5% ; dispatches/round 1871; busy/round 64.49 ms; kernel-sum/round 79.67 ms
| kernel | ms/round | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `_b16mm` | 16.25 | 20.4 | 292.8 | 55.5 |
| `tf_fn_qmmf::qmmf_kernel<3, 16, 64, 1, 4, 4, false, false, ` | 15.29 | 19.2 | 129.8 | 117.8 |
| `tf_int4_128_1_1_2_2` | 10.09 | 12.7 | 44.3 | 227.7 |
| `tf_int4_128_1_1_1_3` | 9.84 | 12.4 | 52.2 | 188.6 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 16, 64, 1, 4, 4, false, fals` | 9.37 | 11.8 | 43.3 | 216.5 |
| `_chunks8` | 5.23 | 6.6 | 18.0 | 289.9 |
| `_router` | 2.40 | 3.0 | 51.3 | 46.7 |
| `tf_fn_gdn_tree::tree_kernel<float, 0, 8, 4, true>` | 1.70 | 2.1 | 32.5 | 52.3 |
| `tf_fn_qmmf::qmmf_kernel<3, 64, 64, 1, 4, 4, false, false, ` | 1.44 | 1.8 | 3.1 | 465.1 |
| `_fp4mm` | 1.06 | 1.3 | 13.6 | 78.0 |
| `tf_strided_copy_kernel` | 1.03 | 1.3 | 95.6 | 10.8 |
| `tf_fn_nvfp4_shape::expert_nt_kernel<2, 2, 1, 2, 1>` | 0.64 | 0.8 | 6.1 | 105.3 |
| `tf_argmax_rows_i32_kernel` | 0.54 | 0.7 | 7.9 | 68.9 |
| `_reduce` | 0.50 | 0.6 | 183.5 | 2.7 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 64, 64, 1, 4, 4, false, fals` | 0.45 | 0.6 | 1.0 | 437.7 |
| `_hc_writeback` | 0.37 | 0.5 | 109.2 | 3.4 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>` | 0.36 | 0.4 | 7.0 | 51.1 |
| `_hc_up_mix` | 0.28 | 0.4 | 2.1 | 132.8 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>` | 0.27 | 0.3 | 0.9 | 300.7 |
| `_hc_mix` | 0.27 | 0.3 | 108.3 | 2.5 |
| `tf_fn_lse_max_kernel` | 0.25 | 0.3 | 7.0 | 35.8 |
| `tf_fn_gdn_io::front_kernel<16, 48>` | 0.20 | 0.2 | 33.2 | 5.9 |
| `_hc_normed` | 0.19 | 0.2 | 108.3 | 1.7 |
| `_hc_act` | 0.18 | 0.2 | 110.4 | 1.6 |
| `_hc_wb_norm` | 0.15 | 0.2 | 2.1 | 70.2 |

### code8: window 9.00s = 55 rounds at 162.5 ms; GPU busy (union) 89.6% ; kernel-sum 114.3% ; dispatches/round 2994; busy/round 145.62 ms; kernel-sum/round 185.80 ms
| kernel | ms/round | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `tf_fn_qmmf::qmmf_kernel<3, 64, 64, 1, 4, 4, false, false, ` | 37.70 | 20.3 | 108.0 | 349.0 |
| `_chunks8` | 28.08 | 15.1 | 115.7 | 242.7 |
| `tf_int4_128_1_1_2_2` | 26.89 | 14.5 | 42.1 | 639.1 |
| `_b16mm` | 24.84 | 13.4 | 303.5 | 81.9 |
| `tf_int4_128_1_1_1_3` | 22.65 | 12.2 | 52.1 | 434.9 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 64, 64, 1, 4, 4, false, fals` | 10.65 | 5.7 | 36.0 | 295.9 |
| `tf_fn_gdn_tree::tree_kernel<float, 0, 8, 4, true>` | 6.89 | 3.7 | 31.2 | 221.0 |
| `tf_fn_qmmf::qmmf_kernel<3, 32, 64, 1, 4, 4, false, false, ` | 3.98 | 2.1 | 18.2 | 218.9 |
| `tf_argmax_rows_i32_kernel` | 3.04 | 1.6 | 44.3 | 68.7 |
| `_router` | 2.52 | 1.4 | 51.2 | 49.2 |
| `tf_strided_copy_kernel` | 2.38 | 1.3 | 100.5 | 23.7 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>` | 2.04 | 1.1 | 7.2 | 282.0 |
| `_fp4mm` | 1.93 | 1.0 | 17.4 | 110.8 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 32, 64, 1, 4, 4, false, fals` | 1.37 | 0.7 | 6.1 | 226.1 |
| `tf_fn_lse_max_kernel` | 1.34 | 0.7 | 37.8 | 35.6 |
| `tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>` | 1.13 | 0.6 | 9.1 | 124.1 |
| `_hc_writeback` | 0.78 | 0.4 | 112.3 | 6.9 |
| `tf_fn_gdn_io::front_kernel<16, 48>` | 0.69 | 0.4 | 31.5 | 21.9 |
| `_reduce` | 0.66 | 0.4 | 191.2 | 3.4 |
| `tf_fn_fp4_serial_kernel` | 0.64 | 0.3 | 0.8 | 792.5 |
| `_attn_prep8` | 0.51 | 0.3 | 115.7 | 4.4 |
| `_hc_mix` | 0.45 | 0.2 | 111.4 | 4.1 |
| `tf_fn_shared_swiglu_kernel` | 0.45 | 0.2 | 51.2 | 8.8 |
| `_hc_normed` | 0.44 | 0.2 | 111.4 | 4.0 |
| `__amd_rocclr_copyBuffer` | 0.39 | 0.2 | 294.1 | 1.3 |


Decode reading:
- prose x1: fp8 dense (qmmf + qmmf_ld) 22.2 ms + bf16 `_b16mm` 11.8 ms = 34 ms of a 47.5 ms busy round. Dense bytes a
  round ~4.6 GiB (checkpoint sum, incl. 1.17 GiB of bf16 hc mixes and the 0.31 GiB head) = 20.5 ms at 242 GB/s.
  `_b16mm` at 43 us for a 6.5 MB weight = ~150 GB/s; qmmf_ld 215 us for the ~40 MB fused in-projection = ~190 GB/s.
- int4 experts: 11.4 ms a round prose x1 (2.9 rows), 19.9 code x1, 31.8 prose x8, 49.5 code x8.
- `_chunks8` attention: 3.2 ms (x1) -> 20.4 (prose x8) -> 28.1 ms (code x8) at contexts < 400 tokens: ~110 launches
  a round at 190-240 us each at x8. It scales with streams x rows, not bytes: the biggest x8-specific waste.
- `tf_fn_qmmf` switches to the 64-row tile at x8 (349 us a launch, 37.7 ms a round code x8): 40-row FP8 matmuls cost
  ~4x a 16-row matvec, no longer bandwidth-bound (no FP8 hardware: dequant + FMA bound). A WMMA bf16 path after
  dequant to LDS would help x8.
- `tf_fn_gdn_tree` 1.3 ms (x1) -> 6.1-6.9 ms (x8).

## 1d. Prefill, 8k tokens (CLI `bench`, no server, no prompt cache)

Is server prefill inflated by non-compute work (kept prompt states, snapshots, spill, n-gram/PLE host work)? No:

| run (8,212-token prompt, 4 chunks of 2048 + tail 20) | prefill s | tok/s |
|---|---:|---:|
| server bench.py (3 fresh prompts, prompt cache on) | 16.8 / 17.2 / 18.0 | 488 / 476 / 456 |
| CLI `bench` graphed, rep 1 (no cache, no server) | 16.53 | 497 |
| CLI `bench` 32k (32,776 tokens), rep 1 | 67.30 | 487 |
| CLI `bench` TF_FLASHNEXT_PROFILE=1, rep 1 | 14.93 | 550 |
| CLI `bench --eager` under rocprofv3, rep 1 | 19.36 | 424 |
| **first prefill after load (any mode)** | 34.0-40.4 | 200-241 |

- The server costs <= 4% over the bare engine; spill is not involved (TENSORFOLD_SPILL_DIR unset, nothing written).
  So the cache/state saves do not explain Mia 2.6k vs thorim 635: that gap is outside this box (thorim's own
  numbers are the right target; ours = 0.75-0.78x thorim).
- **The first prefill after load is ~2x slower** (part profiler: `stage` = 16.2 s in chunk 0 and 2.4 s in chunk 1 of
  the first prompt, 13 ms after). Rep 0 under rocprofv3 shows multi-second GPU-idle holes. This is one-time staging
  (first touch of prompt buffers / lazily loaded code objects); bench.py's 4k row in window-results (276 tok/s) was
  the first prompt and paid it. Warm the prompt path at load (one 2048-row dummy chunk) and the first user pays
  nothing. Engine change, cheap.
- Chunking is as expected: 2048 + 2048 + 2048 + 2068 rows (PREFILL_TAIL 512 absorbs the 20-row tail). Chunk time
  grows with position (3.0 -> 3.9 -> 3.9 -> 4.1 s): attention.
- GPU is ~100% busy during a warm prefill (union 99.7%): no host gaps. It is all kernel time.

Parts (TF_FLASHNEXT_PROFILE, warm 8k, ms per 2048 rows; total 3,723 ms):
attention 1,511 (40.6%) + mtp.attention 126 (3.4%); hc_down 482 (12.9%); hc_mix 329 (8.8%); gdn_proj 240 (6.5%);
experts_gate_up 234 (6.3%); out_proj 140 (3.8%); hc_writeback 108 (2.9%); gdn_back 93; experts_down 89; gdn_chain 84;
attn_proj 69; gdn_front 60; router 33; the rest < 100.

Kernels (rocprofv3, eager, warm rep; this run was slower overall, 4.74 s a chunk, with attention at 2.6 s):

### prefill8k: window 19.01s = 4 rounds at 4750.0 ms; GPU busy (union) 99.7% ; kernel-sum 103.4% ; dispatches/round 2065; busy/round 4737.39 ms; kernel-sum/round 4911.02 ms
| kernel | ms per 2048-row chunk | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `_chunks8` | 2615.59 | 53.3 | 104.7 | 24984.6 |
| `_b16mm_ks` | 526.86 | 10.7 | 148.2 | 3556.0 |
| `_hc_up_mix` | 379.32 | 7.7 | 96.9 | 3912.9 |
| `tf_fn_qmmf::qmmf_kernel<3, 64, 64, 1, 4, 4, false, false, ` | 291.27 | 5.9 | 143.2 | 2034.5 |
| `tf_fn_qmmf_ld::qmmf_kernel<3, 64, 64, 1, 4, 4, false, fals` | 288.37 | 5.9 | 48.0 | 6011.2 |
| `tf_int4_128_2_2_2_2` | 235.39 | 4.8 | 47.7 | 4932.6 |
| `_hc_wb_norm` | 104.30 | 2.1 | 96.9 | 1075.9 |
| `tf_fn_gdn_io::back_kernel<16, 48>` | 90.07 | 1.8 | 36.0 | 2503.5 |
| `tf_int4_128_2_2_1_3` | 88.96 | 1.8 | 47.7 | 1864.2 |
| `tf_fn_gdn_prefill::chain_kernel<float, 128, 32>` | 87.36 | 1.8 | 36.0 | 2428.0 |
| `tf_fn_gdn_io::front_kernel<16, 48>` | 59.65 | 1.2 | 36.0 | 1658.0 |
| `_router` | 31.67 | 0.6 | 48.5 | 653.4 |
| `_merge` | 22.45 | 0.5 | 104.4 | 214.9 |
| `_attn_gate` | 14.33 | 0.3 | 12.5 | 1147.3 |
| `_b16mm` | 14.07 | 0.3 | 4.7 | 2964.4 |
| `tf_strided_copy_kernel` | 12.32 | 0.3 | 96.4 | 127.8 |
| `_scores` | 8.43 | 0.2 | 104.7 | 80.5 |
| `_attn_prep8` | 6.12 | 0.1 | 12.7 | 480.4 |
| `tf_fn_shared_swiglu_kernel` | 5.85 | 0.1 | 48.5 | 120.8 |
| `_fp4mm` | 4.53 | 0.1 | 0.7 | 6049.8 |
| `_select` | 4.26 | 0.1 | 104.7 | 40.7 |
| `_hc_writeback` | 2.95 | 0.1 | 2.5 | 1179.2 |
| `_ple_conv` | 2.70 | 0.1 | 1.0 | 2704.7 |
| `fn_prompt4_gu_t2` | 2.65 | 0.1 | 0.7 | 3532.2 |
| `fn_prompt4_b16_t4` | 2.58 | 0.1 | 0.7 | 3437.8 |


Achieved rates (FLOPs from shapes; active ~4.7 B params => ~19 TFLOP of matmul a 2048-row chunk):

| role | kernel(s) | ms / chunk | work / chunk | achieved | reasonable target | target ms |
|---|---|---:|---|---:|---:|---:|
| sparse attention (12 layers + MTP, indexer budget 2048 keys) | `_chunks8` (+`_merge`,`_scores`,`_select`) | 1,640-2,650 | ~1.3 TFLOP (2048 q x 24 h x <=2048 keys x 256 x 4 x 13) | **0.5-0.8 TF** | 10 TF (WMMA, Triton tl.dot) | ~150 |
| hyper-connection mixes (bf16, K=10240/N=320 and back) | `_b16mm_ks`, `_hc_up_mix`, `_hc_wb_norm` | 810-1,010 | 2.6 TFLOP | **2.5-3.4 TF** | 20 TF | ~150 |
| block-FP8 dense (GDN in_proj, out_proj, attn q/k/v/o, shared) | `tf_fn_qmmf(_ld)<3,64,...>` | 580 | ~10 TFLOP | ~17 TF | 25-30 TF (rocm-next FP8 LDS tile: 29-35) | ~360 |
| int4 routed experts | `tf_int4_128_2_2_{2_2,1_3}` | 324 | 4.8 TFLOP | ~15 TF | 25 TF | ~200 |
| GDN conv/gates/chain | `tf_fn_gdn_io::front/back`, `gdn_prefill::chain` | 237 | memory/latency | | | ~120 |
| router + glue + rest | | ~100 | | | | ~80 |
| **total** | | **3,720-4,740** | | **~490 tok/s** | | **~1,060 => ~1,900 tok/s** |

So the 5.5x gap to Mia's 2.6k (and the ~1.3x gap to thorim's 635) is, in order: (1) `_chunks8` sparse attention at
<1 TF = 40-55% of prefill; (2) the bf16 hyper-connection mixes at ~3 TF = 22-25%; (3) FP8/int4 matmuls at 15-17 TF
= 23%. Fixing (1) and (2) alone takes a chunk from ~3.7 s to ~1.6 s (~1,280 tok/s); all of it to ~1.06 s (~1,900).
