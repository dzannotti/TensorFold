# Window 2 (2026-10-09 06:00-10:05 BST, prod down, GPU exclusive): draft policy, re-profile, PM4 hang

Binary: rocm-cand 5074bde (`/home/dzannotti/tf-cand/zig-out`, same code as rocm 7ed5fc7), Triton set `aot-final`.
Raw logs: `~/tf-window/w2/` (scripts `sweep.sh`, `p2.sh`, `p3.sh`, `tools/`).

## 1. Draft policy sweep (no code changes)

`tensorfold bench-many $M --streams 8 --kind dash-prose,dash-code --rows 1,8 --reps 2 --max-tokens 256 --ignore-eos
--depth D --served --kv-dtype fp8`, one process a setting, rep 1 (rep 0 pays the captures). Aggregate decode tok/s;
in brackets tokens a stream-round and verify + drafting ms a round (for the first column). Default = depth 15,
TF_FLASHNEXT_CONFIDENCE=-0.4 (running product >= 0.4), TF_FLASHNEXT_PRODUCT_STREAMS=2 (above 2 live streams Python's
per-draft rule at the hard-coded `wide_confidence` 0.5). Two values = two runs (the very first run of the window,
d15's 56.8 / 217.6, was slow; its repeat and every other setting agree to <1%).

Sky (CLI `run`, 102 tokens, fp8 KV, `--depth D`): `--no-drafts` sha de8d7a445fe7; every drafted setting checked
(d4, d8, d15, c06, c02, c0, c0d6, p0c05, p0c02) gives the same sha de8d7a445fe7 (accepted/drafted in the last column).
Note: CLI `run` takes `--depth` (default 6), not TF_FLASHNEXT_DEPTH; earlier "accepted 51 of 69" sky runs were depth 6.

| setting | prose x1 | code x1 | prose x8 | code x8 | sky sha (acc/drafted) |
|---|---:|---:|---:|---:|---|
| d15 (default) | 59.0/56.8 (2.55 t/r, 38+5 ms) | 113.0/113.3 (6.71 t/r, 45+14 ms) | 226.4/217.6 (2.72) | 281.7/284.9 (5.28) | de8d7a445fe7 (51/69) |
| d4 | 56.9 (2.43) | 89.7 (4.40) | 226.4 (2.61) | 285.2 (4.10) | de8d7a445fe7 (51/69) |
| d6 | 58.9 (2.55) | 102.4 (5.80) | 225.6 (2.71) | 285.3 (4.68) | |
| d8 | 59.0/59.0 (2.55) | 111.2/111.3 (6.38) | 226.6/226.7 (2.71) | 294.1/294.2 (5.04) | de8d7a445fe7 (51/69) |
| d10 | 58.9 | 111.5 (6.54) | 227.0 | 290.1 (5.10) | |
| d12 | 58.8 | 112.7 (6.71) | 226.5 | 287.4 (5.26) | |
| c06 (product 0.6) | 56.3 (2.38) | 100.1 (5.43) | 226.5 | 281.5 | de8d7a445fe7 (49/66) |
| c05 | 57.4 (2.48) | 107.3 (6.07) | 226.3 | 280.5 | |
| c03 | 63.8 (2.93) | 111.2 (6.89) | 226.4 | 287.5 | |
| c025 | 65.4 (3.04) | 115.7 (7.29) | 226.4 | 286.9 | |
| c02 | 68.0 (3.19, 39+7 ms) | 117.3 (7.50) | 226.2 | 286.2 | de8d7a445fe7 (54/73) |
| **c01** | **71.5 (3.45, 40+8 ms)** | **119.3 (7.97, 48+18 ms)** | 226.2 | 288.8 | |
| c005 | 68.2 (3.49) | 114.4 (7.97) | 227.2 | 288.3 | |
| c0 (every draft, depth 15) | 47.8 (3.98, 55+28 ms) | 100.4 (8.50) | 150.4 (3.87) | 243.1 (7.16) | de8d7a445fe7 (58/592) |
| c0, depth 6 | 67.4 (3.81) | 102.7 (5.93) | 136.9 (3.72) | 309.1 (5.48) | de8d7a445fe7 (58/249) |
| PRODUCT_STREAMS=8 (product 0.4 at x8) | 58.9 | 113.4 | 223.0 (2.66) | 279.5 (5.09) | |
| PRODUCT_STREAMS=8, product 0.2 | 68.1 | 117.3 | 237.7 (3.07) | 290.2 (5.88) | |
| **PRODUCT_STREAMS=8, product 0.1** | **71.5** | **119.4** | **242.3 (3.26, 83+20 ms)** | **303.8 (6.28, 101+30 ms)** | |
| per-draft 0.7 everywhere (PS=0) | 57.1 (2.38) | 106.3 (5.80) | 216.3 (2.44) | 273.4 (4.52) | |
| per-draft 0.5 everywhere | 59.4 (2.63) | 113.3 (6.89) | 226.2 | 288.6 | de8d7a445fe7 (50/68) |
| per-draft 0.3 everywhere | 67.5 (3.27) | 111.1 (7.73) | 228.9 (3.14) | 280.4 (6.20) | |
| per-draft 0.2 everywhere | 66.8 (3.49) | 106.8 (8.23) | 223.6 (3.37) | 285.7 (6.58) | de8d7a445fe7 (55/84) |
| per-draft 0.1 everywhere | 68.0 (3.92) | 108.5 (8.23) | 212.2 (3.68) | 271.2 (6.89) | |

Reading:
- On gfx1151 a verify window's extra rows are nearly free (prose x1 verify 38 ms at 2.55 tokens a round, 40 ms at
  3.45: weights dominate), so the GB10-tuned product threshold 0.4 stops chains too early. Product >= 0.1 is the
  best at one stream: prose x1 59.0 -> 71.5 (+21%, above thorim's 65.2), code x1 113 -> 119.4 (+5.6%). 0.05 is past
  the optimum (drafting cost grows faster than acceptance), every-draft (c0) is far worse.
- At 8 streams the per-draft 0.5 rule (wide) is flat across depth; the running product at 0.1 also wins there:
  prose x8 226 -> 242 (+7%), code x8 282-294 -> 304 (+3-8%). Depth matters only for code (d15 >= d12 > d8 at x1;
  d8 slightly best at code x8 with the old rule), and d15 is best with the product 0.1 rule.
- **Recommended HIP defaults**: depth 15, running product 0.1 at every stream count (served_confidence -0.1 and
  product_streams >= --parallel). Today, as env: `TF_FLASHNEXT_DEPTH=15 TF_FLASHNEXT_CONFIDENCE=-0.1
  TF_FLASHNEXT_PRODUCT_STREAMS=8`. In code: a per-backend served_confidence / product_streams_default in cuda_engine.zig
  (HIP: -0.1 and maxInt). The engine already picks the rule by live stream count (lanes.confidenceNow: product while
  live <= product_streams, else wide_confidence); depth is engine-wide (graph sizes), so a per-stream-count depth
  would be a cap on the chain length in confidenceNow's caller (draft_stops_most), not a depth change. Not needed
  here: with product 0.1 depth 15 wins at x1 and x8. 2 and 4 streams were not measured (x1 and x8 both favour 0.1).
- The x8 samples are one run each; the d15/d8 repeats reproduced to 0.1%, so the steady CLI numbers are tight.
