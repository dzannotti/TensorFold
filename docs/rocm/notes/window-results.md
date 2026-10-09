# Window 2026-10-09: first full-model results (runtime: stock ROCm 10.0 / HIP 7.15)

Build: rocm @ f3696cd, Triton set aot-hip2. Serve: --context 262144 --parallel 8 --kv-dtype fp8 --thinking, TF_FLASHNEXT_DEPTH=15. Load 76.6 s; sequence memory 22.74 GiB.

## Correctness

```
PASS      chunks chat-keeps                   cached 0 sha 72fa50a30ddb fresh 72fa50a30ddb
PASS      chunks chat-resumed                 cached 18103 sha 72fa50a30ddb fresh 72fa50a30ddb
contracts: UNCHECKED (54 pass, 0 fail, 6 unchecked)
overall: 3076/3101 positions agree (99.19%); 7/20 free-run replies identical
agreement: PASS (min 98%)
```

## Speed (tools/rocm/e2e/bench.py, Mia's method)

**Prefill** (one request)

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,116 | 276 tok/s | 14.89 s |
| 8k | 8,203 | 464 tok/s | 17.67 s |
| 16k | 16,396 | 479 tok/s | 34.22 s |
| 32k | 32,782 | 475 tok/s | 69.02 s |
| 64k | 65,557 | 443 tok/s | 147.86 s |
| 128k | 131,084 | 460 tok/s | 285.26 s |

**Decode, prose** (greedy, thinking off, 256 tokens, end-of-sequence ignored, median of 3)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 48.6 tok/s | 48.6 tok/s | 149 ms |
| 2 | 75.7 tok/s | 38.2 tok/s | 185 ms |
| 4 | 111.1 tok/s | 29.1 tok/s | 464 ms |
| 8 | 157.5 tok/s | 21.0 tok/s | 669 ms |

**Decode, code** (greedy, thinking off, 256 tokens, end-of-sequence ignored, median of 3)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 91.3 tok/s | 91.3 tok/s | 268 ms |
| 2 | 109.3 tok/s | 57.8 tok/s | 287 ms |
| 4 | 149.8 tok/s | 41.2 tok/s | 580 ms |
| 8 | 199.6 tok/s | 28.1 tok/s | 934 ms |

Mia GB10: prefill 2,526/2,606/2,643/2,630/2,564/2,415 tok/s (4k..128k); prose 64.4/89.0/140.7/200.9 agg (1/2/4/8); code 57.5/92.7/130.8/189.3 agg.
