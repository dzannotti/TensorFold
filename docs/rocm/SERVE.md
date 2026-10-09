# Serving Qwen3.8 Flash Next on Strix Halo (gfx1151)

Branch `rocm` (a2f8e7e or later). Results and the reasoning behind every default: notes/window2.md.

## Build (dev image)

```bash
cd /home/dzannotti/tf-rocm && docker/rocm-dev/run.sh sh -c 'nice zig build -j12 -Dgpu=hip -Dhipcc=/opt/rocm/bin/hipcc \
  -Dkernel-set=/home/dzannotti/tf-triton-scratch/aot-final native install'
```

The Triton set must match the tree (`aot-final` for 5074bde..a2f8e7e; rebuild with tools/zig/flashnext_aot.py,
notes/triton-aot.md). hipcc missing qmm_group, prefill_attention, qmm_prefill, experts_prefill, fn_qmm,
fn_qmm_prefill, fn_qmm_cluster and fn_roce is expected.

## Serve

```bash
M=/home/dzannotti/models/qwen38fn-int4-autoround
export TENSORFOLD_CUDA_KERNELS=/home/dzannotti/tf-triton-scratch/aot-final TF_FLASHNEXT_DEPTH=15
# optional, +7% single-stream decode: the retained-PM4 HIP runtime (pwilkin/rocm-systems ilintar-experiments)
export LD_LIBRARY_PATH=/home/dzannotti/tf-rt-libs/pm4:/home/dzannotti/tf-rt-libs/core-10.1 DEBUG_HIP_GRAPH_PM4=1 GPU_MAX_HW_QUEUES=1
tools/rocm/memwatch.sh ./zig-out/native/bin/tensorfold-native serve $M --host 0.0.0.0 --port 8088 \
  --context 262144 --parallel 8 --kv-dtype fp8 --thinking --backend hip --no-update-check
```

- Weights take ~66 GiB of GTT; the engine keeps 12 GiB free (TENSORFOLD_MEMORY_RESERVE_GIB) and sizes sequence
  memory from min(MemAvailable, device free). Load ~50 s from NVMe; the first 8k prompt after load is ~2x slower.
- Draft policy (HIP defaults in code): depth 15, running product >= 0.1 at every stream count. Overrides:
  TF_FLASHNEXT_CONFIDENCE (-1..1), TF_FLASHNEXT_PRODUCT_STREAMS, TF_FLASHNEXT_DEPTH.
- Under the PM4 runtime (DEBUG_HIP_GRAPH_PM4 set) the shared expert runs in line (its forked stream hangs retained
  PM4); TF_FLASHNEXT_SHARED_SIDE=1/0 forces either way. Stock runtime: side stream on. The PM4 runtime is out of
  tree: keep the two lib dirs first on LD_LIBRARY_PATH and never install them system-wide.
- Profiling: `rocprofv3 --kernel-trace` works in graph mode only with TF_FLASHNEXT_SHARED_SIDE=0.
- Checks after a change: `tools/rocm/e2e/contracts.py` (0 FAIL; 6 resume cases are UNCHECKED by design),
  `agreement.py` vs ref/thorim.json (~99%), `bench.py` (Mia's method).
