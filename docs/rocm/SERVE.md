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

## Deploy (container, in ornith's place)

`docker/rocm-serve/`: a multi-stage Dockerfile (engine + HIP code objects + the Triton AOT set built in the dev
image, then a Fedora 44 runtime with only the binary, the set, the stock ROCm 10.0 HIP runtime, the retained-PM4
runtime + its core-10.1 libs, and the license notices) and a compose.yaml with ornith's conventions (host network,
kfd/dri, groups 44/991, `homelab.memwatch: stop`, deunhealth, `/health` healthcheck). No GPU is needed to build.

```bash
docker build -t tf-rocm-dev:latest docker/rocm-dev        # once: the build stage's base
docker compose -f docker/rocm-serve/compose.yaml build    # ~JOBS=8 CPU jobs, niced
docker run --rm mimiron/tensorfold:rocm-0e40c85           # capabilities --json (no GPU needed)
docker compose -f docker/rocm-serve/compose.yaml up -d    # serves 127.0.0.1:10001 (needs ornith stopped)
```

- Serves `qwen3.8-flash-next` with alias `ornith-1.5-35b`, so ornith's clients keep working on port 10001.
- The model mounts read-only at /models/qwen38fn-int4-autoround: serving writes nothing into it (disk spill only
  with TENSORFOLD_SPILL_DIR, unset here).
- Image env: TENSORFOLD_CUDA_KERNELS (the baked set), TF_FLASHNEXT_DEPTH=15, PM4 libs first on LD_LIBRARY_PATH,
  DEBUG_HIP_GRAPH_PM4=1, GPU_MAX_HW_QUEUES=1, HF_HUB_OFFLINE=1. Stock runtime instead: set `LD_LIBRARY_PATH: ""` and
  `DEBUG_HIP_GRAPH_PM4: ""` in compose (the engine then loads /opt/rocm/lib/libamdhip64.so.7).
- A tree change that alters any Triton kernel needs a rebuilt image (the set is built from the same tree).

Memory: ~66 GiB of weights in GTT + sequence memory (sized at load from min(MemAvailable, device free), less the
12 GiB reserve, TENSORFOLD_MEMORY_RESERVE_GIB) + 1 GiB n-gram row cache in host RAM, plus page cache for the mapped
48.7 GiB n-gram table. It cannot co-run with ornith or ComfyUI; embedding/reranker should stay small.

Swap in (ornith's compose lives in /homelab/hosts/mimiron/models/ornith):

```bash
docker compose -f /homelab/hosts/mimiron/models/ornith/compose.yaml stop ornith   # frees ~80 GiB with ComfyUI
docker stop comfyui                                                              # if running
docker compose -f docker/rocm-serve/compose.yaml up -d
docker logs -f tensorfold          # load ~50 s; healthy once /health answers
curl -s 127.0.0.1:10001/v1/models
```

Rollback: `docker compose -f docker/rocm-serve/compose.yaml down`, then
`docker compose -f /homelab/hosts/mimiron/models/ornith/compose.yaml up -d` (and start ComfyUI again). Nothing on the
host changes, so rollback is just stopping one container and starting the other.
