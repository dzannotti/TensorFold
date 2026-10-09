# Serving Qwen3.8 Flash Next on Strix Halo (gfx1151)

Branch `rocm`. Results and the reasoning behind every default: notes/window2.md. The quickest route is the container
(README Quickstart, and Deploy below); this page is the detail.

## Build (dev image)

The dev image (docker/rocm-dev, `docker build -t tensorfold-rocm:dev docker/rocm-dev`) is public sources only: Fedora 44
(digest-pinned) + AMD's signed ROCm 10.0.0-4 gfx1151 RPMs (HIP 7.15), Zig 0.17.0 (sha256-checked), AMD's gfx1151
wheels torch 2.11.0+rocm7.13.0 / triton 3.6.0+rocm7.13.0. `docker/rocm-dev/run.sh` runs it with GPU access.

```bash
docker/rocm-dev/run.sh sh -c 'nice zig build -j12 -Dgpu=hip -Dhipcc=/opt/rocm/bin/hipcc -Dkernel-set=SET native install'
```

The Triton set (SET) must match the tree: build it with `PYTHONPATH=src python -B tools/zig/flashnext_aot.py build
--target hip --spec zig/tests/cuda/flashnext/kernels.json --spec zig/tests/cuda/flashnext/kernels_int4ar.json --out SET`
(no GPU; notes/triton-aot.md). hipcc missing fn_qmm, fn_qmm_cluster, fn_qmm_prefill, fn_roce and qmm_group is
expected (unused on gfx1151; the serve image's build fails if that list changes).

## Serve

```bash
M=~/models/qwen38fn-int4-autoround      # docker/rocm-serve/fetch-model.sh
export TENSORFOLD_CUDA_KERNELS=SET TF_FLASHNEXT_DEPTH=15
# optional, +5-7% single-stream decode: the retained-PM4 HIP runtime (pwilkin/rocm-systems ilintar-experiments
# 7dda3ac6cf on ROCm 10.1; the serve image's pm4 stage builds it, or copy /opt/strix/lib + /opt/rocm/core-10.1 out of it)
export LD_LIBRARY_PATH=PM4_LIBS:CORE_10_1_LIBS DEBUG_HIP_GRAPH_PM4=1 GPU_MAX_HW_QUEUES=1
tools/rocm/memwatch.sh ./zig-out/native/bin/tensorfold-native serve $M --host 127.0.0.1 --port 8088 \
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

## Deploy (container)

`docker/rocm-serve/`: a multi-stage Dockerfile and a compose.yaml. No GPU is needed to build.

- build stage (the dev image): the Triton AOT set from both spec files and the engine with its HIP code objects.
- pm4 stage (`PM4=1`, the default): pwilkin/rocm-systems' ROCr + HIP (ilintar-experiments, pinned 7dda3ac6cf) built
  from source on AMD's ROCm 10.1.0-3 RPMs, as the halo-box llama recipe builds it. `--build-arg PM4=0` skips it.
- runtime: Fedora 44 with only the binary, the set, the ldd closures of the stock ROCm 10.0 libamdhip64/libhiprtc
  (comgr, LLVM, rocm_sysdeps) and, with PM4, of the PM4 libamdhip64 (its ROCr, core-10.1 comgr/LLVM), plus the
  license notices (/opt/tensorfold/share/doc). ~1.1 GB (0.8 GB with PM4=0).

```bash
docker build -t tensorfold-rocm:dev docker/rocm-dev            # once: the build stage's toolchain
docker build -f docker/rocm-serve/Dockerfile -t tensorfold-rocm:serve .   # JOBS=8 niced; --build-arg PM4=0: no PM4
docker run --rm tensorfold-rocm:serve                          # capabilities --json (no GPU needed)
docker/rocm-serve/start.sh ~/models/qwen38fn-int4-autoround    # checks, writes .env, compose up -d, waits for /health
```

- start.sh refuses to start without a gfx1151 GPU, with a GTT limit under 100 GiB (`ttm.pages_limit`, 4 KiB pages:
  27262976 = 104 GiB, 31457280 = 120 GiB; plus `amd_iommu=off`, on the kernel command line) or under 90 GiB
  MemAvailable (`FORCE=1` skips the memory checks). It writes docker/rocm-serve/.env (env.example): MODEL_DIR and the
  host's video/render GIDs, which own /dev/kfd and /dev/dri and differ between distros (compose needs them numeric).
- Serves `qwen3.8-flash-next` on 127.0.0.1:8080 (host network) with `--context 262144 --parallel 8 --kv-dtype fp8
  --thinking --backend hip`; healthcheck on /health. memlock unlimited and a 64 MiB stack, as the GB10 deployment.
- The model mounts read-only: serving writes nothing (the prompt cache is in memory; disk spill only with
  TENSORFOLD_SPILL_DIR, unset). `--learn` (on-disk prompt states) would need a writable volume.
- Image env: TENSORFOLD_CUDA_KERNELS (the baked set), TF_FLASHNEXT_DEPTH=15, HF_HUB_OFFLINE=1,
  TENSORFOLD_NO_UPDATE_CHECK=1 and, PM4 images, the PM4 libs first on LD_LIBRARY_PATH, DEBUG_HIP_GRAPH_PM4=1,
  GPU_MAX_HW_QUEUES=1. Stock runtime from a PM4 image: `LD_LIBRARY_PATH: ""`, `DEBUG_HIP_GRAPH_PM4: "0"`,
  `GPU_MAX_HW_QUEUES: "4"` in compose (the engine then loads /opt/rocm/lib/libamdhip64.so.7).
- A tree change that alters any Triton kernel needs a rebuilt image (the set is built from the same tree).

Memory: ~66 GiB of weights in GTT + sequence memory (sized at load from min(MemAvailable, device free), less the
12 GiB reserve, TENSORFOLD_MEMORY_RESERVE_GIB) + 1 GiB n-gram row cache in host RAM, plus page cache for the mapped
48.7 GiB n-gram table. Other GPU users (another LLM server, ComfyUI) must be stopped first.
