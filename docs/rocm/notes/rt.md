# HIP runtime layer (branch rocm-rt): measurements and probes

All on gfx1151 (Radeon 8060S), HIP 7.15.26333, inside tf-rocm-dev, with production sharing the GPU.

## Mangled names (zig/src/cuda/mangle.zig)
hipcc 7.15 `--genco --offload-arch=gfx1151` code objects (unbundled with clang-offload-bundler, read with llvm-nm):
- `__nv_bfloat16` -> `14__hip_bfloat16`, `__nv_bfloat162` -> `15__hip_bfloat162`, `__half`/`__half2` unchanged.
- `uint4` -> `15HIP_vector_typeIjLj4EE` (float2 -> `...IfLj2EE`, int4 -> `...IiLj4EE`): two substitution candidates
  where CUDA's struct has one, so every later `S<n>_` shifts; mangle.zig re-parses and recompresses.
- Test vectors are in mangle.zig; all 58 hard-coded names in zig/src translate without fallback.

## Device facts (hipDeviceGetAttribute / hipGetDeviceProperties)
- compute capability 11.5 (do not infer features from it; use `Context.features()`), gcnArchName gfx1151.
- multiprocessor_count 20 (rocminfo: 40 CUs; HIP counts WGPs), warp 32, max threads/MP 2048,
  shared/block opt-in 65536, VMM supported 1.

## VMM (tf-cuda-test vmm)
hipMemCreate/AddressReserve/Map/SetAccess work; granularity 4 KiB. One physical handle mapped at 4 addresses (the
indexer-key ring) aliases: copies and memset-kernel writes through one mapping read back through the others.
So cuda_vmm.zig stays enabled on HIP (no copy-on-grow fallback needed).

## hipMemGetInfo vs MemAvailable around a 1 GiB hipMalloc (tf-cuda-test meminfo)
```
before:          device free 38.663 GiB of 104.000, MemAvailable 45.479 GiB
1 GiB allocated: device free 37.663 GiB of 104.000, MemAvailable 44.475 GiB
1 GiB written:   device free 37.524 GiB of 104.000, MemAvailable 44.335 GiB
freed:           device free 38.524 GiB of 104.000, MemAvailable 45.339 GiB
```
hipMalloc commits at allocation (both counts drop by 1 GiB before any write). Device "total" is the 104 GiB GTT
limit and "free" is that limit minus GTT in use (prod holds ~65 GiB); MemAvailable is ~6.8 GiB higher (page cache,
non-GTT RAM). The engine's budget (cuda_engine.zig, MemAvailable first) therefore overshoots what hipMalloc can
still get by that gap; under HIP it should take min(MemAvailable, device free).

## Streaming read bandwidth (tf-cuda-test bandwidth 5; 1 GiB, uint4 grid-stride loads, best of 5)
Best 241.9 GB/s (decimal) at 64x320, 128x320 blocks; every shape from 128 threads x 80 blocks up is 239-242 GB/s;
64 threads x 80 blocks 198 GB/s. ~94% of the 256 GB/s LPDDR5X peak. GPU shared with prod: treat as a lower bound.

## Launch cost (tf-cuda-test overhead 1000 10)
plain stream 1.87 us/kernel GPU (0.81 us enqueue); one graph 1.82 us/kernel, graph launch 8.2 us, instantiate 1.6 ms;
exec node update 9.2 us/node.
