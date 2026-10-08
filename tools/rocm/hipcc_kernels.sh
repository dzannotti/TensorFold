#!/usr/bin/env bash
# Compile zig/kernels/cuda sources for gfx1151 as the HIP build should: --genco code objects (default) or, with
# MODE=so, one shared library of the C launchers for the ctypes parity checks. Run inside the rocm-dev container.
# Usage: tools/rocm/hipcc_kernels.sh OUT_DIR file.cu...   (MODE=so: OUT_DIR/NAME.so from all files, NAME=ops)
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$1; shift
mkdir -p "$out"
# CUDA's -O3 --fmad=false --ftz=false: no contraction, denormals kept; the CUDA headers map through hip/
flags=(-x hip -O3 -std=c++20 -ffp-contract=off -fno-gpu-flush-denormals-to-zero --offload-arch=gfx1151
       -include hip_compat.cuh -I "$root/zig/kernels/cuda/hip" -I "$root/zig/kernels/cuda" -I "$root/zig/kernels/cuda/torch_ops")
if [ "${MODE:-genco}" = so ]; then
  nice hipcc "${flags[@]}" -shared -fPIC "$@" -o "$out/${NAME:-ops}.so"
else
  for f in "$@"; do nice hipcc "${flags[@]}" --genco "$f" -o "$out/$(basename "${f%.cu}").hsaco"; done
fi
