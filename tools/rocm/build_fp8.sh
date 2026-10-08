#!/usr/bin/env bash
# Builds zig/kernels/hip/fn_qmmf{,_ld}.hip into gfx1151 code objects (default build/rocm/) for tools/rocm/fp8_check.py.
# Run inside the dev container (docker/rocm-dev/run.sh). Extra hipcc flags: TF_HIPFLAGS.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-$root/build/rocm}
mkdir -p "$out"
for f in fn_qmmf fn_qmmf_ld; do
  hipcc --genco --offload-arch=gfx1151 --no-gpu-bundle-output -O3 ${TF_HIPFLAGS:-} "$root/zig/kernels/hip/$f.hip" -o "$out/$f.co"
done
