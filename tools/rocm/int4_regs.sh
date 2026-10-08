#!/usr/bin/env bash
# Builds fn_int4.hip to ~/.cache/tf-int4 and prints each entry point's resource use (in the dev image).
set -euo pipefail
src=$(realpath "$(dirname "$0")/../../zig/kernels/hip/fn_int4.hip")
out=~/.cache/tf-int4
mkdir -p "$out"
cd "$out"
rm -f fn_int4-hip-amdgcn-amd-amdhsa-gfx1151.s
hipcc --genco --offload-arch=gfx1151 -O3 -std=c++20 -save-temps=obj -o fn_int4.co "$src" 2>&1 | grep -A3 error || true
awk '/^\s+\.name:/{n=$2} /^\s+\.private_segment_fixed_size:/{p=$2} /^\s+\.sgpr_count:/{s=$2}
     /^\s+\.vgpr_count:/{v=$2} /^\s+\.vgpr_spill_count:/{print n, "vgpr", v, "sgpr", s, "scratch", p, "spill", $2}' \
  fn_int4-hip-amdgcn-amd-amdhsa-gfx1151.s | sort
