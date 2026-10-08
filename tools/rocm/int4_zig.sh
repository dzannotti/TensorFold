#!/usr/bin/env bash
# cuda_int4.zig's host tests at both layouts, and its HIP launch paths type-checked against a stand-in cuda module
# (the HIP build is wired elsewhere). Run inside the dev image.
set -euo pipefail
here=$(realpath "$(dirname "$0")")
cd "$here/../../zig/src/families/flashnext"
printf 'pub const is_hip = false;\n' > /tmp/int4_nohip.zig
zig test --dep cuda -Mroot=cuda_int4.zig -Mcuda=/tmp/int4_nohip.zig
zig test --dep cuda -Mroot=cuda_int4.zig -Mcuda="$here/int4_cuda_stub.zig"
zig test --dep int4 -Mroot="$here/int4_typecheck.zig" --dep cuda -Mint4=cuda_int4.zig -Mcuda="$here/int4_cuda_stub.zig"
