#!/bin/bash
# usage: build.sh SRC.cxx ROOT OUT [import flags]: import + compile on xt (hrx-loom 0.3.0), ISA dump, resources
set -e
cd ~/tensorfold/.local/wmma-pipe/bare
V=${LOOMV:-~/tensorfold/.local/loom-venv/bin}
S=$1; R=$2; N=$3; shift 3
$V/loom-import-cxx --I=$(dirname $S) "$@" $S --output=out/$N.loom
~/tf-strix/bin/gpulock work wp-$N -- $V/loom-compile out/$N.loom --root=@$R --target=amdgpu:gfx1151 --format=amdgpu-hsaco \
  --output=out/$N.hsaco --compile-report=text-summary --compile-report-output=out/$N.report.txt
/opt/rocm/core-10.1/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1151 out/$N.hsaco | sed 's|//.*||' > out/$N.s
echo "$N wmma=$(grep -c v_wmma out/$N.s) ds_load=$(grep -c ds_load out/$N.s) waits=$(grep -c s_waitcnt out/$N.s)" \
  $(grep -oE "vector_final_registers=[0-9]+|resident_subgroups_per_simd=[0-9]+|private[a-z_]*=[0-9]+" out/$N.report.txt | sort -u | tr '\n' ' ')
