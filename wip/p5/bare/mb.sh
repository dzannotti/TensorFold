#!/bin/bash
# usage (on xt): mb.sh SRC.cxx ROOT OUT [import flags]: import + compile on mimiron (hrx-loom 0.3.0), fetch hsaco/report, ISA dump
set -e
cd ~/tensorfold/.local/wmma-pipe/bare
S=$1; R=$2; N=$3; shift 3
rsync -a --exclude out ./ 192.168.1.3:tf-strix-remote/wmma-pipe/bare/
ssh 192.168.1.3 "mkdir -p ~/tf-strix-remote/wmma-pipe/bare/out && cd ~/tf-strix-remote/wmma-pipe/bare && V=\${LOOMV:-~/tf-strix-remote/.venv-hrx/bin} && \
  \$V/loom-import-cxx --I=$(dirname $S) $* $S --output=out/$N.loom && \
  ~/tf-strix/bin/gpulock work wp-$N -- \$V/loom-compile out/$N.loom --root=@$R --target=amdgpu:gfx1151 --format=amdgpu-hsaco \
  --output=out/$N.hsaco --compile-report=text-summary --compile-report-output=out/$N.report.txt"
scp -q 192.168.1.3:tf-strix-remote/wmma-pipe/bare/out/$N.{hsaco,report.txt,loom} out/
/opt/rocm/core-10.1/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1151 out/$N.hsaco | sed 's|//.*||' > out/$N.s
echo "$N wmma=$(grep -c v_wmma out/$N.s) ds_load=$(grep -c ds_load out/$N.s) waits=$(grep -c s_waitcnt out/$N.s)" \
  $(grep -oE "vector_final_registers=[0-9]+|resident_subgroups_per_simd=[0-9]+|private[a-z_]*=[0-9]+" out/$N.report.txt | sort -u | tr '\n' ' ')
