#!/bin/bash
# usage (on xt): mb.sh FILE.cxx ROOT OUT [import flags]: compiles this dir on mimiron, fetches out/OUT.{hsaco,report.txt}, ISA dump
set -e
cd ~/tensorfold/.local/wmma-pipe/k4abl
S=$1; R=$2; N=$3; shift 3
ssh 192.168.1.3 mkdir -p tf-strix-remote/wmma-pipe/k4abl/out
rsync -a --exclude out ./ 192.168.1.3:tf-strix-remote/wmma-pipe/k4abl/
ssh 192.168.1.3 "cd ~/tf-strix-remote/wmma-pipe/k4abl && V=~/tf-strix-remote/.venv-hrx/bin && \$V/loom-import-cxx --I=. $* $S --output=out/$N.loom && \
  ~/tf-strix/bin/gpulock work wp-$N -- \$V/loom-compile out/$N.loom --root=@$R --target=amdgpu:gfx1151 --format=amdgpu-hsaco \
  --output=out/$N.hsaco --compile-report=text-summary --compile-report-output=out/$N.report.txt"
scp -q 192.168.1.3:tf-strix-remote/wmma-pipe/k4abl/out/$N.{hsaco,report.txt} out/
/opt/rocm/core-10.1/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1151 out/$N.hsaco | sed 's|//.*||' > out/$N.s
echo "$N wmma=$(grep -c v_wmma out/$N.s) ds_load=$(grep -c ds_load out/$N.s) waits=$(grep -c s_waitcnt out/$N.s)" \
  $(grep -oE "vector_final_registers=[0-9]+|resident_subgroups_per_simd=[0-9]+|private_bytes=[0-9]+" out/$N.report.txt | sort -u | tr '\n' ' ')
