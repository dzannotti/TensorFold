#!/bin/bash
# usage (on xt): xr.sh: bare-loop variants on xt (ROCm 10.1) under one gpulock bench
cd ~/tensorfold/.local/wmma-pipe/bare
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
L=/opt/rocm/core-10.1/lib/libamdhip64.so
for v in "b0 16" "nf6 12" "st8 16" "st6 12" "c2 16" "c2st 16"; do set -- $v
  ./host/run $L out/$1.hsaco bare_loop 2560 320 $2 1 | grep BEST
done
./host/run $L out/b0.hsaco bare_loop 2560 320 16 0 | grep BEST
