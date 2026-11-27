#!/bin/bash
# usage (on xt): mr.sh MODE gu|dn OUTNAME KERNEL TOKENS ROUTING check|bench: k4v2's harness + data on mimiron, this dir's hsaco
set -e
cd ~/tensorfold/.local/wmma-pipe/k4abl
M=$1; W=$2; N=$3; shift 3
scp -q out/$N.hsaco 192.168.1.3:tf-strix-remote/wmma-pipe/k4abl/out/
ssh 192.168.1.3 "cd ~/tf-strix-remote/k4v2 && source ./hrx-env.sh && export K4_LOOM_ROWS=128 K4_SKIP_REF=1 && \
  ~/tf-strix/bin/gpulock $M wp-$N -- ./host/k4h \$TENSORFOLD_HIP ref/refk.co data/l20 $W ~/tf-strix-remote/wmma-pipe/k4abl/out/$N.hsaco $*" 2>&1 \
  | grep -E "^BENCH|^CHECK|FAIL|error"
