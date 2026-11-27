#!/bin/bash
# usage (on xt): mr.sh MODE OUTNAME KERNEL ROWS COLS M N K [threads]: runs out/OUTNAME.hsaco on mimiron (HRX)
set -e
cd ~/tensorfold/.local/wmma-pipe/dense
M=$1; N=$2; shift 2
rsync -a host/ 192.168.1.3:tf-strix-remote/wmma-pipe/dense/host/
scp -q out/$N.hsaco 192.168.1.3:tf-strix-remote/wmma-pipe/dense/out/
ssh 192.168.1.3 "cd ~/tf-strix-remote/wmma-pipe/dense && source ~/tf-strix-remote/k4v2/hrx-env.sh && \
  ~/tf-strix/bin/gpulock $M wp-$N -- ./host/dense \$TENSORFOLD_HIP out/$N.hsaco $*" 2>&1 | grep -E "BEST|launch|fail"
