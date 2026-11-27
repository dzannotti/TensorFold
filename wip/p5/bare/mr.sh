#!/bin/bash
# usage (on xt): mr.sh MODE OUTNAME KERNEL GROUPS ITERS WPI [fill]: runs out/OUTNAME.hsaco on mimiron (HRX) under gpulock MODE
set -e
cd ~/tensorfold/.local/wmma-pipe/bare
M=$1; N=$2; shift 2
rsync -a host/ 192.168.1.3:tf-strix-remote/wmma-pipe/bare/host/
scp -q out/$N.hsaco 192.168.1.3:tf-strix-remote/wmma-pipe/bare/out/
ssh 192.168.1.3 "cd ~/tf-strix-remote/wmma-pipe/bare && (test host/run -nt host/run.c || cc -O2 -o host/run host/run.c -ldl) && \
  source ~/tf-strix-remote/k4v2/hrx-env.sh && ~/tf-strix/bin/gpulock $M wp-$N -- ./host/run \$TENSORFOLD_HIP out/$N.hsaco $*" 2>&1 | grep -v "^gpulock: load\|other GPU\|^$"
