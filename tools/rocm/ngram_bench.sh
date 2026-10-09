#!/bin/sh
# ngram_bench.sh [TOKENS] [--drop] [bench args...]: the host n-gram gather's stall a decode window (no GPU), direct
# mmap vs the row cache, cold pass then warm pass, over a recorded token stream (default: the 8k perf prompt).
# --drop drops ple-table's pages first (POSIX_FADV_DONTNEED) for a cold page cache: only in a test window, never while
# a server relies on them. Runs in the dev image without GPU devices, niced, two CPUs.
set -eu
W=$(cd "$(dirname "$0")/../.." && pwd)
M=${MODEL:-/home/dzannotti/models/qwen38fn-int4-autoround}
T=${1:-/home/dzannotti/tf-window/perf/p8k.ids}; [ $# -gt 0 ] && shift
run() { docker run --rm --cpus 2 -v /home/dzannotti:/home/dzannotti:rw -v /home/dzannotti/models:/home/dzannotti/models:ro \
  -e HOME=/home/dzannotti --user "$(id -u):$(id -g)" -w "$W" tf-rocm-dev:latest nice -n 19 "$@"; }
run zig build flashnext-ngram-bench -Dgpu=hip -j2
for win in 1 6 16; do run zig-out/bin/tf-flashnext-ngram-bench "$M" "$T" --window $win --passes 2 "$@"; done
