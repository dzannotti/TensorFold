#!/bin/sh
# memwatch.sh CMD...: runs CMD and kills it (TERM, then KILL) if MemAvailable drops below TF_MEMWATCH_GIB (default
# 10) GiB; prints the low-water mark at exit. TERM/INT to this script stop CMD too. Run it inside the container
# (docker/rocm-dev/run.sh labels it homelab.memwatch=stop for the host's own watcher).
floor=$(( ${TF_MEMWATCH_GIB:-10} * 1048576 ))
"$@" & pid=$!
trap 'kill $pid; sleep 2; kill -9 $pid 2>/dev/null' TERM INT
low=999999999
while kill -0 $pid 2>/dev/null; do
  m=$(awk '/MemAvailable/{print $2}' /proc/meminfo); [ "$m" -lt "$low" ] && low=$m
  if [ "$m" -lt "$floor" ]; then
    echo "memwatch: MemAvailable $((m / 1024)) MiB below $((floor / 1024)) MiB: stopping $1" >&2
    kill $pid; sleep 3; kill -9 $pid 2>/dev/null
  fi
  sleep 0.5
done
wait $pid; rc=$?
echo "memwatch: low-water MemAvailable $((low / 1024)) MiB, exit $rc" >&2
exit $rc
