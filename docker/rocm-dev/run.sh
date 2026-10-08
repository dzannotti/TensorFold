#!/usr/bin/env bash
# Usage: run.sh                 -> interactive shell
#        run.sh <cmd> [args...] -> run a command
# Image/name override: TFDEV_IMAGE=..., TFDEV_NAME=...
set -euo pipefail
image=${TFDEV_IMAGE:-tf-rocm-dev:latest}
name=${TFDEV_NAME:-tfdev-$$}
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
[ $# -eq 0 ] && set -- bash
exec docker run --rm "${tty[@]}" --name "$name" \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  --label homelab.memwatch=stop \
  -v /home/dzannotti:/home/dzannotti:rw \
  -v /home/dzannotti/models:/home/dzannotti/models:ro \
  -e HOME=/home/dzannotti \
  --user "$(id -u):$(id -g)" \
  -w /home/dzannotti/tf-rocm \
  "$image" "$@"
