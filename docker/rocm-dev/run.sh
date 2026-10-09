#!/usr/bin/env bash
# Usage: run.sh                 -> interactive shell in the dev image (GPU access, as your user)
#        run.sh <cmd> [args...] -> run a command
# Mounts $HOME read-write (the repo must live under it) and $TFDEV_MODELS (default $HOME/models) read-only; the
# working directory is the repo this script is in. Overrides: TFDEV_IMAGE (default tensorfold-rocm:dev, built with
# `docker build -t tensorfold-rocm:dev docker/rocm-dev`), TFDEV_NAME, TFDEV_MODELS.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
image=${TFDEV_IMAGE:-tensorfold-rocm:dev}
name=${TFDEV_NAME:-tfdev-$$}
models=${TFDEV_MODELS:-$HOME/models}
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
mnt=(-v "$HOME:$HOME:rw"); [ -d "$models" ] && mnt+=(-v "$models:$models:ro")
[ $# -eq 0 ] && set -- bash
exec docker run --rm "${tty[@]}" --name "$name" \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  --label homelab.memwatch=stop \
  "${mnt[@]}" \
  -e HOME="$HOME" \
  --user "$(id -u):$(id -g)" \
  -w "$repo" \
  "$image" "$@"
