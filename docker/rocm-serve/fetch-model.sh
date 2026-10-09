#!/usr/bin/env bash
# fetch-model.sh DIR : downloads the checkpoint the port is qualified with into DIR (resumable; re-run to finish).
#   azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound @1464274120d3: ~122 GB on disk (66 GiB weights, the 48.7 GiB
#   n-gram table, MTP). Put it on a fast NVMe: the engine maps it, and a load reads ~70 GB of it.
# Serving it needs a 128 GB Strix Halo: ~66 GiB of weights in GTT + sequence memory (GTT limit >= 100 GiB, ~90 GiB
# MemAvailable at start: docker/rocm-serve/start.sh checks both) plus page cache for the mapped n-gram table.
# Needs the Hugging Face CLI (`hf`): pip install --user -U huggingface_hub (or: uv tool install huggingface_hub).
set -euo pipefail
repo=azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound
rev=1464274120d36a4d8fcaa934552334a7d83ce0fd
need_gib=122
[ $# -eq 1 ] || { echo "usage: $0 DIR" >&2; exit 2; }
dir=$1
command -v hf >/dev/null || { echo "hf not found: pip install --user -U huggingface_hub" >&2; exit 1; }
mkdir -p "$dir"
have=$(du -s --block-size=1G "$dir" | cut -f1)
free=$(df --output=avail --block-size=1G "$dir" | tail -1 | tr -d ' ')
if ((free + have < need_gib + 2)); then
    echo "$dir: ${free} GiB free + ${have} GiB downloaded < ${need_gib} GiB needed" >&2; exit 1
fi
hf download "$repo" --revision "$rev" --local-dir "$dir"
echo "model ready in $dir (MODEL_DIR=$(realpath "$dir"))"
