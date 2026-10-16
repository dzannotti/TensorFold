# Source me. One GPU, one timing at a time:
#   bench <cmd...>   exclusive: waits for every build, test and bench, and blocks new ones (timing runs, profiles)
#   work  <cmd...>   shared: builds, GPU tests, bounded reference runs. Never wrap a long-lived server in it.
LOCK=${TF_GPU_LOCK:-$HOME/.cache/tensorfold-gpu.lock}
bench() { flock -x "$LOCK" "$@"; }
work()  { flock -s "$LOCK" "$@"; }
