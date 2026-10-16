#!/usr/bin/env bash
# tools/strix/bench.sh <label> <model> <drafter|-> [extra serve args]
# Serves <model> (drafter "-": the checkpoint's MTP heads, or none), then runs exactness (alone vs 4 streams vs
# draft:false), bench_openai and sparkbench decode at LEVELS (default 1), and prefill at PREFILL_SIZES (default 8192,
# empty skips). Run it under the exclusive GPU lock (tools/strix/gpu-lock.sh: `bench tools/strix/bench.sh ...`).
# Env: BIN (default zig-out/native-hip/bin/tensorfold-native), TF_ENV (a file to source first, e.g. the ROCm SDK's
# env.sh), SERVE_ENV (VAR=value pairs for the server), OUT (default ./bench-out), PORT (default 8099), RESERVE_GIB
# (host memory the server leaves free, default 24: the prompt cache takes the rest), DECODE_TYPES (default
# prose,code,prose5: prose5 is five prose prompts at C1, a steadier regression signal than the one hash-map reply).
set -u
label=$1 dir=$2 drafter=$3; shift 3
[ -n "${TF_ENV:-}" ] && source "$TF_ENV"
cd "$(dirname "$0")/../.."
bin=${BIN:-zig-out/native-hip/bin/tensorfold-native} port=${PORT:-8099} out=${OUT:-bench-out}/$label
mkdir -p "$(dirname "$out")"
draft=(); [ "$drafter" != - ] && draft=(--drafter "$drafter")
env TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE_GIB:-24} ${SERVE_ENV:-} timeout 3600 "$bin" serve "$dir" --name m --port $port --context 32768 --parallel 8 "${draft[@]}" "$@" > "$out.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 300); do curl -sf localhost:$port/v1/models >/dev/null && break; kill -0 $pid 2>/dev/null || { tail -5 "$out.log"; exit 1; }; sleep 2; done
grep -E 'drafter|mtp|serving|HIP gfx' "$out.log" | head -5
url=http://127.0.0.1:$port/v1
echo "--- exactness (alone vs 4 streams vs draft:false)"
python3 tools/bench_concurrent.py $url m --levels 1,4 --tokens 128 --reps 1 --alone --serial 2>&1 | tail -6
echo "--- bench_openai"
python3 tools/bench_openai.py $url m --tokens 64 --reps 3 --temperatures 1.0,0 --label "$label" > "$out-openai.jsonl"
python3 -c "import sys,json;[print(d['prompt'],d['temperature'],round(d['decode_tps_median'],1)) for d in map(json.loads,open(sys.argv[1]))]" "$out-openai.jsonl"
echo "--- sparkbench decode"
python3 tools/strix/sparkbench.py --base-url $url --model m --mode decode --concurrency ${LEVELS:-1} --decode-types ${DECODE_TYPES:-prose,code,prose5} --out "$out-decode.json" | grep -E 'C1|C8'
if [ -n "${PREFILL_SIZES-8192}" ]; then
  echo "--- sparkbench prefill"
  python3 tools/strix/sparkbench.py --base-url $url --model m --mode prefill --sizes ${PREFILL_SIZES:-8192} --out "$out-prefill.json" | grep prefill
fi
