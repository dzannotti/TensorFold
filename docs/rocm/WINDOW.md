# Test window runbook: Flash Next on gfx1151, first full-model load

Window opens 00:30 BST. Everything runs in `docker/rocm-dev/run.sh` containers (label `homelab.memwatch=stop`). Never
touch production containers, /srv or /homelab; if prod is not down at 00:30, wait or stop: do not load next to it.
Tree: branch `rocm` (this file's commit). Paths below: `R=/home/dzannotti/tf-rocm`,
`M=/home/dzannotti/models/qwen38fn-int4-autoround`, `K=/home/dzannotti/tf-triton-scratch/aot-hip2` (the buffer-ops set),
`O=/home/dzannotti/tf-window` (results; `mkdir -p $O`). Abort rule throughout: MemAvailable < 10 GiB -> stop our
container (`docker stop tf-window`), note the step, and go no further.

## 0. Memory watch (own terminal, whole window)

```bash
while sleep 2; do printf '%s ' "$(date +%T)"; awk '/MemAvailable/{printf "MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo;
  [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -lt 10485760 ] && docker stop tf-window; done | tee -a $O/mem.log
```

Every engine command below also runs under `tools/rocm/memwatch.sh` (stops it below 10 GiB, prints the low-water
mark). Expected: weights ~66 GiB on the GPU (the 1-layer view: 2.80 GiB for embed + head + 1 layer), plus scratch; the
engine keeps 12 GiB free and sizes sequence memory from min(MemAvailable, device free).

## 1. Build (before the window; only rebuild if the tree changed)

```bash
cd $R && docker/rocm-dev/run.sh sh -c 'nice zig build -j12 -Dgpu=hip -Dhipcc=/opt/rocm/bin/hipcc -Dkernel-set=/home/dzannotti/tf-triton-scratch/aot-hip2 native install'
```

Expected warnings: hipcc cannot build qmm_group, prefill_attention, qmm_prefill, experts_prefill, fn_qmm,
fn_qmm_prefill, fn_qmm_cluster, fn_roce (none on Flash Next's serving path; fn_int4 must NOT be in the list). Outputs:
`zig-out/bin/{tensorfold,tf-cuda-test}`, `zig-out/native/bin/tensorfold-native`,
`zig-out/native/share/tensorfold/cuda/gfx1151/` (the Triton set; TENSORFOLD_CUDA_KERNELS=$K is equivalent).
To rebuild the set: `PYTHONPATH=src python -B tools/zig/flashnext_aot.py build --target hip --spec
zig/tests/cuda/flashnext/kernels.json --spec zig/tests/cuda/flashnext/kernels_int4ar.json --out $K` (dev image; it must keep buffer ops, see notes/triton-aot.md).

## 2. Pre-flight (00:30, before any load; small GPU, ~10 min)

```bash
cd $R && docker/rocm-dev/run.sh sh -c "export TENSORFOLD_CUDA_KERNELS=$K
  grep MemAvailable /proc/meminfo; ./zig-out/bin/tf-cuda-test info      # need device free >= 85 GiB, MemAvailable >= 90
  for c in smoke graph occupancy vmm; do ./zig-out/bin/tf-cuda-test \$c | tail -1; done
  for c in int4-check qsa-check fp4-check experts-check glue-check; do ./zig-out/bin/tensorfold \$c $M 2>&1 | tail -1; done"
docker/rocm-dev/run.sh sh -c "nice zig build test -Dgpu=hip -Daot-set=$K"   # dry: every fixture launch has a variant
```

Expected: PASS everywhere (also `glue-check`, and `mm-check $M` if run). If device free < ~85 GiB, prod still holds
memory: stop here. The engine logs `largest Triton argument extent: ...` at load (the token embedding, 1.18 GiB);
`error: TritonSpanPast2GiB` means a Triton argument would reach past buffer ops' 2 GiB (lower --context or
TF_FLASHNEXT_PREFILL_ROWS).

## 3. Smallest first load: CLI, no MTP head (~61 GiB)

The reference reply is thorim's greedy `prose-sky` (tools/rocm/e2e/ref/thorim.json).

```bash
python3 -c "import json;r=[x for x in json.load(open('$R/tools/rocm/e2e/ref/thorim.json'))['items'] if x['name']=='prose-sky'][0]
open('$O/sky.ids','w').write(','.join(map(str,r['prompt_ids'])));open('$O/sky.ref','w').write(','.join(map(str,r['reply_ids'])))"
cd $R && TFDEV_NAME=tf-window docker/rocm-dev/run.sh sh -c "TENSORFOLD_CUDA_KERNELS=$K tools/rocm/memwatch.sh \
  ./zig-out/bin/tensorfold run $M --tokens-file $O/sky.ids --max-tokens 102 --no-drafts --context 8192 --report $O/run-nodraft.json" 2>&1 | tee $O/run-nodraft.log
python3 -c "import json;a=json.load(open('$O/run-nodraft.json'))['tokens'];b=list(map(int,open('$O/sky.ref').read().split(',')))
n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)));print('agree with thorim for',n,'of',len(b),'tokens')"
```

Look for: `loaded in ...: N weight buffers, ~61 GiB`, the `sequence memory` line, no `error:`, low-water MemAvailable
>= 10 GiB. Not all-identical to thorim is expected (fp32 order differs), but a divergence in the first few tokens or
token 0 repeated means a broken kernel: stop and triage (section 7).

## 4. With the MTP head: drafted == plain (CLI)

```bash
cd $R && TFDEV_NAME=tf-window docker/rocm-dev/run.sh sh -c "TENSORFOLD_CUDA_KERNELS=$K TF_FLASHNEXT_DEPTH=15 tools/rocm/memwatch.sh \
  ./zig-out/bin/tensorfold run $M --tokens-file $O/sky.ids --max-tokens 102 --context 8192 --report $O/run-draft.json" 2>&1 | tee $O/run-draft.log
python3 -c "import json;a,b=(json.load(open('$O/run-%s.json'%k)) for k in ('nodraft','draft'));print('drafted == plain:',a['tokens']==b['tokens'],'accepted',b['accepted'],'of',b['drafted'], b['ms_per_token'],'ms/token')"
```

## 5. Serve with Mia's flags

```bash
cd $R && TFDEV_NAME=tf-window nohup docker/rocm-dev/run.sh sh -c "export TENSORFOLD_CUDA_KERNELS=$K TF_FLASHNEXT_DEPTH=15
  tools/rocm/memwatch.sh ./zig-out/native/bin/tensorfold-native serve $M --host 127.0.0.1 --port 8088 \
    --context 262144 --parallel 8 --kv-dtype fp8 --thinking --backend hip --no-update-check" > $O/serve.log 2>&1 &
until grep -q 'serving' $O/serve.log; do sleep 5; grep -m1 'error' $O/serve.log && break; done; tail -5 $O/serve.log
docker exec tf-window curl -s http://127.0.0.1:8088/v1/models
```

Clients run inside the same container (`docker exec tf-window ...`; the port is the container's own).

## 6. Contract checks, then speed (tools/rocm/e2e, README there)

```bash
E() { docker exec -w $R/tools/rocm/e2e tf-window python3 "$@"; }   # a function: zsh does not split $E
E contracts.py --url http://127.0.0.1:8088 --only a,b --cases prose-sky,tool-weather --concurrency 4 --max-tokens 96 | tee $O/contracts-quick.txt
E contracts.py --url http://127.0.0.1:8088 | tee $O/contracts.txt                          # a-d, ~60 requests
E agreement.py --url http://127.0.0.1:8088 --json $O/agreement.json | tee $O/agreement.txt    # vs ref/thorim.json
E bench.py --url http://127.0.0.1:8088 --label strix-rocm --json $O/bench.json | tee $O/bench.txt
```

Order matters: stop after any contract FAIL and triage (contracts are bit-equality; a FAIL is a bug, not noise).
Agreement: `--min-rate 0.98` sets the exit; fp8 KV alone costs ~1.2% top-1, so read the divergences. Bench: Mia's
GB10 numbers are 64 tok/s prose / 57.5 code at 1 request, 201 aggregate at 8, prefill ~2.5k tok/s.

## 7. Triage toggles (same bits by design, so A/B one at a time)

`TF_FLASHNEXT_PROMPT_MM=0`, `TF_FLASHNEXT_PROMPT_EXPERTS=0`, `TF_FLASHNEXT_FP4_SERIAL=0`, `TF_FLASHNEXT_WB_NORM=0`,
`TF_FLASHNEXT_SHARED_SIDE=0`, `TF_FLASHNEXT_GLUE_FUSE=0`, `TF_FLASHNEXT_FP8_LD=0`, `TF_FLASHNEXT_EXPERT_SHAPE=0`, `--eager` (CLI, no graphs),
`TF_FLASHNEXT_PREFILL_TAIL=0` (other chunking). For NaN or garbage: `TF_FLASHNEXT_DUMP_BINS=1 tensorfold prefill $M
P.json NAME --dump DIR --no-drafts` writes each layer's tensors as .bin (first NaN = the culprit). Reproduce small
on a 1-layer view: `python3 tools/rocm/layer_view.py $M OUT [--layer 3]`, then `run OUT ... --no-drafts`.

## 8. Close

`docker stop tf-window` (only ours), save `$O`, and write the results (load time, memory low-water, contracts,
agreement, bench) to docs/rocm/notes/window-results.md.
