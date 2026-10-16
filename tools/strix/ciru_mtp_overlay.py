#!/usr/bin/env python3
# Usage: python3 -I tools/strix/ciru_mtp_overlay.py OUT_DIR, then serve OUT_DIR (MiniCPM ciru_q8 + Qwen3.6 MLX MTP head as drafter).
"""Builds a model dir of symlinks: the ciru checkpoint plus Qwen3.6-35B-A3B's MLX MTP head as a side file, its config
naming the head's MLX widths (4-bit group 64, the router and shared gate 8-bit)."""
import json, os, sys
src = '/home/dzannotti/models/minicpm/MiniCPM-4.7-Ultra-CIRU-HALO-AGENT/weights'
mtp = '/home/dzannotti/models/q36/Qwen3.6-35B-A3B-MLX-4bit-MTP/mtp-4bit.safetensors'
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if f != 'config.json' and not os.path.exists(os.path.join(out, f)):
        os.symlink(os.path.join(src, f), os.path.join(out, f))
if not os.path.exists(os.path.join(out, 'mtp-4bit.safetensors')):
    os.symlink(mtp, os.path.join(out, 'mtp-4bit.safetensors'))
c = json.load(open(os.path.join(src, 'config.json')))
q = {'group_size': 64, 'bits': 4, 'mode': 'affine'}
for k in ('mlp.gate', 'mlp.shared_expert_gate'):
    q[f'language_model.mtp.layers.0.{k}'] = {'group_size': 64, 'bits': 8}
c['quantization'] = q
json.dump(c, open(os.path.join(out, 'config.json'), 'w'), indent=1)
print('ok', out)
