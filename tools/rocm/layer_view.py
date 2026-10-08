#!/usr/bin/env python3
"""A 1-layer view of the Flash Next checkpoint for engine tests that must stay small (~2.8 GiB of weights; --mtp keeps the MTP layer):
the embedding, head and glue tensors plus one decoder layer renamed to layer 0, in new safetensors (~3.7 GB on disk);
everything else symlinked. Layer 0 is linear attention (DeltaNet); --layer 3 gives a full-attention one.

    python3 tools/rocm/layer_view.py MODEL_DIR OUT_DIR [--layer N]
    tensorfold run OUT_DIR --tokens ... --no-drafts        # tensorfold-native serve OUT_DIR ... --no-drafts
"""
import argparse, json, os, re, struct

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--mtp", action="store_true", help="keep the MTP layer (drafted runs)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.model):
    if f == "config.json" or f.startswith("model") or f == "ple-table" or os.path.exists(f"{a.out}/{f}"):
        continue
    os.symlink(f"{a.model}/{f}", f"{a.out}/{f}")
c = json.load(open(f"{a.model}/config.json")); t = c.get("text_config", c)
t["num_hidden_layers"], t["layer_types"], t["ple_layer_ids"] = 1, [t["layer_types"][a.layer]], []
json.dump(c, open(f"{a.out}/config.json", "w"), indent=1)
P = "model.language_model.layers."


def rename(k):
    if k.startswith("mtp"):
        return k if a.mtp else None
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
    return k if m is None else (P + "0." + k[m.end():] if int(m.group(1)) == a.layer else None)


idx = json.load(open(f"{a.model}/model.safetensors.index.json")); wm = {}
for f in sorted(set(idx["weight_map"].values())):
    with open(f"{a.model}/{f}", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n)); base = 8 + n
        names = [(k, rename(k)) for k in h if k != "__metadata__" and rename(k)]
        if not names:
            continue
        out, off = ({"__metadata__": h["__metadata__"]} if "__metadata__" in h else {}), 0
        for k, nk in names:
            s, e = h[k]["data_offsets"]; out[nk] = dict(h[k], data_offsets=[off, off + e - s]); off += e - s
        hb = json.dumps(out).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(f"{a.out}/{f}", "wb") as o:
            o.write(struct.pack("<Q", len(hb))); o.write(hb)
            for k, nk in names:
                s, e = h[k]["data_offsets"]; fh.seek(base + s); o.write(fh.read(e - s)); wm[nk] = f
idx["weight_map"] = wm
json.dump(idx, open(f"{a.out}/model.safetensors.index.json", "w"))
print(f"{a.out}: layer {a.layer} ({t['layer_types'][0]}) as layer 0, {len(wm)} tensors")
