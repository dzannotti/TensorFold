"""python3 tools/strix/test_quality.py: quality.score against hand-computed KL, top-1 and NLL."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality  # noqa: E402

ref = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
cand = np.log(np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])) + 3.0  # an offset logsumexp must remove
ids = np.array([0, 1])  # row 0 predicts token 1
s = quality.score(cand, ref, ids)
p, q = np.exp(ref), np.exp(cand - 3.0)
assert abs(s["kl"] - (p * np.log(p / q)).sum()) < 1e-5, s
assert s["top1"] == 2 and s["rows"] == 2 and s["targets"] == 1
assert abs(s["nll"] + np.log(0.4)) < 1e-5 and abs(s["ref_nll"] + np.log(0.2)) < 1e-5
r = quality.summary([s])
assert abs(r["ppl"] - 1 / 0.4) < 1e-4 and r["top1"] == 100.0
print("ok")
