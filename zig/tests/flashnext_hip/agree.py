"""Teacher-forced top-1 agreement with greedy reference replies; usage: agree.py URL MODEL REFS.json OUT.json."""
import json
import sys
import urllib.request

URL, MODEL, REFS, OUT = sys.argv[1:5]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def piece(ids):
    return post("/v1/detokenize", {"model": MODEL, "tokens": ids})["prompt"]


refs = json.load(open(REFS))
rows, total, agree = [], 0, 0
for row in refs["rows"]:
    run = row["runs"][0]
    body = row["body"]
    prompt = post("/v1/tokenize", {"model": MODEL, "messages": body["messages"],
                                   "chat_template_kwargs": body.get("chat_template_kwargs", {})})["tokens"]
    ref = post("/v1/tokenize", {"model": MODEL, "prompt": run["content"], "add_special_tokens": False})["tokens"]
    strs = [piece([t]) for t in ref]
    j, misses = 0, []
    while j < len(ref):
        doc = post("/v1/completions", {"model": MODEL, "prompt": prompt + ref[:j], "max_tokens": len(ref) - j,
                                       "temperature": 0, "draft": False})
        text = doc["choices"][0]["text"]
        got = []
        if text:
            got = post("/v1/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
        same = 0
        while same < len(got) and j + same < len(ref) and got[same] == ref[j + same]:
            same += 1
        if j + same < len(ref):
            misses.append({"pos": j + same, "want": strs[j + same],
                           "got": piece(got[same:same + 1]) if same < len(got) else None,
                           "context": piece(ref[max(0, j + same - 8):j + same])})
            same += 1
        moved = same
        j += moved
    n = len(ref)
    total += n
    agree += n - len(misses)
    rows.append({"name": row["name"], "prompt_tokens": len(prompt), "ref_tokens": n, "disagree": misses})
    print(f"{row['name']}: {n - len(misses)}/{n} agree; misses " +
          ", ".join(f"@{m['pos']} {m['want']!r}->{m['got']!r}" for m in misses), flush=True)
print(f"TOTAL {agree}/{total} = {100 * agree / max(total, 1):.2f}% teacher-forced top-1 agreement")
json.dump({"refs": REFS, "agree": agree, "total": total, "rows": rows}, open(OUT, "w"), indent=1)
