"""The stdlib client the e2e tools share: JSON posts, a timed SSE chat stream, rendered prompt ids, token-sha lookup.

The server is TENSORFOLD_URL (default http://127.0.0.1:8088) or a tool's --url. The model id is the first of /v1/models.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("TENSORFOLD_URL", "http://127.0.0.1:8088").rstrip("/")
EOS = (248046, 248044)          # <|im_end|>, <|endoftext|>: a stopped reply's token_sha counts the end token
_model = None


class ApiError(Exception):
    pass


def set_url(url: str) -> None:
    global URL
    URL = url.rstrip("/").removesuffix("/v1")


def post(path: str, body: dict, timeout: float = 3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise ApiError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    except (urllib.error.URLError, OSError) as e:
        raise ApiError(f"{path}: cannot reach {URL} ({getattr(e, 'reason', e)})") from None


def post_json(path: str, body: dict, timeout: float = 3600) -> dict:
    with post(path, body, timeout) as r:
        return json.load(r)


def get_json(path: str, timeout: float = 30) -> dict:
    try:
        with urllib.request.urlopen(URL + path, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError) as e:
        raise ApiError(f"{path}: cannot reach {URL} ({getattr(e, 'reason', e)})") from None


def model() -> str:
    global _model
    if _model is None:
        _model = get_json("/v1/models")["data"][0]["id"]
    return _model


def sha(ids) -> str:
    """The server's token_sha: sha256 of the comma-joined decimal ids, 12 hex."""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


_by_sha = None


def id_of(token_sha: str, vocab: int = 300_000) -> int | None:
    """The token id of a one-token reply, from its token_sha."""
    global _by_sha
    if _by_sha is None:
        _by_sha = {sha([i]): i for i in range(vocab)}
    return _by_sha.get(token_sha)


def render(messages, tools=None) -> list:
    """The ids the chat route runs for these messages, thinking off."""
    body = {"model": model(), "messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    return post_json("/tokenize", body, 600)["tokens"]


def tokenize(text: str) -> list:
    return post_json("/tokenize", {"model": model(), "prompt": text, "add_special_tokens": False}, 600)["tokens"]


def reply_ids(text: str, token_sha: str, finish: str) -> list | None:
    """A reply's token ids: its text re-tokenized (plus the end token if it stopped), kept only when the sha agrees."""
    ids = tokenize(text) if text else []
    for tail in ([[]] if finish != "stop" else [[e] for e in EOS] + [[]]):
        if sha(ids + tail) == token_sha:
            return ids + tail
    return None


def chat(messages, max_tokens=256, temperature=0.0, **extra) -> dict:
    body = {"model": model(), "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    return post_json("/v1/chat/completions", body)


def complete(ids, max_tokens=256, temperature=0.0, **extra) -> dict:
    """POST /v1/completions with a token-id prompt: run as given, keeps no prompt state."""
    body = {"model": model(), "prompt": list(ids), "max_tokens": max_tokens, "temperature": temperature, **extra}
    return post_json("/v1/completions", body)


def summary(r: dict) -> dict:
    """token_sha, finish, cached, drafts, accepted and the text of a chat or completion reply."""
    c = r["choices"][0]
    msg = c.get("message") or {}
    calls = "".join(f"{t['function']['name']}({t['function']['arguments']})" for t in msg.get("tool_calls") or [])
    tf, u = r.get("tensorfold") or {}, r.get("usage") or {}
    return {"sha": tf.get("token_sha"), "finish": c.get("finish_reason"), "tokens": u.get("completion_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "drafts": tf.get("drafts"),
            "accepted": (r.get("speculative") or {}).get("accepted"),
            "text": c.get("text") if "text" in c else (msg.get("content") or "") + calls}


def stream_chat(messages, max_tokens=256, temperature=0.0, **extra) -> dict:
    """A streamed chat request timed on the client: t0, first (first text), last (last text), end, usage, tensorfold."""
    body = {"model": model(), "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    out = {"t0": time.perf_counter(), "first": None, "last": None, "usage": {}, "tensorfold": {}, "pieces": 0}
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise ApiError(f"stream error: {chunk['error']}")
            for k in ("usage", "tensorfold", "speculative"):
                if isinstance(chunk.get(k), dict):
                    out[k] = chunk[k]
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter()
                out["first"] = out["first"] or now
                out["last"] = now
                out["pieces"] += 1
    out["end"] = time.perf_counter()
    return out
