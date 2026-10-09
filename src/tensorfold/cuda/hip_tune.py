"""gfx1151 launch options for Triton kernels whose CUDA (GB10) ones spill on 256 VGPRs, in one table: the ROCm
wrappers launch with them and tools/zig/flashnext_aot.py builds the HIP kernel set with them, so the JIT reference and
the Zig engine run the same configuration. Only warps, pipeline stages and tiles (`tile`: BM, BLOCK_N, BK) change,
and only where a kernel's bits do not depend on them (its K reduction is the dot's own MMA chain in BK order; no tl.sum over a warp-split axis): a
row's bits never depend on its tile or on M (tools/rocm/triton_parity.py run and tiles check it). Rows are chosen
spill-free in both pointer forms where one exists: gfx1151 code that spills masked global loads has been seen to
drop the mask on a partial row tile (wrong rows, faults; docs/rocm/notes/triton-aot.md)."""

from __future__ import annotations

# (kernel, constexprs it applies to, options): the first matching row wins
TABLE = (
    ("_router", {"BM": 16}, {"num_warps": 2, "num_stages": 1}),
    ("_router", {"BM": 128}, {"num_warps": 4, "num_stages": 1}),
    ("_router", {}, {"num_warps": 8, "num_stages": 2}),
    ("_b16mm", {"BM": 16, "BK": 256}, {"num_warps": 4, "num_stages": 1}),
    ("_b16mm", {"BM": 16}, {"num_warps": 2, "num_stages": 1}),
    ("_b16mm", {}, {"num_warps": 4, "num_stages": 1}),
    ("_b16mm_ks", {"BM": 64}, {"num_warps": 4, "num_stages": 1}),
    ("_b16mm_ks", {}, {"num_warps": 8, "num_stages": 1}),
    ("_hc_up_mix", {"BM": 32}, {"num_warps": 4, "num_stages": 1}),
    ("_hc_up_mix", {}, {"num_warps": 8, "num_stages": 1}),
)

def tile(name: str, consts: dict) -> dict | None:
    """The tile constexprs gfx1151 launches a specialization with instead, if any: the AOT set adds an entry with them
    beside the spec's (flashnext_aot.hip_entries) and the Zig wrappers launch them on HIP (cuda_prompt.zig ks_tile,
    cuda_triton.zig b16Tile). Tiles are free: an output's bits are its own MMA chain over K, whatever tile holds it."""

    if name == "_b16mm_ks":
        return {"BM": 64, "BLOCK_N": 128}
    if name == "_b16mm" and consts.get("BM") == 16 and (consts["K"] // consts["SK"]) % 256 == 0 \
            and consts["K"] // consts["SK"] >= 1024:
        return {"BK": 256}                        # decode rows, long K slices: 512-byte row runs a step
    return None


def options(name: str, consts: dict, opts: dict) -> dict:
    """`opts` (the CUDA launch's num_warps / num_stages) with this kernel's gfx1151 row applied."""

    for kernel, match, o in TABLE:
        if kernel == name and all(consts.get(k) == v for k, v in match.items()):
            return {**opts, **o}
    return opts


def hip() -> bool:
    import torch

    return torch.version.hip is not None


def launch(name: str, consts: dict, **opts) -> dict:
    """A wrapper's launch options: the table's on ROCm, `opts` on CUDA."""

    return options(name, consts, opts) if hip() else opts
