"""HIP decode glue folded into fewer launches (the Zig engine's decode rounds on gfx1151, docs/rocm/notes/fuse.md):
each kernel runs the separate kernels' arithmetic in their order on the same layouts, so the same bits; only launches
(and the ~3 us idle between graph nodes each costs) go. The 32-group sums of normed / act are not made: only the 4-bit
MTP draft matrices read them, and the engine keeps the separate kernels then.

Nothing in the Python engine calls these; tools/zig/flashnext_aot.py (hip_entries) derives their HIP entries from the
kernels they replace."""

from __future__ import annotations

HAS_TRITON = True
try:
    import triton
    import triton.language as tl

    from .glue import _bsilu, _bsig

    @triton.jit
    def _hc_wbn(H, HOUT, PSS, BR, INJ, Y, WTS, RS, SCALE, NORMED, eps,
                D: tl.constexpr, S: tl.constexpr, MODE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
                BLOCK: tl.constexpr, WORLD: tl.constexpr):
        """``glue._hc_writeback`` then ``glue._hc_normed``, program (r, s): stream s of row r. The chunks are
        _hc_writeback's (same BLOCK and warps: the same squared sums, stored to PSS); rinv from them in chunk order
        from 0.0 as _hc_normed adds the stored ones; normed = bf16(h * rinv * scale) from the stored h. HOUT may
        differ from H (MODE != 0: every element is written)."""

        r = tl.program_id(0)
        s = tl.program_id(1)
        NC: tl.constexpr = D // BLOCK
        total = 0.0
        for c in tl.static_range(NC):
            d = c * BLOCK + tl.arange(0, BLOCK)
            if MODE == 1:
                branch = tl.load(BR + r * D + d).to(tl.float32)
            elif MODE == 3 or MODE == 4:
                acc = tl.load(BR + r * D + d)
                for k in tl.static_range(1, WORLD):
                    acc = acc + tl.load(BR + k * RS + r * D + d)
                branch = acc.to(tl.bfloat16).to(tl.float32)
            elif MODE == 2:
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for k in tl.static_range(TOPK + 1):
                    wk = tl.load(WTS + r * SLOTS + k)
                    yk = tl.load(Y + (r * SLOTS + k) * D + d).to(tl.float32)
                    acc = acc + yk * wk
                branch = acc.to(tl.bfloat16).to(tl.float32)
            hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
            if MODE != 0:
                inj = tl.load(INJ + r * S + s).to(tl.float32)
                hv = (hv + (branch * inj).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
                tl.store(HOUT + r * (S * D) + s * D + d, hv.to(tl.bfloat16))
            q = tl.sum(hv * hv, axis=0)
            tl.store(PSS + (r * NC + c) * S + s, q)
            total += q
        tl.debug_barrier()
        rinv = 1.0 / tl.sqrt(total / D + eps)
        for c in tl.static_range(NC):
            d = c * BLOCK + tl.arange(0, BLOCK)
            if MODE != 0:
                hv = tl.load(HOUT + r * (S * D) + s * D + d).to(tl.float32)
            else:
                hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
            w = tl.load(SCALE + s * D + d).to(tl.float32)
            tl.store(NORMED + r * (S * D) + s * D + d, (hv * rinv * w).to(tl.bfloat16))

    @triton.jit
    def _b16mm_sm_act(X, W, PART, ACT, INJ, TICK, M, x_stride,
                      N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
                      BLOCK_N: tl.constexpr, BK: tl.constexpr, S: tl.constexpr, LOW: tl.constexpr,
                      LOWP: tl.constexpr, HAS_INJ: tl.constexpr):
        """``bf16._b16mm_sm`` (SK > 1: its fp32 slices into PART [SK, M, N]), then the row tile's last program to
        finish runs ``glue._hc_act_sk`` on the tile's rows (slices in slice order, act and the inject gates; no
        32-group sums). TICK [row tiles] int32 counts finished programs and is left at 0 for the next launch."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        n_ok = rn < N
        KS: tl.constexpr = K // SK
        NB: tl.constexpr = KS // BK
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for i in range(NB):
            k0 = (pid_s * NB + i) * BK
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + pid_s * (N * KS) + rn[:, None] * KS + (i * BK + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
        # every wave's stores done before the ticket (the barrier's release), the last program reads past the caches
        tl.debug_barrier()
        tiles = tl.num_programs(1) * tl.num_programs(2)
        t = tl.atomic_add(TICK + pid_m, 1, sem="acq_rel", scope="gpu")
        if t == tiles - 1:
            tl.atomic_xchg(TICK + pid_m, 0, sem="relaxed", scope="gpu")
            j = tl.arange(0, LOWP)
            ok = j < LOW
            r1 = tl.minimum(pid_m * BM + BM, M)
            for r in range(pid_m * BM, r1):
                v = tl.load(PART + r * N + j, mask=ok, other=0.0, cache_modifier=".cv")
                for s_ in range(1, SK):
                    v = v + tl.load(PART + (s_ * M + r) * N + j, mask=ok, other=0.0, cache_modifier=".cv")
                v = (v / S).to(tl.bfloat16).to(tl.float32)
                a = _bsilu(v)
                a = tl.where(ok, a, 0.0)
                tl.store(ACT + r * LOW + j, a.to(tl.bfloat16), mask=ok)
                if HAS_INJ:
                    sv = tl.arange(0, S)
                    iv = tl.load(PART + r * N + LOW + sv, cache_modifier=".cv")
                    for s_ in range(1, SK):
                        iv = iv + tl.load(PART + (s_ * M + r) * N + LOW + sv, cache_modifier=".cv")
                    iv = (iv / S).to(tl.bfloat16).to(tl.float32)
                    tl.store(INJ + r * S + sv, (2.0 * _bsig(iv)).to(tl.bfloat16))
except ModuleNotFoundError:
    HAS_TRITON = False
