"""Flash Next's DeltaNet kernels on gfx1151: fn_gdn_io (front, back), fn_gdn_prefill (chain) and fn_gdn_tree (tree,
replay) as HIP code objects launched with cuda_kernels.zig's geometry, against

  1. the Python extensions' own wrappers (gdn_io.cpp/.cu, gdn.cpp/.cu/gdn_prefill.cu) built by torch's
     cpp_extension on ROCm (hipified; with hip_compat.cuh force-included, since the hipified sources alone do not
     compile on HIP 7: no __float2bfloat16_rn, 32-bit shuffle masks; and the one host-side cast
     hipFuncSetAttribute needs): raw bytes;
  2. an fp64 host-order reference: worst error in units of the output's last place (bf16 or fp32);
  3. themselves: run twice (determinism), one row or stream alone vs in a batch, a prefill split in two chunks.

Run in the rocm-dev container: python tools/rocm/check_gdn.py --out <dir>
"""

import argparse
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hipmod import FLAGS, KERNELS, ROOT, Module, genco, ptr, symbols  # noqa: E402

I, F, B = ctypes.c_int, ctypes.c_float, ctypes.c_bool
NK, NV, DK, DV = 16, 48, 128, 128
C = 2 * NK * DK + NV * DV
PW = C + NV * DV + 2 * NV
SMS = torch.cuda.get_device_properties(0).multi_processor_count
TREES = [(0, 8, 4, True), (1, 8, 4, False), (2, 8, 2, False), (2, 4, 4, False), (4, 2, 4, False), (8, 2, 4, False),
         (16, 2, 2, False), (32, 2, 1, False)]  # cuda_kernels.zig tree_variants


class Pending(ctypes.Structure):
    _fields_ = [('k', ctypes.c_void_p), ('v', ctypes.c_void_p), ('g', ctypes.c_void_p), ('beta', ctypes.c_void_p),
                ('rows', ctypes.c_void_p), ('row_stride', ctypes.c_int), ('counts', ctypes.c_void_p),
                ('count_stride', ctypes.c_int)]


def tree_index(slots):  # cuda_kernels.zig treeIndex on a GPU that is never "wide"
    return 0 if slots == 0 else 3 if slots <= 2 else 4 if slots <= 4 else 5 if slots <= 8 else 6 if slots <= 16 else 7


def references(out):
    """The Python extensions, built from copies of their sources (hipify writes next to them)."""
    from torch.utils import cpp_extension

    src = ROOT / 'src/tensorfold'
    flags = [f for f in FLAGS if f not in ('-x', 'hip') and not f.startswith('--offload-arch')]
    mods = {}
    for name, files in (('gdn_io', ['families/qwen4_exp/cuda/gdn_io.cpp', 'families/qwen4_exp/cuda/gdn_io.cu']),
                        ('gdn', ['cuda/kernels/gdn.cpp', 'cuda/kernels/gdn.cu', 'cuda/kernels/gdn_prefill.cu'])):
        d = out / 'ref' / name
        d.mkdir(parents=True, exist_ok=True)
        copies = []
        for f in files:
            # host side only: hipFuncSetAttribute takes the kernel as const void*
            text = (src / f).read_text().replace('cudaFuncSetAttribute(kernel,', 'cudaFuncSetAttribute((const void*)kernel,')
            (d / Path(f).name).write_text(text)
            copies.append(str(d / Path(f).name))
        mods[name] = cpp_extension.load(name=f'tf_rocm_ref_{name}', sources=copies, extra_cuda_cflags=flags,
                                        build_directory=str(d), verbose=False)
    return mods


class Cells:
    def __init__(self):
        self.bad = self.n = 0

    def same(self, name, a, b):
        torch.cuda.synchronize()
        ok = a.shape == b.shape and torch.equal(a.contiguous().view(-1).view(torch.uint8),
                                                b.contiguous().view(-1).view(torch.uint8))
        self.n += 1
        self.bad += not ok
        print(f"{'EQUAL' if ok else 'DIFFER'} {name}", flush=True)

    def ulps(self, name, got, want, limit, bf16=False, rms=False):
        """Worst |got - want| in units of the last place (bf16, or fp32 unless `bf16`) at want, or (`rms`, for sums
        that cancel) at the larger of want and its root mean square; fails past `limit`."""
        g, w = got.double(), want.double()
        eps = 2.0 ** -7 if bf16 or got.dtype == torch.bfloat16 else 2.0 ** -23
        mag = torch.maximum(w.abs(), w.pow(2).mean().sqrt()) if rms else w.abs()
        ulp = torch.clamp(torch.exp2(torch.floor(torch.log2(mag.clamp_min(1e-30)))), min=2.0 ** -126) * eps
        err = ((g - w).abs() / ulp)
        worst = float(err.max())
        self.n += 1
        self.bad += not worst <= limit
        print(f"{'WITHIN' if worst <= limit else 'PAST'} {name}: worst {worst:.2f} ulp (limit {limit}), "
              f"mean {float(err.mean()):.3f}", flush=True)


def bf(x):
    return x.to(torch.bfloat16).double()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ref = references(a.out)
    cos = {n: genco(f'{n}.cu', a.out) for n in ('fn_gdn_io', 'fn_gdn_prefill', 'fn_gdn_tree')}
    mods = {n: Module(c) for n, c in cos.items()}
    sym = {n: symbols(c) for n, c in cos.items()}
    (a.out / 'symbols.txt').write_text('\n'.join(s for n in sym for s in sym[n]) + '\n')

    def fn(mod, *parts):
        return mods[mod].function(next(s for s in sym[mod] if all(p in s for p in parts)))

    cells = Cells()
    dev = 'cuda'
    torch.manual_seed(5151)

    # ---- front / back (gdn_io), as Ops.gdnFront / gdnBack launch them --------------------------------------------
    front, back = fn('fn_gdn_io', 'front_kernelILi16ELi48E'), fn('fn_gdn_io', 'back_kernelILi16ELi48E')
    streams = 3
    states = [(torch.randn((3, C), device=dev) * 2).to(torch.bfloat16) for _ in range(streams)]
    conv_ptrs = torch.tensor([s.data_ptr() for s in states], dtype=torch.int64, device=dev)
    cw = (torch.randn((C, 4), device=dev) * 0.5).to(torch.bfloat16)
    a_log = torch.randn(NV, device=dev) * 0.5
    dt_bias = torch.randn(NV, device=dev)
    norm_w = (1 + 0.1 * torch.randn(DV, device=dev)).to(torch.bfloat16)

    def run_front(p, sid, win):
        rows = win.shape[0]
        q = torch.empty((rows, NK, DK), device=dev)
        k = torch.empty_like(q)
        v = torch.empty((rows, NV, DV), device=dev, dtype=torch.bfloat16)
        g = torch.empty((rows, NV), device=dev)
        beta = torch.empty_like(g)
        mods['fn_gdn_io'].launch(front, (rows, NV // 3 + NV), (128,), [ptr(x) for x in (p, conv_ptrs, sid, win, cw,
                                 a_log, dt_bias, q, k, v, g, beta)])
        return q, k, v, g, beta

    def run_back(y, p, eps):
        rows = y.shape[0]
        out = torch.empty((rows, NV * DV), device=dev, dtype=torch.bfloat16)
        xs = torch.empty((rows, NV * DV // 32), device=dev)
        mods['fn_gdn_io'].launch(back, (rows, NV), (128,), [ptr(y), ptr(p), ptr(norm_w), F(eps), ptr(out), ptr(xs)])
        return out, xs

    for rows in (1, 5, 16):
        p = (torch.randn((rows, PW), device=dev) * 2).to(torch.bfloat16)
        sid = torch.randint(0, streams, (rows,), dtype=torch.int32, device=dev)
        win = torch.stack([torch.randint(0, 3 + rows, (rows,)) for _ in range(4)], 1).to(torch.int32).to(dev)
        got = run_front(p, sid, win)
        want = tuple(torch.empty_like(x) for x in got)
        ref['gdn_io'].front(p, conv_ptrs, sid, win, cw, a_log, dt_bias, *want)
        for name, x, w in zip(('q', 'k', 'v', 'g', 'beta'), got, want):
            cells.same(f'front/{name}/r{rows} vs extension', x, w)
        again = run_front(p, sid, win)
        cells.same(f'front/r{rows} twice', torch.cat([x.view(-1).view(torch.uint8) for x in again]),
                   torch.cat([x.view(-1).view(torch.uint8) for x in got]))
        # decode windows (three state rows, then the row itself): the last row alone is the same bits
        last = rows - 1
        own = torch.tensor([[0, 1, 2, 3 + r] for r in range(rows)], dtype=torch.int32, device=dev)
        batch = run_front(p, sid, own)
        solo = run_front(p[last:].contiguous(), sid[last:].contiguous(), own[:1].contiguous())
        cells.same(f'front/r{rows} last row alone', torch.cat([x[last:].reshape(-1).view(torch.uint8) for x in batch]),
                   torch.cat([x.reshape(-1).view(torch.uint8) for x in solo]))
        # fp64: conv taps in order, SiLU, the bf16 round, L2 norms; gates
        pd = p.double()
        x = torch.empty((rows, C), dtype=torch.float64, device=dev)
        acc = torch.zeros_like(x)
        for tap in range(4):
            src = win[:, tap].long()
            from_state = torch.stack([states[s][min(int(t), 2)] for s, t in zip(sid.tolist(), src.tolist())]).double()
            from_p = pd[(src - 3).clamp_min(0), :C]
            x = torch.where((src < 3)[:, None], from_state, from_p)
            acc = acc + cw[:, tap].double() * x
        act = bf(acc / (1 + torch.exp(-acc)))
        qd, kd = act[:, :NK * DK].view(rows, NK, DK), act[:, NK * DK:2 * NK * DK].view(rows, NK, DK)
        qd = qd / torch.sqrt((qd * qd).sum(-1, keepdim=True) + 1e-6) / DK ** 0.5
        kd = kd / torch.sqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
        cells.ulps(f'front/q/r{rows} vs fp64', got[0], qd, 8)
        cells.ulps(f'front/k/r{rows} vs fp64', got[1], kd, 8)
        cells.ulps(f'front/v/r{rows} vs fp64', got[2], act[:, 2 * NK * DK:].reshape(rows, NV, DV), 1)
        b, av = pd[:, C + NV * DV:C + NV * DV + NV], pd[:, C + NV * DV + NV:]
        sp = torch.nn.functional.softplus(av + dt_bias.double(), threshold=20)
        cells.ulps(f'front/g/r{rows} vs fp64', got[3], torch.exp(-torch.exp(a_log.double()) * sp), 16)
        cells.ulps(f'front/beta/r{rows} vs fp64', got[4], 1 / (1 + torch.exp(-b)), 1, bf16=True)

        y = torch.randn((rows, NV, DV), device=dev).to(torch.bfloat16)
        got = run_back(y, p, 1e-6)
        want = (torch.empty_like(got[0]), torch.empty_like(got[1]))
        ref['gdn_io'].back(y, p, norm_w, 1e-6, *want)
        cells.same(f'back/out/r{rows} vs extension', got[0], want[0])
        cells.same(f'back/xs/r{rows} vs extension', got[1], want[1])
        solo = run_back(y[last:].contiguous(), p[last:].contiguous(), 1e-6)
        cells.same(f'back/r{rows} last row alone', got[0][last:], solo[0])
        yd = y.double()
        rinv = 1 / torch.sqrt((yd * yd).mean(-1, keepdim=True) + 1e-6)
        z = pd[:, C:C + NV * DV].view(rows, NV, DV)
        o = bf(bf(bf(yd * rinv) * norm_w.double()) * (1 / (1 + torch.exp(-z))))
        cells.ulps(f'back/out/r{rows} vs fp64', got[0], o.view(rows, -1), 1)
        cells.ulps(f'back/xs/r{rows} vs fp64', got[1], got[0].double().view(rows, -1, 32).sum(-1), 8, rms=True)

    # ---- the delta rule: inputs as front gives them ---------------------------------------------------------------
    def inputs(w, hk=NK, hv=NV, keys=torch.float32):
        q = torch.nn.functional.normalize(torch.randn((w, hk, DK), device=dev), dim=-1) / DK ** 0.5
        k = torch.nn.functional.normalize(torch.randn((w, hk, DK), device=dev), dim=-1)
        v = torch.randn((w, hv, DV), device=dev).to(torch.bfloat16)
        g = torch.exp(-torch.rand((w, hv), device=dev) * 0.3)
        beta = torch.rand((w, hv), device=dev)
        return q.to(keys).contiguous(), k.to(keys).contiguous(), v, g, beta

    def chain64(q, k, v, g, beta, s, rows):
        """fp64 serial steps over `rows` (window indices), from state s (hv, dv, dk); outputs (len, hv, dv)."""
        s = s.double().clone()
        rep = NV // q.shape[1]
        ys = []
        for t in rows:
            qt, kt = q[t].double().repeat_interleave(rep, 0), k[t].double().repeat_interleave(rep, 0)
            s = s * g[t].double()[:, None, None]
            mem = (s * kt[:, None, :]).sum(-1)
            delta = (v[t].double() - mem) * beta[t].double()[:, None]
            s = s + kt[:, None, :] * delta[:, :, None]
            ys.append((s * qt[:, None, :]).sum(-1))
        return torch.stack(ys), s

    # prefill (Ops.gdnPrefill): 128 value rows a block when hv >= SMs
    pre = {(kt, r): fn('fn_gdn_prefill', f'chain_kernelI{kt}Li{r}ELi32E') for kt in ('f', '14__hip_bfloat16')
           for r in (128, 64)}

    def run_prefill(q, k, v, g, beta, state):
        w, hk, hv = q.shape[0], q.shape[1], v.shape[1]
        rows = 128 if hv >= SMS else 64
        last = torch.empty_like(state)
        y = torch.empty((w, hv, DV), device=dev, dtype=torch.bfloat16)
        kt = 'f' if q.dtype == torch.float32 else '14__hip_bfloat16'
        mods['fn_gdn_prefill'].launch(pre[(kt, rows)], (hv, 128 // rows), (2 * rows,),
                                      [ptr(x) for x in (q, k, v, g, beta, state, last, y)] + [I(w), I(hk), I(hv)])
        return y, last

    for keys in (torch.float32, torch.bfloat16):
        for w in (1, 33, 300):
            q, k, v, g, beta = inputs(w, keys=keys)
            state = torch.randn((NV, DV, DK), device=dev) * 0.1
            y, last = run_prefill(q, k, v, g, beta, state)
            want_last = torch.empty_like(state)
            want_y = ref['gdn'].prefill(q, k, v, g, beta, state, want_last)
            kn = str(keys)[6:]
            cells.same(f'prefill/{kn}/w{w}/y vs extension', y, want_y)
            cells.same(f'prefill/{kn}/w{w}/last vs extension', last, want_last)
            if w > 1:
                h = w // 3
                y1, mid = run_prefill(q[:h].contiguous(), k[:h].contiguous(), v[:h].contiguous(), g[:h].contiguous(),
                                      beta[:h].contiguous(), state)
                y2, end = run_prefill(q[h:].contiguous(), k[h:].contiguous(), v[h:].contiguous(), g[h:].contiguous(),
                                      beta[h:].contiguous(), mid)
                cells.same(f'prefill/{kn}/w{w} in two chunks', torch.cat([y1, y2]), y)
                cells.same(f'prefill/{kn}/w{w} two chunks last', end, last)
            if keys == torch.float32:
                y64, s64 = chain64(q, k, v, g, beta, state, range(w))
                cells.ulps(f'prefill/w{w}/y vs fp64', y, y64, 2, rms=True)
                cells.ulps(f'prefill/w{w}/last vs fp64', last, s64, 64, rms=True)

    # tree (Ops.gdnTree): chains (slots 0) and trees in every variant; one state, or a table of streams
    tree_fns = [fn('fn_gdn_tree', f'tree_kernelIfLi{s}ELi{r}ELi{wp}ELb{int(c)}E') for s, r, wp, c in TREES]
    replay_fn = fn('fn_gdn_tree', 'replay_kernelIfLi8ELi4E')
    sys.path.insert(0, str(ROOT / 'src'))
    from tensorfold.cuda.kernels.gdn import plan_host

    def run_tree(q, k, v, g, beta, entries, starts, slots, max_rows, state=None, table=None, final=None,
                 pending=None):
        nodes = q.shape[0]
        nstreams = 1 if table is None else table.numel()
        i = tree_index(slots)
        s, r, wp, c = TREES[i]
        shared = 16 * wp * s * r * 32 + (0 if c else 12 * max_rows)
        step = r * wp
        vec = DV % r == 0 and v.data_ptr() % (2 * min(r, 8)) == 0
        y = torch.empty((nodes, NV, DV), device=dev, dtype=torch.bfloat16)
        fs = final if table is None else None
        ft = final if table is not None else None
        mods['fn_gdn_tree'].launch(tree_fns[i], ((DV + step - 1) // step, NV, nstreams), (32 * wp,),
                                   [ptr(x) for x in (q, k, v, g, beta, state, table, starts, entries)] +
                                   [I(nodes), ptr(y), I(NK), I(NV), I(DV), pending or Pending(), ptr(fs), ptr(ft),
                                    B(vec)], shared)
        return y

    def tree_case(name, parents_list):
        entries, starts, slots, max_rows = plan_host(parents_list)
        nodes = starts[-1]
        q, k, v, g, beta = inputs(nodes)
        e_dev = torch.tensor(entries, dtype=torch.int32, device=dev).view(-1, 3)
        st_dev = torch.tensor(starts, dtype=torch.int32, device=dev)
        st = [torch.randn((NV, DV, DK), device=dev) * 0.1 for _ in parents_list]
        table = torch.tensor([x.data_ptr() for x in st], dtype=torch.int64, device=dev)
        y = run_tree(q, k, v, g, beta, e_dev, st_dev, slots, max_rows, table=table)
        want = ref['gdn'].tree(q, k, v, g, beta, None, table, st_dev, e_dev, slots, max_rows, None, None)
        cells.same(f'tree/{name}/slots{slots} vs extension', y, want)
        cells.same(f'tree/{name} twice', run_tree(q, k, v, g, beta, e_dev, st_dev, slots, max_rows, table=table), y)
        # the last stream alone (its own window, one state)
        a0, a1 = starts[-2], starts[-1]
        e1, _, s1, m1 = plan_host(parents_list[-1:])
        solo = run_tree(q[a0:a1].contiguous(), k[a0:a1].contiguous(), v[a0:a1].contiguous(), g[a0:a1].contiguous(),
                        beta[a0:a1].contiguous(), torch.tensor(e1, dtype=torch.int32, device=dev).view(-1, 3), None,
                        s1, max(m1, a1 - a0), state=st[-1])
        if tree_index(s1) == tree_index(slots):
            cells.same(f'tree/{name} last stream alone', solo, y[a0:a1])
        # fp64: each node from its parent's state
        for si, parents in enumerate(parents_list):
            base = starts[si]
            outs = []
            for node in range(len(parents)):
                path = []
                n = node
                while n >= 0:
                    path.append(base + n)
                    n = parents[n]
                outs.append(chain64(q, k, v, g, beta, st[si], path[::-1])[0][-1])
            cells.ulps(f'tree/{name}/stream{si} vs fp64', y[base:base + len(parents)], torch.stack(outs), 2, rms=True)

    tree_case('chains', [[-1, 0, 1, 2], [-1, 0, 1, 2, 3, 4, 5], [-1, 0]])
    tree_case('binary7', [[-1, 0, 0, 1, 1, 2, 2]])
    tree_case('fan4', [[-1, 0, 0, 0, 0], [-1, 0, 1, 1, 1]])
    tree_case('bushy', [[-1] + [max(0, (i - 1) // 3) for i in range(1, 25)]])
    tree_case('wide16', [[-1] + [0] * 16 + [i for i in range(1, 17)]])
    tree_case('wide32', [[-1] + [0] * 31 + [i for i in range(1, 32)] + [32 + i for i in range(31)]])
    tree_case('binary8', [[-1] + [(i - 1) // 2 for i in range(1, 255)]])            # depth-first: 7 live states
    tree_case('binary10', [[-1] + [(i - 1) // 2 for i in range(1, 1023)]])          # 9
    cat = [-1]
    for _ in range(17):                     # 17 caterpillars: each spine node's leaf waits for the whole spine
        spine = len(cat)
        cat.append(0)
        for _ in range(16):
            cat += [spine, spine]
            spine = len(cat) - 2
    tree_case('caterpillars', [cat])

    # a chain with a final state (slots 0, one state) and the replay of its accepted rows
    w = 9
    q, k, v, g, beta = inputs(w)
    state = torch.randn((NV, DV, DK), device=dev) * 0.1
    entries = torch.tensor(plan_host([list(range(-1, w - 1))])[0], dtype=torch.int32, device=dev).view(-1, 3)
    final = torch.empty_like(state)
    y = run_tree(q, k, v, g, beta, entries, None, 0, w, state=state, final=final)
    want_final = torch.empty_like(state)
    want = ref['gdn'].tree(q, k, v, g, beta, state, None, None, entries, 0, w, None, want_final)
    cells.same('tree/chain+final vs extension', y, want)
    cells.same('tree/chain final state vs extension', final, want_final)
    y64, s64 = chain64(q, k, v, g, beta, state, range(w))
    cells.ulps('tree/chain final vs fp64', final, s64, 64, rms=True)
    accepted = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32, device=dev)
    counts = torch.tensor([w], dtype=torch.int32, device=dev)
    st = state.clone()
    table = torch.tensor([x.data_ptr() for x in (k, v, g, beta, st)], dtype=torch.int64, device=dev)
    out = torch.empty((1, 1, NV, DV, DK), device=dev)
    mods['fn_gdn_tree'].launch(replay_fn, ((DV + 31) // 32, NV, 1), (128,), [ptr(table), I(1), ptr(accepted), I(w),
                               ptr(counts), I(1), ptr(out), I(NK), I(NV), I(DV)])
    want_out = ref['gdn'].replay(table, 1, 1, accepted, counts, NK, NV, DV, True, False)
    cells.same('replay vs extension', out, want_out)
    cells.same('replay == the chain final state', out[0, 0], final)

    print(f"{'PASS' if cells.bad == 0 else 'FAIL'} gdn: {cells.n - cells.bad} of {cells.n}", flush=True)
    return 1 if cells.bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
