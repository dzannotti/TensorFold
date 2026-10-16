# Header-only safetensors reader over numpy memmaps (no torch, no model code).
import json, struct, os
import numpy as np

DT = {'I8': np.int8, 'F16': np.float16, 'BF16': np.uint16, 'F32': np.float32, 'U32': np.uint32, 'I32': np.int32, 'U8': np.uint8}


class Dir:
    def __init__(self, d):
        self.where = {}
        for f in sorted(os.listdir(d)):
            if not f.endswith('.safetensors'):
                continue
            p = os.path.join(d, f)
            with open(p, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != '__metadata__':
                    self.where[k] = (p, 8 + n, v)

    def get(self, k):
        p, base, v = self.where[k]
        b, e = v['data_offsets']
        a = np.memmap(p, dtype=DT[v['dtype']], mode='r', offset=base + b, shape=tuple(v['shape']))
        if v['dtype'] == 'BF16':
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a


def fwht128(x):
    """FP32 Sylvester H128 over the last axis in 128 blocks, steps 1..64, times 1/sqrt(128) (0x3db504f3)."""
    x = np.array(x, dtype=np.float32)
    sh = x.shape
    x = x.reshape(-1, 128)
    h = 1
    while h < 128:
        y = x.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        x = y.reshape(-1, 128)
        h *= 2
    x = x * np.float32(np.frombuffer(struct.pack('<I', 0x3db504f3), np.float32)[0])
    return x.reshape(sh)
