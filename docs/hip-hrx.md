# HIP families on AMD HRX

`TENSORFOLD_GPU=hrx` runs the HIP families without a HIP library: `zig/src/hip/hrx/` fills the same HIP entry-point
table (`abi.Api`) from libhrx's native C API (AMD HRX, `libhrx.so.0`; `TENSORFOLD_HRX` names another path).
The kernels are the same hipcc code objects; only the runtime under them changes.

```sh
TENSORFOLD_GPU=hrx TENSORFOLD_HRX=$HRX/lib/libhrx.so.0 LD_LIBRARY_PATH=$HRX/lib ./tensorfold-native ...
```

## What maps to what

| HIP | HRX |
|---|---|
| `hipMalloc`, `hipHostMalloc` | device-local and host-local buffers; a sorted table maps HIP's u64 addresses back to buffer and offset |
| kernels, device copies | direct queue operations on the stream's timeline (`hrx_queue_dispatch`, `hrx_queue_copy_linear`) |
| fills | HRX stream fills, submitted at once |
| copies from or to a pinned (`hipHostMalloc`) range | buffer-to-buffer queue copies |
| copies from or to other host memory | HRX stream copies (HRX stages them) |
| streams, events | one HRX stream on its own acquired queue each; HRX events |
| graphs | the driver's node lists; a replay records every node on one stream batch and submits it once |
| `hipModuleLoadData` | `hrx_executable_load_data`; argument offsets, registers and LDS come from the code object's metadata note |
| device attributes, free memory | the KFD topology and the amdgpu PCI node in sysfs, not HRX |

## Calls that need a patched libhrx

Stock libhrx has no native entry point for three HIP features. The driver resolves them by name; without them only
the calls that need them are refused (`hipErrorNotSupported`), and everything else runs:

| HIP call | libhrx entry point needed |
|---|---|
| launches with dynamic shared memory (`sharedMemBytes > 0`), also inside graphs | `hrx_stream_dispatch_with_local_memory` |
| `hipModuleLaunchCooperativeKernel` | the same, with `HRX_DISPATCH_FLAG_COOPERATIVE` |
| `hipModuleGetGlobal` | `hrx_executable_lookup_global` |

Stream capture records kernels, device copies and fills; event records, event waits and host copies on a capturing
stream are refused.
