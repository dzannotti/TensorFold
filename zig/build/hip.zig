//! The HIP half of the kernel build (-Dgpu=hip): each kernel of cuda.zig's list as an AMDGPU code object from hipcc,
//! with its nvcc flags mapped to clang's. A kernel hipcc cannot build embeds a marker instead, so only loading that
//! module fails (cuda/module.zig prints the compiler's errors); every other kernel still serves.

const std = @import("std");
const Kernel = @import("cuda.zig").Kernel;

/// torch.utils.cpp_extension's hipcc flags (COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS), plus the bf16 twin of its
/// CUDA build's __CUDA_NO_BFLOAT16_CONVERSIONS__, and C++20 as the CUDA build.
const torch_flags = [_][]const u8{
    "-D__HIP_PLATFORM_AMD__=1",
    "-DUSE_ROCM=1",
    "-DCUDA_HAS_FP16=1",
    "-D__HIP_NO_HALF_OPERATORS__=1",
    "-D__HIP_NO_HALF_CONVERSIONS__=1",
    "-D__HIP_NO_BFLOAT16_CONVERSIONS__=1",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
    "-std=c++20",
};

/// A kernel's nvcc flag as clang spells it; a flag with no mapping stops the build here, not at run time.
fn clangFlag(flag: []const u8) []const u8 {
    if (std.mem.eql(u8, flag, "-O3")) return "-O3";
    if (std.mem.eql(u8, flag, "--fmad=false")) return "-ffp-contract=off";
    if (std.mem.eql(u8, flag, "--ftz=false")) return "-fno-gpu-flush-denormals-to-zero";
    std.debug.panic("zig/build/hip.zig: no hipcc spelling for nvcc flag {s}", .{flag});
}

/// Builds code objects with hipcc for `arches`; $1 out, $2 depfile, $3 kernel name, $4 source, then hipcc and flags.
const script =
    \\out=$1 dep=$2 name=$3 src=$4; shift 4; hipcc=$1; shift
    \\"$hipcc" "$@" --offload-host-only -M -MF "$dep" -MT "$out" "$src" 2>/dev/null || printf '%s: %s\n' "$out" "$src" > "$dep"
    \\if ! log=$("$hipcc" --genco "$@" -o "$out" "$src" 2>&1); then
    \\  printf 'TF_HIP_BUILD_FAILED %s (%s): %s\n' "$name" "$src" "$(printf '%s' "$log" | grep -m8 'error' )" > "$out"
    \\  echo "warning: hipcc could not build $name; loading it will fail" >&2
    \\fi
;

fn exists(b: *std.Build, path: []const u8) bool {
    b.root.access(b.graph.io, path, .{}) catch return false;
    return true;
}

/// The source hipcc compiles: zig/kernels/hip/<name>.hip when a kernel has a HIP rewrite, else the shared .cu.
fn source(b: *std.Build, k: Kernel) []const u8 {
    const hip = b.fmt("zig/kernels/hip/{s}.hip", .{k.name});
    if (exists(b, hip)) return hip;
    return b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name});
}

pub fn codeObject(b: *std.Build, hipcc: []const u8, version: std.Build.LazyPath, k: Kernel, arches: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ "sh", "-c", script, "sh" });
    run.addFileInput(version);
    const out = run.addOutputFileArg(b.fmt("{s}.hsaco", .{k.name}));
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg(k.name);
    const src = source(b, k);
    run.addFileArg(b.path(src));
    run.addArgs(&.{ hipcc, "-x", "hip" });
    run.addArgs(&torch_flags);
    for (k.flags) |f| run.addArg(clangFlag(f));
    var it = std.mem.tokenizeScalar(u8, arches, ',');
    while (it.next()) |arch| run.addArg(b.fmt("--offload-arch={s}", .{arch}));
    // the CUDA-compat header the shared .cu sources build against, when the tree has it
    const compat = "zig/kernels/cuda/hip_compat.cuh";
    if (exists(b, compat)) run.addPrefixedFileArg("-include", b.path(compat));
    run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/cuda/hip"));
    run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/cuda"));
    return out;
}
