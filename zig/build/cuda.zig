//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");
const hip_build = @import("hip.zig");

/// The GPU runtime a Linux build targets: NVIDIA's driver, or AMD's HIP runtime (-Dgpu=hip).
pub const Gpu = enum { cuda, hip };

var gpu_option: ?Gpu = null;

/// -Dgpu, declared once for the targets and the host tests.
fn gpuOption(b: *std.Build) Gpu {
    if (gpu_option == null) gpu_option = b.option(Gpu, "gpu", "GPU runtime: cuda (default) or hip") orelse .cuda;
    return gpu_option.?;
}

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
pub const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

pub const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    // Flash Next (qwen4_exp): copies of its extensions' device code (zig/tests/cuda/copies.py), the same flags
    .{ .name = "fn_gdn", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn.py, tensorfold_qwen4_exp_gdn
    .{ .name = "fn_gdn_io", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn_io.py, tensorfold_qwen4_exp_gdn_io
    .{ .name = "fn_gdn_prefill", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "fn_gdn_tree", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "fn_nvfp4_experts", .flags = &.{"-O3"} }, // cuda/nvfp4/linear.py, tensorfold_nvfp4_v3
    .{ .name = "fn_qmm", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "fn_qmm_prefill", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "fn_pack", .flags = &.{"-O3"} }, // ours: the n-gram table's GPU gather (flashnext/cuda_weights.zig)
    .{ .name = "torch_fn_ops", .src = "torch_ops/fn_ops", .flags = torch_ops }, // ours: Flash Next torch ops (SwiGLU, sampler packing)
    .{ .name = "torch_fn_logsumexp", .src = "torch_ops/fn_logsumexp", .flags = torch_ops }, // ours: torch.logsumexp in ATen's order
    .{ .name = "fn_experts_prompt", .flags = &.{"-O3"} }, // ours: routed NVFP4 experts reading weights once a prompt call (fn_nvfp4_experts' bits)
    // INT4-AutoRound checkpoint (GPTQ int4 experts and head, 128x128-block FP8 dense linears)
    .{ .name = "fn_qmmf", .flags = &.{"-O3"} }, // cuda/nvfp4/qmmf.cu (FP8G lane matmul), tensorfold_nvfp4_v3
    .{ .name = "fn_int4", .flags = &.{"-O3"} }, // ours: GPTQ int4 g128 routed experts and lm_head (W4A16, fp32 sums)
    .{ .name = "fn_qmmf_ld", .flags = &.{"-O3"} }, // fn_qmmf.cu with an output row stride (tools/zig/flashnext_qmmf_ld.py)
    .{ .name = "fn_nvfp4_shape", .flags = &.{"-O3"} }, // ours (decode D3): fn_nvfp4_experts' unit arithmetic in other launch shapes
    .{ .name = "fn_qmm_cluster", .flags = &.{"-O3"} }, // ours (decode D5): qmm_kernel's K-slice cluster forms for the MTP head in 4-bit
    .{ .name = "fn_roce", .flags = &.{"-O3"} }, // ours (decode D1): the one-shot RoCE all-gather's GPU half (cuda/roce.zig)
    .{ .name = "fn_qsa_scores", .flags = &.{"-O3"} }, // ours: attention._scores' bits from row tiles (the prompt indexer)
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, gpu: Gpu, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(Gpu, "gpu", gpu);
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    // the one-shot RoCE all-gather's proxy (decode D1): plain C over libibverbs' header, the library opened at run time
    // (roce_proxy_stub.c, which refuses every open, where /usr/include has no infiniband/verbs.h)
    const verbs = if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module, tokenizer: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    // Flash Next on CUDA: its root sits beside the Metal family's host files (config.zig), which it shares
    const flashnext = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    flashnext.addImport("cuda", cuda);
    flashnext.addImport("core", core);
    flashnext.addImport("lanes", lanes);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .flashnext = flashnext, .tokenizer = tokenizer };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones; -Dgpu=hip -Dhipcc builds AMDGPU code
/// objects instead), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module) void {
    const gpu = gpuOption(b);
    const hipcc = b.option([]const u8, "hipcc", "hipcc that builds the HIP kernel code objects (-Dgpu=hip)");
    const arches = b.option([]const u8, "offload-arch", "AMDGPU targets, comma separated (default gfx1151)") orelse "gfx1151";
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const compiler: ?[]const u8 = if (gpu == .hip) hipcc else nvcc;
    const version: ?std.Build.LazyPath = if (prebuilt == null and compiler != null) blk: {
        const run = b.addSystemCommand(&.{ compiler.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (gpu == .hip) {
            if (hipcc) |tool| image.* = hip_build.codeObject(b, tool, version.?, k, arches);
        } else if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.{s}", .{ k.name, if (gpu == .hip) "hsaco" else "fatbin" })).step);
    }
    const built = if (gpu == .hip) hipcc != null else nvcc != null or prebuilt != null;
    const cuda = runtime(b, target, optimize, gpu, if (built) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addImport("flashnext", mods.flashnext);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("flashnext", mods.flashnext);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    const native = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.flashnext, mods.tokenizer);
    // the Triton AOT set (tools/zig/flashnext_aot.py build) where tensorfold-native looks without TENSORFOLD_CUDA_KERNELS
    if (b.option([]const u8, "kernel-set", "absolute directory of a Triton AOT set (aot.json + cubins/ or hsaco/) to install as native/share/tensorfold/cuda/<arch>")) |dir| {
        const first = struct {
            fn of(list: []const u8) []const u8 {
                return std.mem.sliceTo(list, ',');
            }
        }.of;
        const arch = if (gpu == .hip) first(arches) else b.fmt("sm{s}", .{first(sms)});
        native.dependOn(&b.addInstallDirectory(.{ .source_dir = b.graph.cwdRelativePath(dir), .install_dir = .{ .custom = "native" }, .install_subdir = b.fmt("share/tensorfold/cuda/{s}", .{arch}) }).step);
    }
    // Flash Next's two-rank transport test (zig/tests/cuda/flashnext/box/comm.sh): its own step, not in `install`
    const comm_test = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext/comm_test.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "flashnext", .module = mods.flashnext } } });
    b.step("flashnext-comm", "tf-flashnext-comm RANK MASTER PORT: two-rank all-gathers, exact and timed").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-flashnext-comm", .root_module = comm_test }), .{}).step);
    // Flash Next's weights against the oracle's digests (zig/tests/cuda/flashnext/weights_check.zig): its own step
    const weights_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext/weights_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "flashnext", .module = mods.flashnext } } });
    b.step("flashnext-weights", "tf-flashnext-weights SNAPSHOT [WEIGHTS.json]: the loaded weights' digests against the oracle's").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-flashnext-weights", .root_module = weights_check }), .{}).step);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "flashnext", .module = flashnext } },
    });
    return .{ .api = api, .engines = mod };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module, tokenizer: *std.Build.Module) *std.Build.Step {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, flashnext);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines } },
    }) });
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    const step = b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin (-Dkernel-set: its Triton set)");
    step.dependOn(&install.step);
    return step;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, gpuOption(b), &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const native = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.flashnext).engines;
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.flashnext, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addImport("flashnext", mods.flashnext);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}
