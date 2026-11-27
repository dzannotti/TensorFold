//! Loom kernels (zig/kernels/loom/<kernel>/) compiled ahead of time by -Dloom-compile; absent without it (hipcc's run).
const std = @import("std");

pub const Options = struct { compile: ?[]const u8, target: []const u8 };

/// The hrx-loom release the sources are written and checked against; another one is refused.
pub const version = std.mem.trim(u8, @embedFile("../../.loom-version"), " \t\r\n");

/// A Loom code object from zig/kernels/loom/`dir`: gen.py writes its module, or loom-import-cxx makes it of `cxx`.
pub const Kernel = struct { name: []const u8, dir: []const u8, cxx: ?[]const u8 = null, inputs: []const []const u8 };

const k4_headers: []const []const u8 = &.{ "grouped_tile.h", "nvfp4_decode.h", "swiglu_exact.h" };
const sa_headers: []const []const u8 = &.{ "wmma_tile.h", "exp_exact.h" };
const dg_headers: []const []const u8 = &.{ "gemm_tile.h", "bf16_rows.h" };
const hc_headers: []const []const u8 = &.{ "gemm_tile.h", "bf16_rows.h", "hc_exact.h", "hc_tiles.h" };

/// The kernels in zig/src/hip/kernels.zig's Loom order.
pub const kernels = [_]Kernel{
    .{ .name = "mxfp8_gemv", .dir = "mxfp8_gemv", .inputs = &.{"gen_epilogue.py"} },
    .{ .name = "nvfp4_expert_gate_up", .dir = "nvfp4_grouped_expert", .cxx = "gate_up.cxx", .inputs = k4_headers },
    .{ .name = "nvfp4_expert_down", .dir = "nvfp4_grouped_expert", .cxx = "down.cxx", .inputs = k4_headers },
    .{ .name = "sparse_attend", .dir = "sparse_attention", .cxx = "attend.cxx", .inputs = sa_headers },
    .{ .name = "idx_scores", .dir = "sparse_attention", .cxx = "scores.cxx", .inputs = sa_headers },
    .{ .name = "idx_select", .dir = "sparse_attention", .cxx = "select.cxx", .inputs = sa_headers },
    .{ .name = "bf16_gemm_128x128", .dir = "dense_prompt_gemm", .cxx = "bf16_gemm.cxx", .inputs = dg_headers },
    .{ .name = "bf16_gemm_128x256", .dir = "dense_prompt_gemm", .cxx = "bf16_gemm.cxx", .inputs = dg_headers },
    .{ .name = "gdn_recur", .dir = "gdn_prompt", .cxx = "gdn_recur.cxx", .inputs = &.{"recur.h"} },
    .{ .name = "hc_norm", .dir = "dense_prompt_gemm", .cxx = "hc_norm.cxx", .inputs = hc_headers },
    .{ .name = "hc_down_wide", .dir = "dense_prompt_gemm", .cxx = "hc_readout.cxx", .inputs = hc_headers },
    .{ .name = "hc_down_narrow", .dir = "dense_prompt_gemm", .cxx = "hc_readout.cxx", .inputs = hc_headers },
    .{ .name = "hc_up_mix", .dir = "dense_prompt_gemm", .cxx = "hc_readout.cxx", .inputs = hc_headers },
};

/// A directory's loom-test module: gen_tests.py's output (with gen.py's module as its library), or C++ checks.
const Checks = struct { dir: []const u8, cxx: ?[]const u8 = null, inputs: []const []const u8 };

const checks = [_]Checks{
    .{ .dir = "mxfp8_gemv", .inputs = &.{ "gen.py", "gen_epilogue.py" } },
    .{ .dir = "nvfp4_grouped_expert", .cxx = "checks.cxx", .inputs = &.{ "grouped_tile.h", "nvfp4_decode.h", "swiglu_exact.h", "gate_up.cxx", "down.cxx" } },
    .{ .dir = "sparse_attention", .cxx = "checks.cxx", .inputs = &.{ "wmma_tile.h", "exp_exact.h", "oracle.h", "attend.cxx", "scores.cxx", "select.cxx" } },
    .{ .dir = "dense_prompt_gemm", .cxx = "checks.cxx", .inputs = &.{ "gemm_tile.h", "bf16_rows.h", "bf16_gemm.cxx" } },
    .{ .dir = "gdn_prompt", .cxx = "checks.cxx", .inputs = &.{ "recur.h", "oracle.h", "gdn_recur.cxx" } },
    .{ .dir = "dense_prompt_gemm", .cxx = "hc_checks.cxx", .inputs = &.{ "gemm_tile.h", "bf16_rows.h", "hc_exact.h", "hc_tiles.h", "hc_norm.cxx", "hc_readout.cxx" } },
};

/// The images' declarations of a build without Loom kernels.
pub const stub = std.fmt.comptimePrint(
    "pub const loom_arch = \"\";\npub const loom_images: [{d}][]align(8) const u8 = @splat(&.{{}});\n" ++
        "pub const loom_manifests: [{d}][]const u8 = @splat(\"\");\n",
    .{ kernels.len, kernels.len },
);

/// Loom's options, declared with HIP's on every target.
pub fn options(b: *std.Build) Options {
    return .{
        .compile = b.option([]const u8, "loom-compile", "hrx-loom's loom-compile (the version in .loom-version); omit to build without Loom kernels"),
        .target = b.option([]const u8, "loom-target", "Loom's compile target, family:selector (default amdgpu:gfx1151)") orelse "amdgpu:gfx1151",
    };
}

/// The architecture an amdgpu:<arch> target names, or null for another family.
pub fn arch(target: []const u8) ?[]const u8 {
    const prefix = "amdgpu:";
    if (!std.mem.startsWith(u8, target, prefix) or target.len == prefix.len) return null;
    return target[prefix.len..];
}

/// The hrx-loom install a build uses: loom-compile, loom-test, loom-import-cxx, and the Python that runs gen.py.
const Tools = struct { compile: []const u8, test_tool: []const u8, import: []const u8, python: []const u8, target: []const u8 };

/// The install for `hip_arch`, its version checked; null without -Dloom-compile or for another architecture.
fn tools(b: *std.Build, o: Options, hip_arch: []const u8) ?Tools {
    const tool = o.compile orelse return null;
    const want = arch(o.target) orelse std.debug.panic("-Dloom-target {s} is not amdgpu:<arch>", .{o.target});
    if (!std.mem.eql(u8, want, hip_arch)) return null;
    const resolved = b.findProgram(.{ .names = &.{tool} }) orelse tool;
    const found = installed(b, resolved) catch |err|
        std.debug.panic("-Dloom-compile {s}: cannot read its hrx-loom version ({t}); want {s}", .{ resolved, err, version });
    if (!std.mem.eql(u8, found, version))
        std.debug.panic("-Dloom-compile {s} is hrx-loom {s}; .loom-version pins {s}", .{ resolved, found, version });
    const bin = std.fs.path.dirname(resolved) orelse ".";
    return .{
        .compile = resolved,
        .test_tool = b.pathJoin(&.{ bin, "loom-test" }),
        .import = b.pathJoin(&.{ bin, "loom-import-cxx" }),
        .python = b.pathJoin(&.{ bin, "python3" }),
        .target = o.target,
    };
}

/// The Loom images' declarations for `files` (a kernels module's directory), built for `hip_arch`.
pub fn decls(b: *std.Build, files: *std.Build.Step.WriteFile, o: Options, hip_arch: []const u8) []const u8 {
    const t = tools(b, o, hip_arch) orelse return stub;
    var out: []const u8 = b.fmt("pub const loom_arch = \"{s}\";\n", .{hip_arch});
    var refs: []const u8 = "";
    var docs: []const u8 = "";
    for (kernels, 0..) |k, i| {
        const image = compile(b, t, k);
        const name = b.fmt("loom{d}.hsaco", .{i});
        _ = files.addCopyFile(image.hsaco, name);
        _ = files.addCopyFile(image.manifest, b.fmt("loom{d}.manifest.json", .{i}));
        out = b.fmt("{s}const l{d} align(8) = @embedFile(\"{s}\").*;\n", .{ out, i, name });
        refs = b.fmt("{s}&l{d}, ", .{ refs, i });
        docs = b.fmt("{s}@embedFile(\"loom{d}.manifest.json\"), ", .{ docs, i });
    }
    const images = b.fmt("pub const loom_images = [_][]align(8) const u8{{ {s}}};\n", .{refs});
    return b.fmt("{s}{s}pub const loom_manifests = [_][]const u8{{ {s}}};\n", .{ out, images, docs });
}

/// The Loom objects for chip `name` into `dir` as <name>/loom_<kernel>.hsaco: none without -Dloom-compile or elsewhere.
pub fn objects(b: *std.Build, dir: *std.Build.Step.WriteFile, o: Options, name: []const u8) void {
    const t = tools(b, o, name) orelse return;
    for (kernels) |k| _ = dir.addCopyFile(compile(b, t, k).hsaco, b.fmt("{s}/loom_{s}.hsaco", .{ name, k.name }));
}

/// loom-test over every kernel's generated checks on this GPU (-Dhip-arch), in the Loom framework's own runner.
pub fn testStep(b: *std.Build, o: Options, hip_arch: []const u8) void {
    const step = b.step("loom-test", "Every Loom kernel's loom-test cases on this GPU (-Dloom-compile, -Dhip-arch)");
    const t = tools(b, o, hip_arch) orelse return;
    for (checks) |c| {
        const run = b.addSystemCommand(&.{t.test_tool});
        if (c.cxx) |src| {
            run.addFileArg(import(b, t, c.dir, src, c.inputs, "checks.loom"));
        } else {
            run.addFileArg(generate(b, t, c.dir, "gen_tests.py", c.inputs, "tests.loom"));
            run.addPrefixedFileArg("--library=", generate(b, t, c.dir, "gen.py", c.inputs, b.fmt("{s}.loom", .{c.dir})));
        }
        run.addArgs(&.{ "--device=amdgpu", b.fmt("--target={s}", .{t.target}) });
        run.has_side_effects = true;
        step.dependOn(&run.step);
    }
}

pub const Image = struct { hsaco: std.Build.LazyPath, manifest: std.Build.LazyPath };

fn source(b: *std.Build, dir: []const u8, file: []const u8) std.Build.LazyPath {
    return b.path(b.fmt("zig/kernels/loom/{s}/{s}", .{ dir, file }));
}

/// A generated module (`script` writes it to stdout), regenerated when it or `inputs` change.
fn generate(b: *std.Build, t: Tools, dir: []const u8, script: []const u8, inputs: []const []const u8, out: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{t.python});
    run.addFileArg(source(b, dir, script));
    for (inputs) |f| run.addFileInput(source(b, dir, f));
    return run.captureStdOut(.{ .basename = out });
}

/// The module loom-import-cxx makes of C++ source `cxx`, its includes found in `dir` (`inputs`).
fn import(b: *std.Build, t: Tools, dir: []const u8, cxx: []const u8, inputs: []const []const u8, out: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{t.import});
    run.addPrefixedDirectoryArg("--I=", b.path(b.fmt("zig/kernels/loom/{s}", .{dir})));
    run.addFileArg(source(b, dir, cxx));
    for (inputs) |f| run.addFileInput(source(b, dir, f));
    return run.addPrefixedOutputFileArg("--output=", out);
}

/// A kernel's code object and its artifact manifest (details: each function's interface and workgroup size).
fn compile(b: *std.Build, t: Tools, k: Kernel) Image {
    const run = b.addSystemCommand(&.{t.compile});
    const module = b.fmt("{s}.loom", .{k.name});
    if (k.cxx) |src| {
        run.addFileArg(import(b, t, k.dir, src, k.inputs, module));
        run.addArg(b.fmt("--root=@{s}", .{k.name}));
    } else run.addFileArg(generate(b, t, k.dir, "gen.py", k.inputs, module));
    run.addArgs(&.{ b.fmt("--target={s}", .{t.target}), "--format=amdgpu-hsaco", "--artifact-manifest=details" });
    const hsaco = run.addPrefixedOutputFileArg("--output=", b.fmt("{s}.hsaco", .{k.name}));
    const manifest = run.addPrefixedOutputFileArg("--emit-artifact-manifest=", b.fmt("{s}.manifest.json", .{k.name}));
    return .{ .hsaco = hsaco, .manifest = manifest };
}

/// The hrx-loom version installed with `tool` (<prefix>/bin): <prefix>/lib/python*/site-packages/hrx_loom-V.dist-info.
fn installed(b: *std.Build, tool: []const u8) ![]const u8 {
    const io = b.graph.io;
    const real = try std.Io.Dir.realPathFileAbsoluteAlloc(io, tool, b.allocator);
    const prefix = std.fs.path.dirname(std.fs.path.dirname(real) orelse return error.NoPrefix) orelse return error.NoPrefix;
    const lib_path = b.pathJoin(&.{ prefix, "lib" });
    var lib = try std.Io.Dir.openDirAbsolute(io, lib_path, .{ .iterate = true });
    defer lib.close(io);
    var pythons = lib.iterate();
    while (try pythons.next(io)) |py| {
        if (!std.mem.startsWith(u8, py.name, "python")) continue;
        const site_path = b.pathJoin(&.{ lib_path, py.name, "site-packages" });
        var site = std.Io.Dir.openDirAbsolute(io, site_path, .{ .iterate = true }) catch continue;
        defer site.close(io);
        var it = site.iterate();
        while (try it.next(io)) |entry| {
            const name = entry.name;
            if (std.mem.startsWith(u8, name, "hrx_loom-") and std.mem.endsWith(u8, name, ".dist-info"))
                return b.dupe(name["hrx_loom-".len .. name.len - ".dist-info".len]);
        }
    }
    return error.NoDistInfo;
}

test "only amdgpu targets name an architecture" {
    try std.testing.expectEqualStrings("gfx1151", arch("amdgpu:gfx1151").?);
    try std.testing.expect(arch("spirv:vulkan1.3") == null);
    try std.testing.expect(arch("amdgpu:") == null);
}
