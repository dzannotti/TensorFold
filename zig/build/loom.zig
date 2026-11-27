//! Loom kernels (zig/kernels/loom/<kernel>/) compiled ahead of time by -Dloom-compile; absent without it (hipcc's run).
const std = @import("std");

pub const Options = struct { compile: ?[]const u8, target: []const u8 };

/// The hrx-loom release the sources are written and checked against; another one is refused.
pub const version = std.mem.trim(u8, @embedFile("../../.loom-version"), " \t\r\n");

/// A kernel's directory under zig/kernels/loom: gen.py writes <name>.loom, gen_tests.py its loom-test module.
pub const Kernel = struct { name: []const u8 };

/// The kernels in zig/src/hip/kernels.zig's Loom order.
pub const kernels = [_]Kernel{.{ .name = "mxfp8_gemv" }};

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

/// The hrx-loom install a build uses: its loom-compile and loom-test, and the Python beside them that runs gen.py.
const Tools = struct { compile: []const u8, test_tool: []const u8, python: []const u8, target: []const u8 };

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
    for (kernels) |k| {
        const run = b.addSystemCommand(&.{t.test_tool});
        run.addFileArg(generate(b, t, k, "gen_tests.py", "tests.loom"));
        run.addPrefixedFileArg("--library=", generate(b, t, k, "gen.py", b.fmt("{s}.loom", .{k.name})));
        run.addArgs(&.{ "--device=amdgpu", b.fmt("--target={s}", .{t.target}) });
        run.has_side_effects = true;
        step.dependOn(&run.step);
    }
}

pub const Image = struct { hsaco: std.Build.LazyPath, manifest: std.Build.LazyPath };

/// A kernel's generated module (`script` writes it to stdout), regenerated when its generators change.
fn generate(b: *std.Build, t: Tools, k: Kernel, script: []const u8, out: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{t.python});
    run.addFileArg(b.path(b.fmt("zig/kernels/loom/{s}/{s}", .{ k.name, script })));
    for ([_][]const u8{ "gen.py", "gen_epilogue.py" }) |f| run.addFileInput(b.path(b.fmt("zig/kernels/loom/{s}/{s}", .{ k.name, f })));
    return run.captureStdOut(.{ .basename = out });
}

/// A kernel's code object and its artifact manifest (details: each function's interface and workgroup size).
fn compile(b: *std.Build, t: Tools, k: Kernel) Image {
    const run = b.addSystemCommand(&.{t.compile});
    run.addFileArg(generate(b, t, k, "gen.py", b.fmt("{s}.loom", .{k.name})));
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
