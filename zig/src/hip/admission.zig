//! What the HIP engine may serve: the CUDA path's memory rules (cuda_memory.zig) and the window and weights it reads.
const std = @import("std");
const memory = @import("cuda_memory");
const Context = @import("context.zig").Context;
const Allocator = std.mem.Allocator;

pub const MemInfo = memory.MemInfo;
pub const Pool = memory.Pool;
pub const meminfo = memory.meminfo;
pub const reserveBytes = memory.reserveBytes;
pub const limitBytes = memory.limitBytes;
pub const admit = memory.admit;
pub const counts = memory.counts;

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
pub fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const doc = readConfig(a, io, dir) orelse return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

fn readConfig(a: Allocator, io: std.Io, dir: []const u8) ?std.json.Value {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return null;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return null;
    return if (doc == .object) doc else null;
}

/// The rope scaling config.json declares that no HIP family computes (yarn, linear, dynamic, longrope...), null when none.
pub fn unsupportedRope(a: Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    return ropeScaling(readConfig(a, io, dir) orelse return null);
}

/// `rope_scaling` (HF's legacy key) or `rope_parameters` (transformers v5's), in text_config or at the root, whose
/// `rope_type` (or `type`) is not "default" ("mrope" is Qwen-VL's name for default frequencies over sections); an
/// object that names no type but carries a `factor` counts as scaled.
pub fn ropeScaling(doc: std.json.Value) ?[]const u8 {
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    for ([_]std.json.Value{ text, doc }) |o| for ([_][]const u8{ "rope_scaling", "rope_parameters" }) |key| {
        const r = o.object.get(key) orelse continue;
        if (r != .object) continue;
        const kind = r.object.get("rope_type") orelse r.object.get("type") orelse {
            if (r.object.get("factor") != null) return "unnamed";
            continue;
        };
        if (kind != .string) return "unnamed";
        if (!std.mem.eql(u8, kind.string, "default") and !std.mem.eql(u8, kind.string, "mrope")) return kind.string;
    };
    return null;
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
pub fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// The bytes of the checkpoint's safetensors and GGUF files: what its weights need on the device, near enough to refuse early.
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors") and !std.mem.endsWith(u8, e.name, ".gguf")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

/// What a plan may take on `ctx`'s GPU, as the native backends count it: free (host's if integrated), reserve, cap.
pub fn budget(gpa: Allocator, io: std.Io, ctx: Context) !Pool {
    const unified = (try ctx.attribute(.integrated)) != 0;
    const card = try ctx.memInfo();
    const text = if (unified) procText(gpa, io, "/proc/meminfo") else null;
    defer if (text) |t| gpa.free(t);
    const c = counts(unified, .{ .total = card.total, .available = card.free }, text) catch return error.HostMemoryUnavailable;
    const reserve = reserveBytes(getenv("TENSORFOLD_MEMORY_RESERVE_GIB"), c.total, unified) catch return error.BadReserve;
    const limit = limitBytes(getenv("TENSORFOLD_HIP_MEMORY_LIMIT_GB")) catch return error.BadLimit;
    return .{ .free = c.available, .total = c.total, .reserve = reserve, .limit = limit, .unified = unified };
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A /proc file's text, streamed: procfs reports size 0, and a positional read (readFileAlloc) stops there.
pub fn procText(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
}

test "rope scaling: yarn and the rest are named in either key and either tower, default and mrope pass" {
    const a = std.testing.allocator;
    const cases = .{
        .{ "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"default\",\"mrope_section\":[11,11,10]}}}", null },
        .{ "{\"rope_scaling\":null,\"max_position_embeddings\":32768}", null },
        .{ "{\"rope_scaling\":{\"type\":\"mrope\",\"mrope_section\":[16,24,24]}}", null },
        .{ "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":262144}}}", "yarn" },
        .{ "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0},\"text_config\":{}}", "yarn" },
        .{ "{\"text_config\":{\"rope_scaling\":{\"type\":\"linear\",\"factor\":2}}}", "linear" },
        .{ "{\"rope_scaling\":{\"factor\":8.0}}", "unnamed" },
    };
    inline for (cases) |c| {
        const doc = try std.json.parseFromSlice(std.json.Value, a, c[0], .{});
        defer doc.deinit();
        const got = ropeScaling(doc.value);
        if (@as(?[]const u8, c[1])) |want| try std.testing.expectEqualStrings(want, got.?) else try std.testing.expect(got == null);
    }
}
