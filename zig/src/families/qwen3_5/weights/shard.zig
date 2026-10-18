//! One safetensors file mapped read-only; tensors above core's max rank and unnamed dtypes stay out of its index.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const st = @import("core").safetensors;
const gguf = @import("core").gguf;
const gguf_names = @import("gguf_names.zig");

pub const Tensor = st.Tensor;

pub const Shard = struct {
    file: Io.File,
    map: Io.File.MemoryMap,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !Shard {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        if (gguf.is(map.memory)) {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            errdefer arena.deinit();
            const g = try parseGguf(arena.allocator(), map.memory);
            return .{ .file = file, .map = map, .data = g.data, .names = g.names, .arena = arena };
        }
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const names = try parse(arena.allocator(), map.memory[8..][0..header_len], len - 8 - header_len);
        return .{ .file = file, .map = map, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn close(s: *Shard, io: Io) void {
        s.map.destroy(io);
        s.file.close(io);
        s.arena.deinit();
        s.* = undefined;
    }

    /// Drops data bytes [begin, end) (their whole pages) from this mapping; the page cache keeps them, a read refaults.
    pub fn release(s: *const Shard, begin: usize, end: usize) void {
        if (builtin.os.tag != .linux or end <= begin) return;
        const page = std.heap.pageSize();
        const lo = std.mem.alignBackward(usize, s.data + begin, page);
        const hi = @min(std.mem.alignForward(usize, s.data + end, page), s.map.memory.len);
        if (hi <= lo) return;
        _ = std.os.linux.madvise(s.map.memory.ptr + lo, hi - lo, std.os.linux.MADV.DONTNEED);
    }

    pub fn get(s: *const Shard, name: []const u8) ?Tensor {
        const e = s.names.get(name) orelse return null;
        return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .bytes = s.map.memory[s.data + e.begin .. s.data + e.end] };
    }
};

/// A GGUF file's tensors under the MLX names (gguf_names.zig): float ones as their dtype and shape (ggml's dims
/// reversed), a quantized projection as its rows' blocks, u8 [rows, row bytes], under `<name>.ggml.<type>`.
fn parseGguf(arena: std.mem.Allocator, bytes: []const u8) !struct { data: usize, names: st.Header } {
    const f = gguf.parse(arena, bytes) catch return error.BadGguf;
    if (!std.mem.eql(u8, f.string("general.architecture") orelse "", "qwen35")) return error.UnsupportedGgufArchitecture;
    const blocks: u64 = @intCast(f.int("qwen35.block_count") orelse return error.BadGguf);
    const layers = blocks - @as(u64, @intCast(f.int("qwen35.nextn_predict_layers") orelse 0));
    var out: st.Header = .empty;
    for (f.tensors) |t| {
        const name = try gguf_names.mlx(arena, t.name, layers) orelse continue;
        const projection = std.mem.endsWith(u8, t.name, ".weight") and !std.mem.endsWith(u8, name, ".weight");
        const dtype: ?st.DType = switch (t.type) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            else => null,
        };
        var e: st.Entry = .{ .dtype = dtype orelse .u8, .rank = @intCast(t.rank), .shape = @splat(1), .begin = @intCast(t.offset), .end = @intCast(t.offset + t.bytes) };
        if (dtype != null) {
            for (0..t.rank) |i| e.shape[i] = @intCast(t.dims[t.rank - 1 - i]);
        } else {
            e.rank = 2;
            e.shape[0] = @intCast(t.rows());
            e.shape[1] = @intCast(t.bytes / t.rows());
        }
        const key = if (dtype == null)
            try std.fmt.allocPrint(arena, "{s}.ggml.{s}", .{ name, gguf.typeName(t.type) })
        else if (projection) try std.fmt.allocPrint(arena, "{s}.weight", .{name}) else name;
        try out.put(arena, key, e);
    }
    return .{ .data = @intCast(f.data), .names = out };
}

/// core's `parseHeader` without its refusals of rank and dtype: dimensions past the max rank fold into the last kept one
/// (ciru's [E, N/16, K/32, 2, 16, 16] codes read as [E, N/16, K/32, 2, 256]); sizes and ranges are still checked.
fn parse(arena: std.mem.Allocator, json: []const u8, data_len: usize) !st.Header {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    var out: st.Header = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const o = kv.value_ptr.object;
        const dtype = st.DType.parse(o.get("dtype").?.string) orelse continue;
        const shape = o.get("shape").?.array.items;
        var e: st.Entry = .{ .dtype = dtype, .rank = @intCast(@min(shape.len, st.max_rank)), .shape = @splat(1), .begin = 0, .end = 0 };
        var n: usize = dtype.size();
        for (shape, 0..) |d, i| {
            const at = @min(i, st.max_rank - 1);
            const v: usize = @intCast(d.integer);
            e.shape[at] = if (i > at) e.shape[at] * v else v;
            n *= v;
        }
        const offs = o.get("data_offsets").?.array.items;
        e.begin = @intCast(offs[0].integer);
        e.end = @intCast(offs[1].integer);
        if (e.end < e.begin or e.end - e.begin != n or e.end > data_len) return error.BadSafetensors;
        try out.put(arena, kv.key_ptr.*, e);
    }
    return out;
}

test "a tensor above the max rank folds its last dimensions, an unknown dtype is left out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a": {"dtype": "U32", "shape": [2], "data_offsets": [0, 8]},
        \\ "patch": {"dtype": "BF16", "shape": [1, 1, 1, 1, 1, 1], "data_offsets": [8, 10]},
        \\ "q9": {"dtype": "Q9_UNKNOWN", "shape": [2], "data_offsets": [10, 12]}}
    ;
    const h = try parse(arena.allocator(), json, 12);
    try std.testing.expectEqual(@as(usize, 2), h.count());
    try std.testing.expect(h.contains("a"));
    try std.testing.expectEqual(@as(u8, 5), h.get("patch").?.rank);
}

test "released pages read back the same bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const n = 3 * 4096;
    const header = "{\"t\":{\"dtype\":\"BF16\",\"shape\":[6144],\"data_offsets\":[0,12288]}}";
    var bytes: [8 + header.len + n]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    for (bytes[8 + header.len ..], 0..) |*b, i| b.* = @truncate(i *% 31);
    try tmp.dir.writeFile(io, .{ .sub_path = "s.safetensors", .data = &bytes });
    const path = try tmp.dir.realPathFileAlloc(io, "s.safetensors", std.testing.allocator);
    defer std.testing.allocator.free(path);
    var s = try Shard.open(std.testing.allocator, io, path);
    defer s.close(io);
    const before = try std.testing.allocator.dupe(u8, s.get("t").?.bytes);
    defer std.testing.allocator.free(before);
    s.release(0, n);
    try std.testing.expectEqualSlices(u8, before, s.get("t").?.bytes);
}
