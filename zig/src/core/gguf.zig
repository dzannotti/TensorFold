//! GGUF v3 files as llama.cpp writes them (ggml-org/llama.cpp ggml/include/gguf.h, ggml/src/gguf.cpp): the metadata
//! a loader reads, every tensor's ggml type, dimensions and offset, and the aligned data section. Read-only; every
//! length and offset is checked against the file, which is untrusted.

const std = @import("std");

/// ggml's type numbers (ggml/include/ggml.h `enum ggml_type`); `_` is a type this reader does not name.
pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    _,

    /// Elements and bytes of one block (ggml-common.h's block structs; 1 element for the plain types), null when unnamed.
    pub fn block(t: Type) ?struct { elems: u32, bytes: u32 } {
        return switch (t) {
            .f32, .i32 => .{ .elems = 1, .bytes = 4 },
            .f16, .bf16, .i16 => .{ .elems = 1, .bytes = 2 },
            .i8 => .{ .elems = 1, .bytes = 1 },
            .f64, .i64 => .{ .elems = 1, .bytes = 8 },
            .q4_0 => .{ .elems = 32, .bytes = 18 },
            .q4_1 => .{ .elems = 32, .bytes = 20 },
            .q5_0 => .{ .elems = 32, .bytes = 22 },
            .q5_1 => .{ .elems = 32, .bytes = 24 },
            .q8_0 => .{ .elems = 32, .bytes = 34 },
            .q8_1 => .{ .elems = 32, .bytes = 36 },
            .q2_k => .{ .elems = 256, .bytes = 84 },
            .q3_k => .{ .elems = 256, .bytes = 110 },
            .q4_k => .{ .elems = 256, .bytes = 144 },
            .q5_k => .{ .elems = 256, .bytes = 176 },
            .q6_k => .{ .elems = 256, .bytes = 210 },
            .q8_k => .{ .elems = 256, .bytes = 292 },
            .iq2_xxs => .{ .elems = 256, .bytes = 66 },
            .iq2_xs => .{ .elems = 256, .bytes = 74 },
            .iq3_xxs => .{ .elems = 256, .bytes = 98 },
            .iq1_s => .{ .elems = 256, .bytes = 50 },
            .iq4_nl => .{ .elems = 32, .bytes = 18 },
            .iq3_s => .{ .elems = 256, .bytes = 110 },
            .iq2_s => .{ .elems = 256, .bytes = 82 },
            .iq4_xs => .{ .elems = 256, .bytes = 136 },
            .iq1_m => .{ .elems = 256, .bytes = 56 },
            _ => null,
        };
    }

    /// Bytes of a row of `k` elements, null when `k` is not whole blocks or the type is unnamed.
    pub fn rowBytes(t: Type, k: u64) ?u64 {
        const b = t.block() orelse return null;
        if (k % b.elems != 0) return null;
        return k / b.elems * b.bytes;
    }
};

/// A metadata value: integers (and bools) widened, a string as stored, an array by its element type and length (its
/// contents are skipped; `ints` holds the first ones of an integer array).
pub const Value = union(enum) {
    int: i64,
    float: f64,
    string: []const u8,
    array: struct { type: u32, len: u64, ints: []const i64 },
};

pub const Info = struct {
    name: []const u8,
    rank: u32,
    /// ne[0] (the contiguous one, a row's elements) first, as ggml numbers them.
    dims: [4]u64,
    type: Type,
    /// From the data section's start.
    offset: u64,
    bytes: u64,

    /// Rows: every dimension past the first.
    pub fn rows(i: Info) u64 {
        var n: u64 = 1;
        for (i.dims[1..i.rank]) |d| n *= d;
        return n;
    }
};

pub const File = struct {
    version: u32,
    kv: std.StringArrayHashMapUnmanaged(Value),
    tensors: []Info,
    /// The data section's offset in the file.
    data: u64,
    alignment: u64,

    pub fn get(f: *const File, key: []const u8) ?Value {
        return f.kv.get(key);
    }

    pub fn int(f: *const File, key: []const u8) ?i64 {
        const v = f.kv.get(key) orelse return null;
        return if (v == .int) v.int else null;
    }

    pub fn string(f: *const File, key: []const u8) ?[]const u8 {
        const v = f.kv.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }
};

pub const Error = error{BadGguf} || std.mem.Allocator.Error;

/// Whether `bytes` start as a GGUF file.
pub fn is(bytes: []const u8) bool {
    return bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "GGUF");
}

const Reader = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: u64) Error![]const u8 {
        if (n > r.b.len - r.at) return error.BadGguf;
        const s = r.b[r.at..][0..@intCast(n)];
        r.at += @intCast(n);
        return s;
    }

    fn int(r: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn str(r: *Reader) Error![]const u8 {
        return r.take(try r.int(u64));
    }

    /// A value of gguf type `t` (gguf.h `enum gguf_type`).
    fn value(r: *Reader, a: std.mem.Allocator, t: u32, depth: u8) Error!Value {
        return switch (t) {
            0 => .{ .int = try r.int(u8) },
            1 => .{ .int = try r.int(i8) },
            2 => .{ .int = try r.int(u16) },
            3 => .{ .int = try r.int(i16) },
            4 => .{ .int = try r.int(u32) },
            5 => .{ .int = try r.int(i32) },
            6 => .{ .float = @as(f32, @bitCast(try r.int(u32))) },
            7 => .{ .int = @intFromBool((try r.int(u8)) != 0) },
            8 => .{ .string = try r.str() },
            9 => {
                if (depth > 0) return error.BadGguf;
                const et = try r.int(u32);
                const n = try r.int(u64);
                if (n > r.b.len) return error.BadGguf;
                var ints: std.ArrayList(i64) = .empty;
                for (0..@intCast(n)) |i| {
                    const v = try r.value(a, et, depth + 1);
                    if (v == .int and i < 64) try ints.append(a, v.int);
                }
                return .{ .array = .{ .type = et, .len = n, .ints = ints.items } };
            },
            10 => .{ .int = @bitCast(try r.int(u64)) },
            11 => .{ .int = try r.int(i64) },
            12 => .{ .float = @bitCast(try r.int(u64)) },
            else => error.BadGguf,
        };
    }
};

/// The header of the GGUF file `bytes` (its names and strings point into `bytes`); every tensor must lie in the file.
pub fn parse(a: std.mem.Allocator, bytes: []const u8) Error!File {
    if (!is(bytes)) return error.BadGguf;
    var r: Reader = .{ .b = bytes, .at = 4 };
    const version = try r.int(u32);
    if (version != 2 and version != 3) return error.BadGguf;
    const n_tensors = try r.int(u64);
    const n_kv = try r.int(u64);
    if (n_tensors > bytes.len or n_kv > bytes.len) return error.BadGguf;
    var f: File = .{ .version = version, .kv = .empty, .tensors = &.{}, .data = 0, .alignment = 32 };
    for (0..@intCast(n_kv)) |_| {
        const key = try r.str();
        const t = try r.int(u32);
        try f.kv.put(a, key, try r.value(a, t, 0));
    }
    if (f.int("general.alignment")) |al| {
        if (al <= 0 or !std.math.isPowerOfTwo(al)) return error.BadGguf;
        f.alignment = @intCast(al);
    }
    const tensors = try a.alloc(Info, @intCast(n_tensors));
    for (tensors) |*t| {
        t.name = try r.str();
        t.rank = try r.int(u32);
        if (t.rank == 0 or t.rank > 4) return error.BadGguf;
        t.dims = @splat(1);
        for (t.dims[0..t.rank]) |*d| d.* = try r.int(u64);
        t.type = @enumFromInt(try r.int(u32));
        t.offset = try r.int(u64);
    }
    f.data = std.mem.alignForward(u64, r.at, f.alignment);
    if (f.data > bytes.len) return error.BadGguf;
    const room = bytes.len - f.data;
    for (tensors) |*t| {
        const row = t.type.rowBytes(t.dims[0]) orelse return error.BadGguf;
        t.bytes = std.math.mul(u64, row, t.rows()) catch return error.BadGguf;
        if (t.offset % f.alignment != 0 or t.offset > room or t.bytes > room - t.offset) return error.BadGguf;
    }
    f.tensors = tensors;
    return f;
}

/// A GGUF file in memory, for the tests: `kv` string pairs and `tensors` (name, type, dims, data) laid out as gguf.cpp
/// writes them.
pub fn write(a: std.mem.Allocator, kv: []const [2][]const u8, tensors: []const struct { name: []const u8, type: Type, dims: []const u64, data: []const u8 }) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const w = struct {
        fn int(o: *std.ArrayList(u8), al: std.mem.Allocator, comptime T: type, v: T) !void {
            var buf: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &buf, v, .little);
            try o.appendSlice(al, &buf);
        }
        fn str(o: *std.ArrayList(u8), al: std.mem.Allocator, s: []const u8) !void {
            try int(o, al, u64, s.len);
            try o.appendSlice(al, s);
        }
    };
    try out.appendSlice(a, "GGUF");
    try w.int(&out, a, u32, 3);
    try w.int(&out, a, u64, tensors.len);
    try w.int(&out, a, u64, kv.len);
    for (kv) |p| {
        try w.str(&out, a, p[0]);
        try w.int(&out, a, u32, 8);
        try w.str(&out, a, p[1]);
    }
    var off: u64 = 0;
    for (tensors) |t| {
        try w.str(&out, a, t.name);
        try w.int(&out, a, u32, @intCast(t.dims.len));
        for (t.dims) |d| try w.int(&out, a, u64, d);
        try w.int(&out, a, u32, @intFromEnum(t.type));
        try w.int(&out, a, u64, off);
        off = std.mem.alignForward(u64, off + t.data.len, 32);
    }
    try out.appendNTimes(a, 0, std.mem.alignForward(usize, out.items.len, 32) - out.items.len);
    for (tensors) |t| {
        try out.appendSlice(a, t.data);
        try out.appendNTimes(a, 0, std.mem.alignForward(usize, t.data.len, 32) - t.data.len);
    }
    return out.toOwnedSlice(a);
}

test "a written file parses back: metadata, types, dims, offsets; a tensor past the end is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q: [2 * 144]u8 = @splat(7);
    const v: [16]u8 = @splat(1);
    const bytes = try write(a, &.{.{ "general.architecture", "qwen35" }}, &.{
        .{ .name = "blk.0.ffn_up.weight", .type = .q4_k, .dims = &.{ 256, 2 }, .data = &q },
        .{ .name = "blk.0.attn_norm.weight", .type = .f32, .dims = &.{4}, .data = &v },
    });
    const f = try parse(a, bytes);
    try std.testing.expectEqualStrings("qwen35", f.string("general.architecture").?);
    try std.testing.expectEqual(@as(usize, 2), f.tensors.len);
    try std.testing.expectEqual(Type.q4_k, f.tensors[0].type);
    try std.testing.expectEqual(@as(u64, 2), f.tensors[0].rows());
    try std.testing.expectEqual(@as(u64, 288), f.tensors[0].bytes);
    try std.testing.expectEqualSlices(u8, &q, bytes[@intCast(f.data + f.tensors[0].offset)..][0..288]);
    try std.testing.expectEqual(@as(u64, 288), f.tensors[1].offset);
    try std.testing.expectError(error.BadGguf, parse(a, bytes[0 .. bytes.len - 20]));
    try std.testing.expectError(error.BadGguf, parse(a, bytes[0..40]));
}

/// A type's name as ggml spells it lower-case (`q4_k`), "unknown" for one this reader does not name.
pub fn typeName(t: Type) []const u8 {
    return std.enums.tagName(Type, t) orelse "unknown";
}
