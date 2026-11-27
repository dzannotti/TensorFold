//! What a hipcc code object says about each kernel (AMDGPU metadata note): explicit argument bytes, registers and LDS.
const std = @import("std");

pub const Arg = struct { offset: u32, size: u32 };

pub const Kernel = struct {
    name: []const u8 = "",
    /// The explicit (non-hidden) arguments in declaration order; hidden ones follow and the runtime writes those.
    args: []const Arg = &.{},
    /// Bytes from offset 0 to the end of the last explicit argument.
    explicit: u32 = 0,
    lds: u32 = 0,
    private: u32 = 0,
    vgprs: u32 = 0,
    sgprs: u32 = 0,
    bound: u32 = 0,
    wave: u32 = 0,
};

pub const Error = error{ NotACodeObject, BadMetadata, OutOfMemory };
const nt_amdgpu_metadata = 32;

fn int(comptime T: type, b: []const u8, at: usize) Error!T {
    if (at + @sizeOf(T) > b.len) return error.NotACodeObject;
    return std.mem.readInt(T, b[at..][0..@sizeOf(T)], .little);
}

const bundle_magic = "__CLANG_OFFLOAD_BUNDLE__";

/// The device ELF behind a bare code-object pointer: the ELF itself, or a clang offload bundle's amdgcn entry.
pub fn elf(p: [*]const u8) Error![]const u8 {
    if (std.mem.eql(u8, p[0..4], "\x7fELF")) return p[0..try elfLength(p)];
    if (!std.mem.eql(u8, p[0..bundle_magic.len], bundle_magic)) return error.NotACodeObject;
    const head = p[0..32];
    const n = try int(u64, head, 24);
    var at: usize = 32;
    for (0..n) |_| {
        const e = p[at..][0..24];
        const off = try int(u64, e, 0);
        const len = try int(u64, e, 16);
        const id = p[at + 24 ..][0..len];
        at += 24 + len;
        if (std.mem.indexOf(u8, id, "amdgcn-amd-amdhsa") == null) continue;
        const body = p + off;
        if (!std.mem.eql(u8, body[0..4], "\x7fELF")) return error.NotACodeObject;
        return body[0..try int(u64, e, 8)];
    }
    return error.NotACodeObject;
}

fn elfLength(p: [*]const u8) Error!usize {
    const head = p[0..64];
    if (head[4] != 2) return error.NotACodeObject;
    const shoff = try int(u64, head, 0x28);
    const shentsize = try int(u16, head, 0x3a);
    const shnum = try int(u16, head, 0x3c);
    var end: usize = shoff + @as(usize, shentsize) * shnum;
    const sections = p[shoff..][0 .. @as(usize, shentsize) * shnum];
    for (0..shnum) |i| {
        const sh = i * shentsize;
        if (try int(u32, sections, sh + 4) == 8) continue; // SHT_NOBITS occupies no file bytes
        end = @max(end, try int(u64, sections, sh + 24) + try int(u64, sections, sh + 32));
    }
    return end;
}

fn note(e: []const u8) Error![]const u8 {
    const shoff = try int(u64, e, 0x28);
    const shnum = try int(u16, e, 0x3c);
    for (0..shnum) |i| {
        const sh = shoff + i * 64;
        if (try int(u32, e, sh + 4) != 7) continue; // SHT_NOTE
        const off = try int(u64, e, sh + 24);
        const size = try int(u64, e, sh + 32);
        var at = off;
        while (at + 12 <= off + size) {
            const namesz = try int(u32, e, at);
            const descsz = try int(u32, e, at + 4);
            const name = at + 12;
            const desc = name + std.mem.alignForward(usize, namesz, 4);
            if (desc + descsz > e.len) return error.NotACodeObject;
            if (try int(u32, e, at + 8) == nt_amdgpu_metadata and std.mem.startsWith(u8, e[name..][0..namesz], "AMDGPU"))
                return e[desc..][0..descsz];
            at = desc + std.mem.alignForward(usize, descsz, 4);
        }
    }
    return error.BadMetadata;
}

const Value = union(enum) { map: usize, array: usize, str: []const u8, uint: u64, other };

const Reader = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: usize) Error![]const u8 {
        if (r.at + n > r.b.len) return error.BadMetadata;
        defer r.at += n;
        return r.b[r.at..][0..n];
    }

    fn be(r: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .big);
    }

    fn next(r: *Reader) Error!Value {
        const t = (try r.take(1))[0];
        return switch (t) {
            0x00...0x7f => .{ .uint = t },
            0x80...0x8f => .{ .map = t & 0x0f },
            0x90...0x9f => .{ .array = t & 0x0f },
            0xa0...0xbf => .{ .str = try r.take(t & 0x1f) },
            0xc0, 0xc2, 0xc3, 0xe0...0xff => .other,
            0xc4, 0xd9 => r.blob(try r.be(u8), t),
            0xc5, 0xda => r.blob(try r.be(u16), t),
            0xc6, 0xdb => r.blob(try r.be(u32), t),
            0xca, 0xd2 => r.skip(4),
            0xcb, 0xd3 => r.skip(8),
            0xcc => .{ .uint = try r.be(u8) },
            0xcd => .{ .uint = try r.be(u16) },
            0xce => .{ .uint = try r.be(u32) },
            0xcf => .{ .uint = try r.be(u64) },
            0xd0 => r.skip(1),
            0xd1 => r.skip(2),
            0xdc => .{ .array = try r.be(u16) },
            0xdd => .{ .array = try r.be(u32) },
            0xde => .{ .map = try r.be(u16) },
            0xdf => .{ .map = try r.be(u32) },
            else => error.BadMetadata,
        };
    }

    fn blob(r: *Reader, n: usize, t: u8) Error!Value {
        const s = try r.take(n);
        return if (t >= 0xd9) .{ .str = s } else .other;
    }

    fn skip(r: *Reader, n: usize) Error!Value {
        _ = try r.take(n);
        return .other;
    }

    fn pass(r: *Reader) Error!void {
        var left: usize = 1;
        while (left > 0) : (left -= 1) switch (try r.next()) {
            .map => |n| left += 2 * n,
            .array => |n| left += n,
            else => {},
        };
    }

    fn str(r: *Reader) Error![]const u8 {
        return switch (try r.next()) {
            .str => |s| s,
            else => error.BadMetadata,
        };
    }

    fn uint(r: *Reader) Error!u32 {
        return switch (try r.next()) {
            .uint => |v| std.math.cast(u32, v) orelse error.BadMetadata,
            else => error.BadMetadata,
        };
    }
};

/// Every kernel in the code object; names point into `elf`.
pub fn kernels(a: std.mem.Allocator, image: []const u8) Error![]Kernel {
    var r = Reader{ .b = try note(image) };
    var out: std.ArrayList(Kernel) = .empty;
    errdefer out.deinit(a);
    const top = switch (try r.next()) {
        .map => |n| n,
        else => return error.BadMetadata,
    };
    for (0..top) |_| {
        if (!std.mem.eql(u8, try r.str(), "amdhsa.kernels")) {
            try r.pass();
            continue;
        }
        const n = switch (try r.next()) {
            .array => |n| n,
            else => return error.BadMetadata,
        };
        for (0..n) |_| try out.append(a, try kernel(a, &r));
    }
    return out.toOwnedSlice(a);
}

fn kernel(a: std.mem.Allocator, r: *Reader) Error!Kernel {
    const n = switch (try r.next()) {
        .map => |n| n,
        else => return error.BadMetadata,
    };
    var k: Kernel = .{};
    const fields = .{
        .{ ".group_segment_fixed_size", "lds" },   .{ ".private_segment_fixed_size", "private" },
        .{ ".vgpr_count", "vgprs" },               .{ ".sgpr_count", "sgprs" },
        .{ ".max_flat_workgroup_size", "bound" }, .{ ".wavefront_size", "wave" },
    };
    outer: for (0..n) |_| {
        const key = try r.str();
        if (std.mem.eql(u8, key, ".name")) {
            k.name = try r.str();
            continue;
        }
        if (std.mem.eql(u8, key, ".args")) {
            k.args = try args(a, r);
            for (k.args) |x| k.explicit = @max(k.explicit, x.offset + x.size);
            continue;
        }
        inline for (fields) |f| if (std.mem.eql(u8, key, f[0])) {
            @field(k, f[1]) = try r.uint();
            continue :outer;
        };
        try r.pass();
    }
    return k;
}

fn args(a: std.mem.Allocator, r: *Reader) Error![]const Arg {
    const n = switch (try r.next()) {
        .array => |n| n,
        else => return error.BadMetadata,
    };
    var out: std.ArrayList(Arg) = .empty;
    errdefer out.deinit(a);
    for (0..n) |_| {
        const m = switch (try r.next()) {
            .map => |m| m,
            else => return error.BadMetadata,
        };
        var offset: u32 = 0;
        var size: u32 = 0;
        var hidden = false;
        for (0..m) |_| {
            const key = try r.str();
            if (std.mem.eql(u8, key, ".offset")) {
                offset = try r.uint();
            } else if (std.mem.eql(u8, key, ".size")) {
                size = try r.uint();
            } else if (std.mem.eql(u8, key, ".value_kind")) {
                hidden = std.mem.startsWith(u8, try r.str(), "hidden_");
            } else try r.pass();
        }
        if (!hidden) try out.append(a, .{ .offset = offset, .size = size });
    }
    return out.toOwnedSlice(a);
}

pub fn free(a: std.mem.Allocator, list: []const Kernel) void {
    for (list) |k| a.free(k.args);
    a.free(list);
}

pub fn find(list: []const Kernel, name: []const u8) ?Kernel {
    for (list) |k| if (std.mem.eql(u8, k.name, name)) return k;
    return null;
}
