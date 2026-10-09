//! A captured set of Triton binaries (aot.json + cubins/, or hsaco/ for AMD): each launch picks the variant Triton itself would have picked.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const triton = @import("triton.zig");

/// `range32`: a pointer built with AMD's `tt.pointer_range = 32` (buffer loads and stores, 32-bit offsets), which a
/// launch takes only when its tensor lies within 2 GiB of the pointer (Triton's JIT: storage <= 2^31 - 1 bytes).
const ParamJson = struct { name: []const u8, type: []const u8, div16: bool, nospec: bool, range32: bool = false };
const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
const KernelJson = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    warp_size: u32 = 32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []ParamJson,
    consts: std.json.ArrayHashMap(ConstJson),
};
const SetJson = struct { kernels: []KernelJson, bin_dir: []const u8 = "cubins", bin_ext: []const u8 = "cubin" };

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with (Python's keyword arguments): ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

const Variant = struct { spec: KernelJson, kernel: triton.Kernel };

/// Bytes a buffer-op pointer may address (Triton's `is_within_2gb`).
pub const max_range: u64 = (1 << 31) - 1;

/// The driver's allocation lookups `Set.small` uses (cuMemGetAddressRange / hipMemGetAddressRange and the VMM handle
/// retain that tells a mapped chunk from a whole allocation).
const Ranges = struct {
    range: *const fn (*u64, *usize, u64) callconv(.c) c_int,
    retain: *const fn (*u64, u64) callconv(.c) c_int,
    release: *const fn (u64) callconv(.c) c_int,

    fn init(d: *const Driver) ?Ranges {
        var lib = d.lib;
        const hip = @import("driver.zig").hip;
        return .{
            .range = lib.lookup(*const fn (*u64, *usize, u64) callconv(.c) c_int, if (hip) "hipMemGetAddressRange" else "cuMemGetAddressRange_v2") orelse return null,
            .retain = lib.lookup(*const fn (*u64, u64) callconv(.c) c_int, if (hip) "hipMemRetainAllocationHandle" else "cuMemRetainAllocationHandle") orelse return null,
            .release = lib.lookup(*const fn (u64) callconv(.c) c_int, if (hip) "hipMemRelease" else "cuMemRelease") orelse return null,
        };
    }
};

pub const Set = struct {
    parsed: std.json.Parsed(SetJson),
    variants: []Variant,
    gpa: std.mem.Allocator,
    /// the set holds `range32` variants and the driver can size allocations (else no pointer is ever `small`)
    ranges: ?Ranges = null,

    /// Loads every binary listed in `dir`/aot.json (cubins, or AMD code objects) into its own module.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, device: abi.Device, dir: []const u8) !Set {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        const variants = try gpa.alloc(Variant, parsed.value.kernels.len);
        var n: usize = 0;
        var ranged = false;
        errdefer {
            for (variants[0..n]) |*v| v.kernel.unload();
            gpa.free(variants);
        }
        for (parsed.value.kernels) |k| {
            if (k.global_scratch != 0 or k.profile_scratch != 0) return error.ScratchUnsupported;
            const file = try std.fmt.allocPrint(gpa, "{s}/{s}/{s}.{s}", .{ dir, parsed.value.bin_dir, k.hash, parsed.value.bin_ext });
            defer gpa.free(file);
            const cubin = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(cubin);
            const name_z = try gpa.dupeSentinel(u8, k.name, 0);
            defer gpa.free(name_z);
            const meta: triton.Meta = .{ .name = k.name, .num_warps = k.num_warps, .warp_size = k.warp_size, .num_ctas = k.num_ctas, .shared = k.shared, .launch_pdl = k.pdl };
            variants[n] = .{ .spec = k, .kernel = try triton.Kernel.load(d, device, cubin, meta, name_z) };
            n += 1;
            for (k.params) |p| ranged = ranged or p.range32;
        }
        return .{ .parsed = parsed, .variants = variants, .gpa = gpa, .ranges = if (ranged) Ranges.init(d) else null };
    }

    pub fn deinit(self: *Set) void {
        for (self.variants) |*v| v.kernel.unload();
        self.gpa.free(self.variants);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// Whether a buffer-op variant may take the pointer: its allocation ends within `max_range` bytes of it. A VMM
    /// mapping answers for one physical chunk, not its reservation, so VMM memory (the KV caches) never counts.
    /// ponytail: two driver lookups a pointer a launch (~140 ns on gfx1151); graphs replay without them.
    pub fn small(self: *const Set, addr: u64) bool {
        const r = self.ranges orelse return false;
        var base: u64 = 0;
        var size: usize = 0;
        if (r.range(&base, &size, addr) != 0 or addr < base or base + size - addr > max_range) return false;
        var h: u64 = 0;
        if (r.retain(&h, addr) == 0) {
            _ = r.release(h);
            return false;
        }
        return true;
    }

    /// The variant of `function` compiled for these constexprs and these arguments' specialization.
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*const Variant {
        const all_small = allSmall(self, args);
        for (self.variants) |*v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            if (matches(v.spec, args, consts, all_small)) return v;
        }
        std.log.err("no captured Triton variant of {s} for this launch:", .{function});
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const c = (v.spec.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, scratch null.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        const v = try self.find(function, args, consts);
        var packed_args: launch.Args = .{};
        for (v.spec.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = grid[0], .y = grid[1], .z = grid[2] }, stream, &packed_args, .{}, &.{});
    }
};

/// A set's aot.json alone, no GPU: whether a launch would find a variant (dry coverage checks of a built set).
pub const Specs = struct {
    parsed: std.json.Parsed(SetJson),

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Specs {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) };
    }

    pub fn deinit(self: *Specs) void {
        self.parsed.deinit();
    }

    pub fn has(self: *const Specs, function: []const u8, args: []const Arg, consts: []const Const) bool {
        for (self.parsed.value.kernels) |k| {
            if (std.mem.eql(u8, k.@"fn", function) and matches(k, args, consts, null)) return true;
        }
        return false;
    }
};

/// Every pointer argument of a launch `small` (the form a variant's pointers are built in).
pub fn allSmall(set: *const Set, args: []const Arg) bool {
    if (set.ranges == null) return false;
    for (args) |a| switch (a.value) {
        .ptr => |x| if (!set.small(x.addr)) return false,
        else => {},
    };
    return true;
}

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

/// `small` null (a dry check): either pointer form matches; else a `range32` param exactly when every pointer of the
/// launch is `small` (allSmall: a variant is built with all its pointers in one form).
fn matches(k: KernelJson, args: []const Arg, consts: []const Const, small: ?bool) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
                if (small) |all| if (p.range32 != all) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}
