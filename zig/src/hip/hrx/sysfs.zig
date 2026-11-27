//! The GPU's facts from the kernel (KFD topology and the amdgpu PCI node), not from the runtime that drives it.
const std = @import("std");
const abi = @import("../abi.zig");

pub const Facts = struct {
    simds: u64 = 0,
    simds_per_cu: u64 = 0,
    target: u64 = 0,
    wave: u64 = 0,
    lds_kib: u64 = 0,
    clock_mhz: u64 = 0,
    local_mem: u64 = 0,
    location: u64 = 0,
    domain: u64 = 0,
    l2_kib: u64 = 0,

    /// HIP's numbering: RDNA (gfx10 and later) counts a work-group processor, two compute units, as one.
    fn units(f: Facts) u64 {
        const cus = f.simds / @max(f.simds_per_cu, 1);
        return if (f.target >= 100000) cus / 2 else cus;
    }

    pub fn attribute(f: Facts, a: abi.DeviceAttribute) ?c_int {
        const per_unit: u64 = if (f.target >= 100000) 2 else 1;
        const v: u64 = switch (a) {
            .clock_rate => f.clock_mhz * 1000,
            .cooperative_launch => 1,
            .integrated => @intFromBool(f.local_mem == 0),
            .l2_cache_size => f.l2_kib * 1024,
            .max_threads_per_block => 1024,
            .multiprocessor_count => f.units(),
            .pci_bus_id => f.location >> 8,
            .pci_device_id => (f.location >> 3) & 31,
            .pci_domain_id => f.domain,
            .max_shared_memory_per_block, .shared_mem_per_block_optin => f.lds_kib * 1024,
            .warp_size => f.wave,
            .max_shared_memory_per_multiprocessor => f.lds_kib * 1024 * per_unit,
        };
        return std.math.cast(c_int, v);
    }

    /// The amdgpu PCI device directory, as /sys/bus/pci/devices names it.
    pub fn pciPath(f: Facts, buf: []u8, leaf: []const u8) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.{d}/{s}", .{ f.domain, f.location >> 8, (f.location >> 3) & 31, f.location & 7, leaf }, 0);
    }
};

/// The first KFD topology node that is a GPU (a node with a graphics target version).
pub fn gpu() !Facts {
    var path: [128]u8 = undefined;
    var node: usize = 0;
    while (node < 64) : (node += 1) {
        const dir = try std.fmt.bufPrint(&path, "/sys/class/kfd/kfd/topology/nodes/{d}", .{node});
        var f: Facts = .{};
        if (!try properties(dir, &f)) return error.NoDevice;
        if (f.target == 0) continue;
        f.l2_kib = l2(dir);
        return f;
    }
    return error.NoDevice;
}

fn properties(dir: []const u8, f: *Facts) !bool {
    var buf: [8192]u8 = undefined;
    var path: [160]u8 = undefined;
    const text = read(try std.fmt.bufPrintSentinel(&path, "{s}/properties", .{dir}, 0), &buf) orelse return false;
    const keys = .{
        .{ "simd_count", "simds" },          .{ "simd_per_cu", "simds_per_cu" },
        .{ "gfx_target_version", "target" }, .{ "wave_front_size", "wave" },
        .{ "lds_size_in_kb", "lds_kib" },    .{ "max_engine_clk_fcompute", "clock_mhz" },
        .{ "local_mem_size", "local_mem" },  .{ "location_id", "location" },
        .{ "domain", "domain" },
    };
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const sp = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
        const value = std.fmt.parseInt(u64, line[sp + 1 ..], 10) catch continue;
        inline for (keys) |k| if (std.mem.eql(u8, line[0..sp], k[0])) {
            @field(f, k[1]) = value;
        };
    }
    return true;
}

/// The largest level-2 cache a node lists, in KiB.
fn l2(dir: []const u8) u64 {
    var best: u64 = 0;
    var path: [192]u8 = undefined;
    var buf: [1024]u8 = undefined;
    for (0..256) |i| {
        const p = std.fmt.bufPrintSentinel(&path, "{s}/caches/{d}/properties", .{ dir, i }, 0) catch return best;
        const text = read(p, &buf) orelse return best;
        if (std.mem.indexOf(u8, text, "level 2\n") == null) continue;
        if (field(text, "size")) |kib| best = @max(best, kib);
    }
    return best;
}

/// One `name value` line's value from a sysfs properties file.
pub fn field(text: []const u8, name: []const u8) ?u64 {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len > name.len and std.mem.startsWith(u8, line, name) and line[name.len] == ' ')
            return std.fmt.parseInt(u64, line[name.len + 1 ..], 10) catch null;
    }
    return null;
}

/// A whole small sysfs file, or null when it cannot be read.
pub fn read(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n < 0) return null;
    return buf[0..@intCast(n)];
}

/// A sysfs file holding one decimal number.
pub fn number(path: [:0]const u8) ?u64 {
    var buf: [64]u8 = undefined;
    const text = read(path, &buf) orelse return null;
    return std.fmt.parseInt(u64, std.mem.trim(u8, text, " \n"), 10) catch null;
}

test "RDNA counts work-group processors and the PCI node path comes from the location id" {
    const f: Facts = .{ .simds = 80, .simds_per_cu = 2, .target = 110501, .wave = 32, .lds_kib = 64, .clock_mhz = 2900, .location = 0xbd00, .l2_kib = 2048 };
    try std.testing.expectEqual(@as(?c_int, 20), f.attribute(.multiprocessor_count));
    try std.testing.expectEqual(@as(?c_int, 1), f.attribute(.integrated));
    try std.testing.expectEqual(@as(?c_int, 131072), f.attribute(.max_shared_memory_per_multiprocessor));
    try std.testing.expectEqual(@as(?c_int, 2900000), f.attribute(.clock_rate));
    try std.testing.expectEqual(@as(?c_int, 0xbd), f.attribute(.pci_bus_id));
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("/sys/bus/pci/devices/0000:bd:00.0/mem_info_gtt_total", try f.pciPath(&buf, "mem_info_gtt_total"));
    try std.testing.expectEqual(@as(?u64, 2048), field("level 2\nsize 2048\ncache_line_size 128\n", "size"));
}
