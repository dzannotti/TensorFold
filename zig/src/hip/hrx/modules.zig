//! hipcc code objects as HRX executables, with the kernel facts HIP's function queries need from their metadata.
const std = @import("std");
const abi = @import("../abi.zig");
const c = @import("c.zig");
const drv = @import("driver.zig");
const meta = @import("meta.zig");
const memory = @import("memory.zig");

pub const Module = struct {
    exe: c.Executable,
    image: []u8,
    kernels: []meta.Kernel,
    functions: std.ArrayList(*Function) = .empty,
    globals: std.ArrayList(Global) = .empty,
    lock: std.atomic.Mutex = .unlocked,
};

pub const Function = struct { module: *Module, ordinal: u32, k: meta.Kernel };
const Global = struct { buffer: c.Buffer, address: u64, len: usize, name: []const u8 };

pub fn function(h: abi.Function) *Function {
    return @ptrCast(@alignCast(h.?));
}

pub fn hipModuleLoadData(out: *abi.Module, image: [*]const u8) callconv(.c) abi.Result {
    const dv = drv.d();
    const copy = drv.gpa.dupe(u8, meta.elf(image) catch return 200) catch return drv.out_of_memory;
    const kernels = meta.kernels(drv.gpa, copy) catch {
        drv.gpa.free(copy);
        return 200;
    };
    var exe: ?c.Executable = null;
    const r = drv.hip(dv.f.hrx_executable_load_data(dv.device, copy.ptr, copy.len, "amdgpu", @ptrCast(&dv.arch), &exe));
    if (r != drv.success) {
        meta.free(drv.gpa, kernels);
        drv.gpa.free(copy);
        return r;
    }
    const m = drv.gpa.create(Module) catch return drv.out_of_memory;
    m.* = .{ .exe = exe.?, .image = copy, .kernels = kernels };
    out.* = @ptrCast(m);
    return drv.success;
}

pub fn hipModuleUnload(h: abi.Module) callconv(.c) abi.Result {
    const m: *Module = @ptrCast(@alignCast(h orelse return drv.invalid_handle));
    const dv = drv.d();
    for (m.functions.items) |f| drv.gpa.destroy(f);
    m.functions.deinit(drv.gpa);
    for (m.globals.items) |g| {
        memory.disown(g.address);
        dv.f.hrx_buffer_release(g.buffer);
    }
    m.globals.deinit(drv.gpa);
    dv.f.hrx_executable_release(m.exe);
    meta.free(drv.gpa, m.kernels);
    drv.gpa.free(m.image);
    drv.gpa.destroy(m);
    return drv.success;
}

pub fn hipModuleGetFunction(out: *abi.Function, h: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
    const m: *Module = @ptrCast(@alignCast(h orelse return drv.invalid_handle));
    const n = std.mem.span(name);
    drv.lock(&m.lock);
    defer m.lock.unlock();
    for (m.functions.items) |f| if (std.mem.eql(u8, f.k.name, n)) {
        out.* = @ptrCast(f);
        return drv.success;
    };
    const k = meta.find(m.kernels, n) orelse return drv.not_found;
    var ordinal: u32 = 0;
    const r = drv.hip(drv.d().f.hrx_executable_lookup_export_by_name(m.exe, name, &ordinal));
    if (r != drv.success) return r;
    const f = drv.gpa.create(Function) catch return drv.out_of_memory;
    f.* = .{ .module = m, .ordinal = ordinal, .k = k };
    m.functions.append(drv.gpa, f) catch {
        drv.gpa.destroy(f);
        return drv.out_of_memory;
    };
    out.* = @ptrCast(f);
    return drv.success;
}

pub fn hipModuleGetGlobal(out: *abi.DevicePtr, bytes: *usize, h: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
    const m: *Module = @ptrCast(@alignCast(h orelse return drv.invalid_handle));
    const n = std.mem.span(name);
    drv.lock(&m.lock);
    defer m.lock.unlock();
    for (m.globals.items) |g| if (std.mem.eql(u8, g.name, n)) {
        out.* = @ptrFromInt(g.address);
        bytes.* = g.len;
        return drv.success;
    };
    const dv = drv.d();
    var b: ?c.Buffer = null;
    const lookup = dv.x.hrx_executable_lookup_global orelse return drv.not_supported;
    const r = drv.hip(lookup(m.exe, name, &b));
    if (r != drv.success) return r;
    var len: usize = 0;
    _ = drv.hip(dv.f.hrx_buffer_get_size(b.?, &len));
    const address = memory.adopt(b.?, len) catch {
        dv.f.hrx_buffer_release(b.?);
        return drv.unknown;
    };
    m.globals.append(drv.gpa, .{ .buffer = b.?, .address = address, .len = len, .name = n }) catch return drv.out_of_memory;
    out.* = @ptrFromInt(address);
    bytes.* = len;
    return drv.success;
}

/// Blocks one work-group processor holds: four SIMD32s (1536 VGPRs, 16 waves each) sharing two CUs' LDS.
fn occupancy(k: meta.Kernel, threads: u32, shared: usize) u32 {
    const f = drv.d().facts;
    const wave: u32 = if (k.wave != 0) k.wave else @intCast(f.wave);
    const waves = (threads + wave - 1) / wave * (wave / 32);
    const per_simd: u32 = @min(16, 1536 / std.mem.alignForward(u32, @max(k.vgprs, 1), 8));
    const by_regs = 4 * per_simd / waves;
    const lds = @as(u64, k.lds) + shared;
    const by_lds: u64 = if (lds == 0) by_regs else f.lds_kib * 1024 * 2 / lds;
    return @intCast(@min(by_regs, by_lds));
}

pub fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(out: *c_int, h: abi.Function, threads: c_int, shared: usize) callconv(.c) abi.Result {
    if (h == null or threads <= 0) return drv.invalid_value;
    out.* = @intCast(occupancy(function(h).k, @intCast(threads), shared));
    return drv.success;
}

pub fn hipFuncGetAttribute(out: *c_int, a: abi.FunctionAttribute, h: abi.Function) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const k = function(h).k;
    const lds: u32 = @intCast(drv.d().facts.lds_kib * 1024);
    out.* = switch (a) {
        .max_threads_per_block => @intCast(k.bound),
        .shared_size_bytes => @intCast(k.lds),
        .const_size_bytes => 0,
        .local_size_bytes => @intCast(k.private),
        .num_regs => @intCast(k.vgprs),
        .max_dynamic_shared_size_bytes => @intCast(lds - k.lds),
    };
    return drv.success;
}
