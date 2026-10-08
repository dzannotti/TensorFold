//! Type-only stand-in for the cuda module (int4 host code under cuda.hip): checks cuda_int4.zig's HIP paths compile.
pub const hip = true;
pub const Error = error{Cuda};
pub const Stream = struct {
    pub fn synchronize(_: Stream) Error!void {}
};
pub const Args = struct {
    n: usize = 0,
    pub fn add(a: *Args, v: anytype) void {
        _ = v;
        a.n += 1;
    }
};
pub const Function = struct {
    pub fn occupancy(_: Function, _: u32, _: usize) Error!u32 {
        return 1;
    }
    pub fn allowDynamicShared(_: Function, _: u32) Error!void {}
};
pub const Module = struct {
    pub fn load(_: *const Driver, _: []const u8) Error!Module {
        return .{};
    }
    pub fn function(_: Module, _: [:0]const u8) Error!Function {
        return .{};
    }
    pub fn unload(_: Module) void {}
};
pub const Driver = struct {};
pub const Context = struct {
    d: *const Driver,
    pub fn attribute(_: *const Context, _: anytype) Error!c_int {
        return 20;
    }
};
pub const Dim3 = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };
pub const launch = struct {
    pub const Config = struct { grid: Dim3 = .{}, block: Dim3 = .{}, shared: u32 = 0 };
    pub fn launch(_: Function, _: Config, _: Stream, _: *Args) Error!void {}
};
pub const kernels = struct {
    pub const available = true;
    pub const fn_int4: []const u8 = &.{};
};
