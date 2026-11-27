//! libhrx's C ABI as the native driver calls it (hrx_runtime.h, hrx_device.h, hrx_memory.h), resolved at run time.
const std = @import("std");

pub const Status = ?*opaque {};
pub const Runtime = *opaque {};
pub const Group = *opaque {};
pub const Device = *opaque {};
pub const Queue = *opaque {};
pub const Allocator = *opaque {};
pub const Semaphore = *opaque {};
pub const Stream = *opaque {};
pub const Buffer = *opaque {};
pub const Executable = *opaque {};
pub const Event = *opaque {};
pub const Graph = *opaque {};
pub const GraphExec = *opaque {};
pub const GraphNode = *opaque {};

pub const HostAllocator = extern struct { self: ?*anyopaque, ctl: ?*anyopaque };
pub const DeviceInfo = extern struct { uri: [*:0]const u8, name: [*:0]const u8 };
pub const DeviceParams = extern struct { uri: [*:0]const u8, completion_wait_flags: u32 };
pub const Scope = extern struct { domain: ?*const anyopaque = null, id: u32 = 0 };
pub const BufferParams = extern struct { type: u32, access: u16, usage: u32 };
pub const BufferRef = extern struct { buffer: Buffer, offset: usize, length: usize };
pub const TimelinePoint = extern struct { semaphore: ?Semaphore, value: u64 };
pub const SemaphoreList = extern struct { semaphores: ?[*]const Semaphore, values: ?[*]const u64, count: usize };
pub const DispatchConfig = extern struct { workgroup_count: [3]u32, workgroup_size: [3]u32, subgroup_size: u32 = 0 };
pub const ExportInfo = extern struct {
    name: ?[*:0]const u8,
    flags: u32,
    constant_byte_length: u32,
    binding_count: u32,
    parameter_count: u32,
    workgroup_size: [3]u32,
};
pub const QueueFamilyInfo = extern struct {
    provisioned_queue_count: u32,
    supports_acquisition: bool,
    roles: u32,
    placements: u32,
    features: u64,
    priority_count: usize,
    priorities: ?[*]const i32,
    execution_unit_count: u32,
    execution_resource_count: usize,
    execution_resource_group_count: usize,
};
pub const QueueParams = extern struct { priority: i32 = 0, placement: c_int = 0, features: u64 = 0, resources: extern struct { count: usize = 0, ordinals: ?[*]const u32 = null } = .{} };
pub const MemoryBarrier = extern struct { source_scope: u32, target_scope: u32 };
pub const MemoryEffects = extern struct { bits: u32 = 0 };
pub const Barrier = extern struct {
    source_stage_mask: u32,
    target_stage_mask: u32,
    flags: u64 = 0,
    effects: MemoryEffects = .{},
    memory_barrier_count: usize = 0,
    memory_barriers: ?[*]const MemoryBarrier = null,
    buffer_barrier_count: usize = 0,
    buffer_barriers: ?*const anyopaque = null,
};
pub const BarrierList = extern struct { count: usize, values: ?[*]const Barrier };
pub const QueueBarriers = extern struct { before: ?*const BarrierList, after: ?*const BarrierList };
pub const KernelNode = extern struct {
    executable: Executable,
    export_ordinal: u32,
    config: DispatchConfig,
    constants: ?*const anyopaque,
    constants_size: usize,
    bindings: ?[*]const BufferRef = null,
    binding_count: usize = 0,
    flags: u32,
};
pub const Strides = extern struct { row_stride: u64 = 0, slice_stride: u64 = 0 };
pub const TransferRegion = extern struct {
    row_length: u64,
    row_count: u64 = 1,
    slice_count: u64 = 1,
    source_strides: Strides = .{},
    target_strides: Strides = .{},
};
pub const FillRegion = extern struct { row_length: u64, row_count: u64 = 1, slice_count: u64 = 1, target_strides: Strides = .{} };
pub const CopyNode = extern struct { src: BufferRef, dst: BufferRef, region: TransferRegion };
pub const FillNode = extern struct { dst: BufferRef, pattern: u32, pattern_size: usize, region: FillRegion };

pub const memory_device_local: u32 = 0x30;
pub const memory_host_local: u32 = 0x42;
pub const memory_host_coherent: u32 = 0x04;
pub const memory_device_visible: u32 = 0x10;
pub const access_all: u16 = 0x03;
pub const usage_default: u32 = 0x0C03;
pub const usage_mapping_persistent: u32 = 0x02000000;
pub const map_read_write: u16 = 0x03;
pub const interface_host: c_int = 0;
pub const interface_device_address: c_int = 1;
pub const site_queue: c_int = 1;
pub const site_program: c_int = 2;
pub const wait_yield: u32 = 1 << 0;
pub const wait_active: u32 = 1 << 1;
pub const role_dispatch: u32 = 1 << 0;
pub const role_transfer: u32 = 1 << 1;
pub const dispatch_direct_arguments: u32 = 1 << 0;
pub const dispatch_cooperative: u32 = 1 << 2;
/// HRX_QUEUE_FEATURE_COOPERATIVE_DISPATCH: such a queue runs cooperative dispatches as well as plain ones.
pub const queue_cooperative: u64 = 1 << 0;
pub const event_disable_timing: c_int = 1 << 1;
pub const property_name: c_int = 0;
pub const property_architecture: c_int = 1;
pub const stage_dispatch: u32 = 1 << 2;
pub const stage_transfer: u32 = 1 << 3;
pub const scope_dispatch_read: u32 = 1 << 2;
pub const scope_dispatch_write: u32 = 1 << 3;
pub const barrier_release_system: u64 = 1 << 1;
pub const status_unavailable: c_int = 14;
pub const status_deadline: c_int = 4;
pub const status_out_of_memory: c_int = 8;
pub const status_not_found: c_int = 5;
pub const status_unimplemented: c_int = 12;
pub const forever: u64 = std.math.maxInt(u64);

/// Every libhrx entry point the driver calls; resolved by name, a missing one refuses the driver.
pub const Lib = struct {
    hrx_status_code: *const fn (Status) callconv(.c) c_int,
    hrx_status_ignore: *const fn (Status) callconv(.c) void,
    hrx_status_to_string: *const fn (Status, *?[*]u8, *usize) callconv(.c) Status,
    hrx_status_free_message: *const fn (?[*]u8) callconv(.c) void,
    hrx_host_allocator_system: *const fn () callconv(.c) HostAllocator,
    hrx_host_allocator_free: *const fn (HostAllocator, ?*anyopaque) callconv(.c) void,
    hrx_runtime_create: *const fn (?*const anyopaque, HostAllocator, *?Runtime) callconv(.c) Status,
    hrx_runtime_query_available_devices: *const fn (Runtime, [*:0]const u8, HostAllocator, *usize, *?[*]DeviceInfo) callconv(.c) Status,
    hrx_device_group_create: *const fn (Runtime, usize, [*]const DeviceParams, *?Group) callconv(.c) Status,
    hrx_device_group_get_device: *const fn (Group, usize) callconv(.c) ?Device,
    hrx_device_get_property: *const fn (Device, c_int, *anyopaque, usize) callconv(.c) Status,
    hrx_device_query_queue_family: *const fn (Device, u32, *QueueFamilyInfo) callconv(.c) Status,
    hrx_device_get_queue: *const fn (Device, u32, u32) callconv(.c) ?Queue,
    hrx_device_query_queue_family_memory_scope: *const fn (Device, u32, c_int, *Scope) callconv(.c) Status,
    hrx_device_allocator: *const fn (Device) callconv(.c) Allocator,
    hrx_allocator_allocate_buffer: *const fn (Allocator, BufferParams, usize, *?Buffer) callconv(.c) Status,
    hrx_buffer_release: *const fn (Buffer) callconv(.c) void,
    hrx_buffer_get_size: *const fn (Buffer, *usize) callconv(.c) Status,
    hrx_buffer_query_address: *const fn (Buffer, Scope, c_int, *u64) callconv(.c) Status,
    hrx_buffer_map: *const fn (Buffer, u16, usize, usize, *?*anyopaque) callconv(.c) Status,
    hrx_buffer_unmap: *const fn (Buffer) callconv(.c) Status,
    hrx_semaphore_wait: *const fn (Semaphore, u64, u64, u32) callconv(.c) Status,
    hrx_stream_create: *const fn (Queue, u32, *?Stream) callconv(.c) Status,
    hrx_stream_release: *const fn (Stream) callconv(.c) void,
    hrx_stream_synchronize: *const fn (Stream, u32) callconv(.c) Status,
    hrx_stream_query: *const fn (Stream, *bool) callconv(.c) Status,
    hrx_stream_flush: *const fn (Stream) callconv(.c) Status,
    hrx_stream_get_timeline_position: *const fn (Stream, *TimelinePoint) callconv(.c) Status,
    hrx_stream_wait_on: *const fn (Stream, TimelinePoint) callconv(.c) Status,
    hrx_stream_copy_h2d_linear: *const fn (Stream, *const anyopaque, Buffer, usize, usize) callconv(.c) Status,
    hrx_stream_copy_d2h_linear: *const fn (Stream, Buffer, usize, *anyopaque, usize) callconv(.c) Status,
    hrx_stream_wait_event: *const fn (Stream, Event) callconv(.c) Status,
    hrx_stream_advance_timeline: *const fn (Stream, *u64) callconv(.c) Status,
    hrx_stream_dispatch: *const fn (Stream, Executable, u32, *const DispatchConfig, ?*const anyopaque, usize, ?[*]const BufferRef, usize, u32) callconv(.c) Status,
    hrx_semaphore_signal: *const fn (Semaphore, u64) callconv(.c) Status,
    hrx_queue_dispatch: *const fn (Queue, ?*const SemaphoreList, ?*const SemaphoreList, Executable, u32, *const DispatchConfig, ?*const anyopaque, usize, ?[*]const BufferRef, usize, ?*const QueueBarriers, u32) callconv(.c) Status,
    hrx_queue_copy_linear: *const fn (Queue, ?*const SemaphoreList, ?*const SemaphoreList, Buffer, usize, Buffer, usize, usize, ?*const QueueBarriers) callconv(.c) Status,
    hrx_executable_load_data: *const fn (Device, *const anyopaque, usize, [*:0]const u8, [*:0]const u8, *?Executable) callconv(.c) Status,
    hrx_executable_release: *const fn (Executable) callconv(.c) void,
    hrx_executable_lookup_export_by_name: *const fn (Executable, [*:0]const u8, *u32) callconv(.c) Status,
    hrx_stream_copy_buffer_linear: *const fn (Stream, Buffer, usize, Buffer, usize, usize) callconv(.c) Status,
    hrx_stream_fill_buffer_linear: *const fn (Stream, Buffer, usize, usize, *const anyopaque, usize) callconv(.c) Status,
    hrx_queue_acquire: *const fn (Device, u32, *const QueueParams, *?Queue) callconv(.c) Status,
    hrx_queue_release: *const fn (Queue) callconv(.c) void,
    hrx_event_create: *const fn (Device, c_int, *?Event) callconv(.c) Status,
    hrx_event_release: *const fn (Event) callconv(.c) void,
    hrx_event_record: *const fn (Event, Stream) callconv(.c) Status,
    hrx_event_query: *const fn (Event, *bool) callconv(.c) Status,
    hrx_event_synchronize: *const fn (Event, u32) callconv(.c) Status,
    hrx_event_elapsed_time: *const fn (Event, Event, *f32) callconv(.c) Status,
};

/// Entry points only patched libhrx has (hrx-system patch 0004); an absent one refuses only the HIP calls needing it.
pub const Extra = struct {
    /// HIP's dynamic shared memory (sharedMemBytes > 0) and cooperative launches.
    hrx_stream_dispatch_with_local_memory: ?*const fn (Stream, Executable, u32, *const DispatchConfig, u32, ?*const anyopaque, usize, ?[*]const BufferRef, usize, u32) callconv(.c) Status = null,
    /// hipModuleGetGlobal.
    hrx_executable_lookup_global: ?*const fn (Executable, [*:0]const u8, *?Buffer) callconv(.c) Status = null,
};

test "HRX value records keep their C layouts" {
    try std.testing.expectEqual(16, @sizeOf(HostAllocator));
    try std.testing.expectEqual(16, @sizeOf(Scope));
    try std.testing.expectEqual(12, @sizeOf(BufferParams));
    try std.testing.expectEqual(28, @sizeOf(DispatchConfig));
    try std.testing.expectEqual(40, @sizeOf(ExportInfo));
    try std.testing.expectEqual(24, @sizeOf(SemaphoreList));
    try std.testing.expectEqual(56, @sizeOf(TransferRegion));
    try std.testing.expectEqual(56, @offsetOf(KernelNode, "bindings"));
    try std.testing.expectEqual(56, @sizeOf(Barrier));
}
