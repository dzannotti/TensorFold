//! HIP's graph entry points over the driver's node lists: capture, explicit nodes, instantiate, replay and update.
const std = @import("std");
const abi = @import("../abi.zig");
const drv = @import("driver.zig");
const cap = @import("capture.zig");
const modules = @import("modules.zig");
const streams = @import("streams.zig");
const launch = @import("launch.zig");

const Exec = struct {
    /// The instantiated graph's nodes, compared by address only: HIP lets that graph be destroyed meanwhile.
    from: []*cap.Node,
    ops: []cap.Op,
    edges: []usize,

    fn destroy(e: *Exec) void {
        for (e.ops) |op| if (op == .kernel) drv.gpa.free(op.kernel.constants);
        drv.gpa.free(e.ops);
        drv.gpa.free(e.from);
        drv.gpa.free(e.edges);
        drv.gpa.destroy(e);
    }
};

fn graph(h: abi.Graph) *cap.Graph {
    return @ptrCast(@alignCast(h.?));
}

fn exec(h: abi.GraphExec) *Exec {
    return @ptrCast(@alignCast(h.?));
}

fn node(h: abi.GraphNode) *cap.Node {
    return @ptrCast(@alignCast(h.?));
}

fn kernelOp(p: *const abi.KernelNodeParams, buf: *[launch.max_bytes]u8) ?cap.Kernel {
    if (p.func == null or p.extra != null) return null;
    if (p.shared_bytes != 0 and !launch.local()) return null;
    const f = modules.function(p.func);
    const constants = launch.pack(f.k, p.params, buf) orelse return null;
    return .{
        .f = f,
        .cfg = .{ .workgroup_count = .{ p.grid.x, p.grid.y, p.grid.z }, .workgroup_size = .{ p.block.x, p.block.y, p.block.z } },
        .shared = p.shared_bytes,
        .constants = @constCast(constants),
    };
}

pub fn hipStreamBeginCapture(h: abi.Stream, mode: abi.CaptureMode) callconv(.c) abi.Result {
    _ = mode;
    const s = streams.of(h);
    if (h == null or s.capture != null) return drv.invalid_value;
    const g = cap.Graph.create() catch return drv.out_of_memory;
    const k = drv.gpa.create(cap.Capture) catch {
        g.destroy();
        return drv.out_of_memory;
    };
    k.* = .{ .graph = g };
    s.capture = k;
    return drv.success;
}

pub fn hipStreamEndCapture(h: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
    const s = streams.of(h);
    const k = s.capture orelse return drv.invalid_value;
    s.capture = null;
    defer drv.gpa.destroy(k);
    out.* = null;
    if (k.failed) {
        k.graph.destroy();
        return 901;
    }
    out.* = @ptrCast(k.graph);
    return drv.success;
}

pub fn hipStreamIsCapturing(h: abi.Stream, out: *abi.CaptureStatus) callconv(.c) abi.Result {
    const s = streams.of(h);
    out.* = if (s.capture) |k| (if (k.failed) .invalidated else .active) else .none;
    return drv.success;
}

pub fn hipGraphCreate(out: *abi.Graph, flags: c_uint) callconv(.c) abi.Result {
    if (flags != 0) return drv.invalid_value;
    out.* = @ptrCast(cap.Graph.create() catch return drv.out_of_memory);
    return drv.success;
}

pub fn hipGraphDestroy(h: abi.Graph) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    graph(h).destroy();
    return drv.success;
}

pub fn hipGraphAddKernelNode(out: *abi.GraphNode, h: abi.Graph, deps: ?[*]const abi.GraphNode, n: usize, p: *const abi.KernelNodeParams) callconv(.c) abi.Result {
    if (h == null or (n > 0 and deps == null)) return drv.invalid_value;
    const g = graph(h);
    var buf: [launch.max_bytes]u8 = undefined;
    const k = kernelOp(p, &buf) orelse return drv.invalid_value;
    var list: [64]*cap.Node = undefined;
    if (n > list.len) return drv.not_supported;
    for (0..n) |i| {
        const d = node(deps.?[i]);
        if (!g.has(d)) return drv.invalid_value;
        list[i] = d;
    }
    out.* = @ptrCast(g.add(.{ .kernel = k }, list[0..n]) catch return drv.out_of_memory);
    return drv.success;
}

pub fn hipGraphKernelNodeSetParams(h: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const n = node(h);
    if (n.op != .kernel) return drv.invalid_value;
    var buf: [launch.max_bytes]u8 = undefined;
    const k = kernelOp(p, &buf) orelse return drv.invalid_value;
    n.setKernel(k) catch return drv.out_of_memory;
    return drv.success;
}

pub fn hipGraphAddDependencies(h: abi.Graph, from: [*]const abi.GraphNode, to: [*]const abi.GraphNode, n: usize) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const g = graph(h);
    for (0..n) |i| {
        const a = node(from[i]);
        const b = node(to[i]);
        if (!g.has(a) or !g.has(b)) return drv.invalid_value;
        b.deps.append(drv.gpa, a) catch return drv.out_of_memory;
    }
    return drv.success;
}

pub fn hipGraphGetNodes(h: abi.Graph, out: ?[*]abi.GraphNode, count: *usize) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const g = graph(h);
    if (out) |o| {
        const n = @min(count.*, g.nodes.items.len);
        for (g.nodes.items[0..n], 0..) |x, i| o[i] = @ptrCast(x);
    }
    count.* = g.nodes.items.len;
    return drv.success;
}

pub fn hipGraphInstantiateWithFlags(out: *abi.GraphExec, h: abi.Graph, flags: c_ulonglong) callconv(.c) abi.Result {
    if (h == null or flags != 0) return drv.invalid_value;
    const g = graph(h);
    const n = g.nodes.items.len;
    const from = drv.gpa.alloc(*cap.Node, n) catch return drv.out_of_memory;
    const placed = drv.gpa.alloc(bool, n) catch {
        drv.gpa.free(from);
        return drv.out_of_memory;
    };
    defer drv.gpa.free(placed);
    const sorted = g.order(from, placed) orelse {
        drv.gpa.free(from);
        return drv.invalid_value;
    };
    const ops = drv.gpa.alloc(cap.Op, n) catch {
        drv.gpa.free(from);
        return drv.out_of_memory;
    };
    const edges = drv.gpa.alloc(usize, n) catch return drv.out_of_memory;
    for (sorted, ops, edges) |x, *op, *k| {
        op.* = x.op;
        k.* = x.deps.items.len;
        if (op.* == .kernel) op.kernel.constants = drv.gpa.dupe(u8, op.kernel.constants) catch return drv.out_of_memory;
    }
    const e = drv.gpa.create(Exec) catch return drv.out_of_memory;
    e.* = .{ .from = sorted, .ops = ops, .edges = edges };
    out.* = @ptrCast(e);
    return drv.success;
}

pub fn hipGraphUpload(h: abi.GraphExec, s: abi.Stream) callconv(.c) abi.Result {
    _ = s;
    return if (h == null) drv.invalid_handle else drv.success;
}

/// Every node recorded on the stream in order, then one submission.
pub fn hipGraphLaunch(h: abi.GraphExec, sh: abi.Stream) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    const s = streams.of(sh);
    if (s.capture != null) return drv.not_supported;
    for (exec(h).ops) |op| {
        const r = drv.hip(cap.record(s.h, op));
        if (r != drv.success) return r;
    }
    return s.flush(null);
}

pub fn hipGraphExecDestroy(h: abi.GraphExec) callconv(.c) abi.Result {
    if (h == null) return drv.invalid_handle;
    exec(h).destroy();
    return drv.success;
}

fn at(e: *Exec, n: *cap.Node) ?usize {
    return std.mem.indexOfScalar(*cap.Node, e.from, n);
}

fn replace(e: *Exec, i: usize, op: cap.Op) !void {
    var next = op;
    if (next == .kernel) next.kernel.constants = try drv.gpa.dupe(u8, next.kernel.constants);
    if (e.ops[i] == .kernel) drv.gpa.free(e.ops[i].kernel.constants);
    e.ops[i] = next;
}

pub fn hipGraphExecKernelNodeSetParams(h: abi.GraphExec, n: abi.GraphNode, p: *const abi.KernelNodeParams) callconv(.c) abi.Result {
    if (h == null or n == null) return drv.invalid_handle;
    const e = exec(h);
    const i = at(e, node(n)) orelse return drv.invalid_value;
    if (e.ops[i] != .kernel) return drv.invalid_value;
    var buf: [launch.max_bytes]u8 = undefined;
    const k = kernelOp(p, &buf) orelse return drv.invalid_value;
    if (k.f != e.ops[i].kernel.f) return drv.invalid_value;
    replace(e, i, .{ .kernel = k }) catch return drv.out_of_memory;
    return drv.success;
}

/// Same topology and kinds node for node (in replay order) or the exec is left as it was.
pub fn hipGraphExecUpdate(h: abi.GraphExec, gh: abi.Graph, bad: *abi.GraphNode, result: *abi.ExecUpdateResult) callconv(.c) abi.Result {
    if (h == null or gh == null) return drv.invalid_handle;
    const e = exec(h);
    const g = graph(gh);
    bad.* = null;
    const n = g.nodes.items.len;
    const order = drv.gpa.alloc(*cap.Node, n) catch return drv.out_of_memory;
    defer drv.gpa.free(order);
    const placed = drv.gpa.alloc(bool, n) catch return drv.out_of_memory;
    defer drv.gpa.free(placed);
    const sorted = g.order(order, placed) orelse {
        result.* = .topology_changed;
        return drv.invalid_value;
    };
    if (n != e.ops.len) {
        result.* = .topology_changed;
        return drv.invalid_value;
    }
    for (sorted, e.ops, e.edges) |x, op, edges| {
        if (std.meta.activeTag(x.op) != std.meta.activeTag(op) or x.deps.items.len != edges) {
            bad.* = @ptrCast(x);
            result.* = if (std.meta.activeTag(x.op) != std.meta.activeTag(op)) .node_type_changed else .topology_changed;
            return drv.invalid_value;
        }
        if (x.op == .kernel and x.op.kernel.f != op.kernel.f) {
            bad.* = @ptrCast(x);
            result.* = .function_changed;
            return drv.invalid_value;
        }
    }
    for (sorted, 0..) |x, i| replace(e, i, x.op) catch return drv.out_of_memory;
    result.* = .success;
    return drv.success;
}

test "kernel node parameters are refused without a function or with HIP's extra launch array" {
    var buf: [launch.max_bytes]u8 = undefined;
    var extra = [_]?*anyopaque{null};
    const p: abi.KernelNodeParams = .{ .block = .{}, .extra = &extra, .func = null, .grid = .{}, .params = null, .shared_bytes = 0 };
    try std.testing.expect(kernelOp(&p, &buf) == null);
}
