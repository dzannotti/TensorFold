//! HIP graphs as node lists (kernels, device copies, fills); replay records them on one stream and submits once.
const std = @import("std");
const c = @import("c.zig");
const drv = @import("driver.zig");
const Function = @import("modules.zig").Function;

pub const Kernel = struct {
    f: *Function,
    cfg: c.DispatchConfig,
    shared: u32,
    constants: []u8,
};
pub const Copy = struct { src: c.Buffer, src_offset: usize, dst: c.Buffer, dst_offset: usize, n: usize };
pub const Fill = struct { dst: c.Buffer, offset: usize, n: usize, pattern: u32, width: usize };
pub const Op = union(enum) { kernel: Kernel, copy: Copy, fill: Fill };

pub const Node = struct {
    op: Op,
    index: usize = 0,
    deps: std.ArrayList(*Node) = .empty,

    pub fn setKernel(n: *Node, k: Kernel) !void {
        const copy = try drv.gpa.dupe(u8, k.constants);
        if (n.op == .kernel) drv.gpa.free(n.op.kernel.constants);
        n.op = .{ .kernel = .{ .f = k.f, .cfg = k.cfg, .shared = k.shared, .constants = copy } };
    }

    fn destroy(n: *Node) void {
        if (n.op == .kernel) drv.gpa.free(n.op.kernel.constants);
        n.deps.deinit(drv.gpa);
        drv.gpa.destroy(n);
    }
};

pub const Graph = struct {
    nodes: std.ArrayList(*Node) = .empty,

    pub fn create() !*Graph {
        const g = try drv.gpa.create(Graph);
        g.* = .{};
        return g;
    }

    pub fn destroy(g: *Graph) void {
        for (g.nodes.items) |n| n.destroy();
        g.nodes.deinit(drv.gpa);
        drv.gpa.destroy(g);
    }

    /// A node after `deps`; kernel argument bytes are copied, so the caller's may change afterwards.
    pub fn add(g: *Graph, op: Op, deps: []const *Node) !*Node {
        const n = try drv.gpa.create(Node);
        n.* = .{ .op = .{ .fill = undefined } };
        errdefer drv.gpa.destroy(n);
        switch (op) {
            .kernel => |k| try n.setKernel(k),
            else => n.op = op,
        }
        errdefer if (n.op == .kernel) drv.gpa.free(n.op.kernel.constants);
        try n.deps.appendSlice(drv.gpa, deps);
        n.index = g.nodes.items.len;
        try g.nodes.append(drv.gpa, n);
        return n;
    }

    pub fn has(g: *const Graph, n: *const Node) bool {
        return std.mem.indexOfScalar(*Node, g.nodes.items, @constCast(n)) != null;
    }

    /// Nodes in an order every edge respects, ties in insertion order; null on a cycle. `placed`: a slot per node.
    pub fn order(g: *const Graph, out: []*Node, placed: []bool) ?[]*Node {
        @memset(placed, false);
        var done: usize = 0;
        while (done < g.nodes.items.len) {
            const before = done;
            for (g.nodes.items) |n| {
                if (placed[n.index]) continue;
                const ready = for (n.deps.items) |dep| {
                    if (!placed[dep.index]) break false;
                } else true;
                if (!ready) continue;
                placed[n.index] = true;
                out[done] = n;
                done += 1;
            }
            if (done == before) return null;
        }
        return out[0..done];
    }
};

/// A stream's capture: each recorded operation depends on the one before it.
pub const Capture = struct {
    graph: *Graph,
    last: ?*Node = null,
    failed: bool = false,

    fn append(cap: *Capture, op: Op) c_int {
        const deps: []const *Node = if (cap.last) |l| &.{l} else &.{};
        cap.last = cap.graph.add(op, deps) catch {
            cap.failed = true;
            return drv.out_of_memory;
        };
        return drv.success;
    }

    pub fn kernel(cap: *Capture, f: *Function, cfg: c.DispatchConfig, shared: u32, constants: []const u8) c_int {
        return cap.append(.{ .kernel = .{ .f = f, .cfg = cfg, .shared = shared, .constants = @constCast(constants) } });
    }

    pub fn copy(cap: *Capture, src: c.Buffer, src_offset: usize, dst: c.Buffer, dst_offset: usize, n: usize) c_int {
        return cap.append(.{ .copy = .{ .src = src, .src_offset = src_offset, .dst = dst, .dst_offset = dst_offset, .n = n } });
    }

    pub fn fill(cap: *Capture, dst: c.Buffer, offset: usize, n: usize, pattern: u32, width: usize) c_int {
        return cap.append(.{ .fill = .{ .dst = dst, .offset = offset, .n = n, .pattern = pattern, .width = width } });
    }
};

/// Records one node on `s` without submitting it.
pub fn record(s: c.Stream, op: Op) c.Status {
    const f = &drv.d().f;
    return switch (op) {
        .kernel => |k| f.hrx_stream_dispatch_with_local_memory(s, k.f.module.exe, k.f.ordinal, &k.cfg, k.shared, k.constants.ptr, k.constants.len, null, 0, c.dispatch_direct_arguments),
        .copy => |x| f.hrx_stream_copy_buffer_linear(s, x.src, x.src_offset, x.dst, x.dst_offset, x.n),
        .fill => |x| f.hrx_stream_fill_buffer_linear(s, x.dst, x.offset, x.n, &x.pattern, x.width),
    };
}

test "replay order follows edges added after the nodes and refuses a cycle" {
    var g: Graph = .{};
    defer {
        for (g.nodes.items) |n| {
            n.deps.deinit(std.testing.allocator);
            std.testing.allocator.destroy(n);
        }
        g.nodes.deinit(std.testing.allocator);
    }
    var nodes: [3]*Node = undefined;
    for (&nodes, 0..) |*n, i| {
        n.* = try std.testing.allocator.create(Node);
        n.*.* = .{ .op = .{ .fill = undefined }, .index = i };
        try g.nodes.append(std.testing.allocator, n.*);
    }
    try nodes[0].deps.append(std.testing.allocator, nodes[2]);
    var out: [3]*Node = undefined;
    var placed: [3]bool = undefined;
    const got = g.order(&out, &placed).?;
    try std.testing.expectEqualSlices(*Node, &.{ nodes[1], nodes[2], nodes[0] }, got);
    try nodes[2].deps.append(std.testing.allocator, nodes[0]);
    try std.testing.expect(g.order(&out, &placed) == null);
}
