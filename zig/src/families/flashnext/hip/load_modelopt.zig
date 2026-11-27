//! FP4-Local onto one HIP rank: MXFP8 and bf16 as stored, row groups stitched, routed experts baked (load_experts).
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");
const loader = @import("load.zig");
const mh = @import("../modelopt_host.zig");
const names = @import("../names_modelopt.zig");
const nvfp4_read = @import("../nvfp4_read.zig");
const experts = @import("load_experts.zig");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const Host = quant.Host;
const Projection = model.Projection;
const Allocator = std.mem.Allocator;

/// Rows [from, from + n) of a stitched product: the same planes at those rows' addresses.
pub fn rowsOf(p: Projection, from: u32, n: u32) !Projection {
    var out = p;
    out.n = n;
    const at: u64 = from;
    out.handle = switch (p.handle) {
        .mxfp8 => |m| .{ .mxfp8 = .{ .codes = m.codes + at * p.k, .scales = m.scales + at * (p.k / quant.mxfp8_format.block) } },
        .dense => |d| .{ .dense = .{ .weight = d.weight + at * p.k * d.dtype.size(), .dtype = d.dtype, .fold = d.fold } },
        else => return error.UnexpectedTensor,
    };
    return out;
}

/// A host projection's rows and inputs.
fn dims(h: Host) [2]usize {
    return switch (h) {
        .mxfp8 => |m| .{ m.codes.shape[0], m.codes.shape[1] },
        .dense => |d| .{ d.weight.shape[0], d.weight.shape[1] },
        .nvfp4 => |x| .{ quant.fp4local_matrix.nvfp4.rows(x), x.words.shape[x.words.rank - 1] * 8 },
        .mlx, .bf16 => .{ 0, 0 },
    };
}

const Loader = struct {
    a: Allocator,
    ck: *Checkpoint,
    c: *const config.Config,
    up: hip.Upload,
    stage: *experts.Stage,
    around_one: bool,
    threads: usize,

    fn store(l: *const Loader) mh.Store(*Checkpoint) {
        return mh.store(l.ck);
    }

    /// `name`, built from tensors.* in the MLX spelling, as FP4-Local stores it.
    fn stored(l: *const Loader, name: []const u8) ![]const u8 {
        var buf: [512]u8 = undefined;
        return l.a.dupe(u8, try names.exported(name, &buf));
    }

    fn named(l: *const Loader, stem: []const u8, suffix: []const u8) ![]const u8 {
        return std.fmt.allocPrint(l.a, "{s}.{s}", .{ stem, suffix });
    }

    /// Uploaded; a bf16 matrix is marked FP4-Local's, so its products take the FP4-Local tiles.
    fn put(l: *const Loader, h: Host) !Projection {
        var p = try quant.view(try quant.upload(l.up.uploader(), h), false);
        if (p.handle == .dense and p.handle.dense.dtype == .bf16) p.handle.dense.fold = true;
        return p;
    }

    fn tensor(l: *const Loader, t: Tensor) !u64 {
        return (try l.up.uploader().tensor(t)).ptr;
    }

    fn float(l: *const Loader, k: []const u8) !Tensor {
        return quant.convert.float32(l.a, try l.ck.get(k));
    }

    /// A centred RMSNorm's fp32 scale from its stored value (tensors.centred).
    fn scale(l: *const Loader, k: []const u8) !u64 {
        var t = try l.float(k);
        const v = try loader.floats(l.a, t);
        for (v) |*x| x.* = tensors.centred(x.*, l.around_one);
        t.bytes = std.mem.sliceAsBytes(v);
        return l.tensor(t);
    }

    /// `got` against the (n, k) the config gives the projection `stem` names, where it gives one.
    fn shaped(l: *const Loader, stem: []const u8, got: [2]usize) !void {
        const want = (try tensors.shape(l.c, stem)) orelse return;
        if (got[0] == want.n and got[1] == want.k) return;
        std.log.err("{s}: [{d}, {d}], the config gives [{d}, {d}]", .{ stem, got[0], got[1], want.n, want.k });
        return error.UnexpectedShape;
    }

    fn proj(l: *const Loader, stem: []const u8) !Projection {
        const h = try mh.proj(l.a, l.store(), stem);
        try l.shaped(stem, dims(h));
        return l.put(h);
    }

    /// `parts` under `stem` stitched along N into one product, `views` each part's rows of it.
    fn group(l: *const Loader, stem: []const u8, parts: []const []const u8, views: []Projection) !Projection {
        const prefix = try std.fmt.allocPrint(l.a, "{s}.", .{stem});
        const whole = try l.put(try mh.stitched(l.a, l.store(), prefix, parts));
        var row: u32 = 0;
        for (parts, views) |p, *v| {
            const name = try std.fmt.allocPrint(l.a, "{s}{s}", .{ prefix, p });
            const w = try l.ck.get(try l.named(name, "weight"));
            try l.shaped(name, .{ w.dim(0), w.dim(1) });
            v.* = try rowsOf(whole, row, @intCast(w.dim(0)));
            row += @intCast(w.dim(0));
        }
        return whole;
    }

    /// The down projection with the inject rows under it (one product), and the up.
    fn hc(l: *const Loader, base: []const u8, inject: bool) !model.Hc {
        const t = tensors.hc;
        var views: [2]Projection = undefined;
        return .{
            .scale = try l.scale(try l.named(base, t.norm)),
            .down = if (inject) try l.group(base, &.{ t.down, t.inject }, &views) else try l.proj(try l.named(base, t.down)),
            .up = try l.proj(try l.named(base, t.up)),
        };
    }

    fn linear(l: *const Loader, base: []const u8) !model.Linear {
        const t = tensors.linear;
        var p: [4]Projection = undefined;
        const inputs = try l.group(base, &t.inputs, &p);
        var conv = try l.float(try l.named(base, t.conv));
        conv.rank = 2;
        return .{
            .qkv = p[0],
            .z = p[1],
            .b = p[2],
            .a = p[3],
            .out = try l.proj(try l.named(base, t.out)),
            .conv = try l.tensor(conv),
            .a_log = try l.tensor(try l.float(try l.named(base, t.a_log))),
            .dt_bias = try l.tensor(try l.float(try l.named(base, t.dt_bias))),
            .norm = try l.tensor(try l.float(try l.named(base, t.norm))),
            .inputs = inputs,
        };
    }

    fn sparse(l: *const Loader, base: []const u8) !model.Sparse {
        const t = tensors.sparse;
        var p: [4]Projection = undefined;
        const inputs = try l.group(base, &t.inputs, &p);
        return .{
            .q = p[0],
            .k = p[1],
            .v = p[2],
            .index = p[3],
            .o = try l.proj(try l.named(base, t.o)),
            .q_norm = try l.scale(try l.named(base, t.q_norm)),
            .k_norm = try l.scale(try l.named(base, t.k_norm)),
            .iq_norm = try l.scale(try l.named(base, t.iq_norm)),
            .ik_norm = try l.scale(try l.named(base, t.ik_norm)),
            .inputs = inputs,
        };
    }

    /// The router (E routed rows, the shared gate last), the shared expert beside the stack, the stacked experts.
    fn moe(l: *const Loader, base: []const u8) !model.Routed {
        const c = l.c;
        const t = tensors.moe;
        const r = try mh.router(l.a, l.store(), try l.named(base, t.gate), try l.named(base, t.shared_gate), c.experts, c.hidden);
        var views: [2]Projection = undefined;
        const gate_up = try l.group(base, t.shared[0..2], &views);
        const down = try l.proj(try l.named(base, t.shared[2]));
        if (gate_up.n != 2 * c.shared_width or gate_up.k != c.hidden or down.n != c.hidden or down.k != c.shared_width) {
            std.log.err("{s}.mlp.shared_expert: [{d}, {d}] and [{d}, {d}] are not gate | up and down of width {d}", .{ base, gate_up.n, gate_up.k, down.n, down.k, c.shared_width });
            return error.UnexpectedShape;
        }
        const x = try mh.Experts.read(l.a, l.store(), try l.named(base, "mlp"), c.experts, c.expert_width, c.hidden, l.threads);
        const s = try experts.upload(x, l.stage, &l.up);
        return .{
            .rows32 = try l.tensor(r.rows32),
            .rows16 = try l.tensor(r.rows16),
            .top_k = c.topk,
            .rows = c.experts + 1,
            .exact = true,
            .matrix = true,
            .experts = .{
                .fused = s.fused,
                .down = s.down,
                .count = c.experts,
                .width = c.expert_width,
                .dims = c.hidden,
                .shared = .{ .gate_up = gate_up, .down = down },
            },
        };
    }

    fn layer(l: *const Loader, base: []const u8, kind: config.Kind) !model.Layer {
        return .{
            .ahc = try l.hc(try l.named(base, tensors.hc.attn), true),
            .mhc = try l.hc(try l.named(base, tensors.hc.mlp), true),
            .mixer = switch (kind) {
                .linear_attention => .{ .linear = try l.linear(base) },
                .sparse_attention => .{ .sparse = try l.sparse(base) },
            },
            .moe = try l.moe(base),
        };
    }

    /// The n-gram layer's key and value projections stacked (bf16), its norms and conv; its table stays on disk.
    fn ple(l: *const Loader, base: []const u8, index: usize) !model.Ple {
        const t = tensors.ple;
        var views: [2]Projection = undefined;
        var conv = try l.float(try l.named(base, t.conv));
        conv.rank = 2;
        return .{
            .layer = index,
            .kv = try l.group(base, &.{ t.key, t.value }, &views),
            .key_scale = try l.scale(try l.named(base, t.norm_key)),
            .query_scale = try l.scale(try l.named(base, t.norm_query)),
            .conv_scale = try l.scale(try l.named(base, t.norm_conv)),
            .conv = try l.tensor(conv),
        };
    }

    /// A bf16 table [n, k] read by id or by row (embed, lm_head), its shape checked.
    fn table(l: *const Loader, stem: []const u8, n: usize, k: usize) !Host {
        const h = try mh.proj(l.a, l.store(), stem);
        const got = dims(h);
        if (h != .dense or got[0] != n or got[1] != k) {
            std.log.err("{s}: [{d}, {d}] {t}, expected bf16 [{d}, {d}]", .{ stem, got[0], got[1], std.meta.activeTag(h), n, k });
            return error.UnexpectedShape;
        }
        return h;
    }
};

/// FP4-Local's weights on one rank, the MTP head under `mtp_root`; layers read past the page cache, the next ahead.
pub fn load(gpa: Allocator, d: *const hip.Runtime, ck: *Checkpoint, c: *const config.Config, rank: ?quant.Rank, act: model.Kind, mtp_root: []const u8) !loader.Weights {
    if (rank != null) {
        std.log.err("Flash Next FP4-Local on HIP loads on one rank: tensor parallelism cuts no baked plane", .{});
        return error.Fp4LocalTensorParallel;
    }
    if (act != .bf16) {
        std.log.err("Flash Next FP4-Local on HIP keeps bf16 activations; this card's kernels take {t}", .{act});
        return error.Fp4LocalActivations;
    }
    var w: loader.Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var stage = try experts.Stage.init(d);
    defer stage.deinit();
    const l: Loader = .{
        .a = arena.allocator(),
        .ck = ck,
        .c = c,
        .up = .{ .r = d, .buffers = &w.buffers, .gpa = gpa },
        .stage = &stage,
        .around_one = try loader.aroundOne(arena.allocator(), ck, c.layers, true),
        .threads = try nvfp4_read.configured(),
    };
    w.layers = try gpa.alloc(model.Layer, c.layers);
    var ple: ?model.Ple = null;
    var direct = core.checkpoint.direct.Layers.init(gpa, ck, try core.checkpoint.loadReaders());
    defer direct.deinit();
    for (0..c.layers) |i| {
        _ = arena.reset(.retain_capacity);
        var lb: [2][128]u8 = undefined;
        var at: [2][144]u8 = undefined;
        const base = try names.exported(try tensors.layer(&lb[0], i), &at[0]);
        const stem = try l.a.dupe(u8, base);
        // the n-gram layer reads through the mapping: its table's shards are never read whole
        if (c.ple[i]) direct.mapped() else try direct.enter(
            try std.fmt.allocPrint(l.a, "{s}.", .{stem}),
            if (i + 1 < c.layers and !c.ple[i + 1]) try std.fmt.allocPrint(l.a, "{s}.", .{try names.exported(try tensors.layer(&lb[1], i + 1), &at[1])}) else null,
        );
        w.layers[i] = try l.layer(stem, try c.kind(i));
        if (c.ple[i]) ple = try l.ple(stem, i);
    }
    direct.mapped();
    _ = arena.reset(.retain_capacity);
    const embed = try l.table(try l.stored(tensors.embed), c.vocab, c.hidden);
    const head = try l.table(try l.stored(tensors.lm_head), c.vocab, c.hidden);
    w.model = .{
        .dims = try model.Dims.of(c, 1),
        .act = act,
        .embed = try l.put(embed),
        .layers = w.layers,
        .mix = try l.hc(try l.stored(tensors.mixer), false),
        .head = try l.put(head),
        .ple = ple orelse return error.NoPleLayer,
    };
    if (c.mtp.layers > 0) {
        const bf16_rows = try quant.sliceRows(l.a, head, try loader.draftSpans(l.a));
        const affine = @import("draft_affine.zig");
        const draft_rows = if (try affine.mode() == .affine4) try affine.affine4(l.a, bf16_rows) else bf16_rows;
        const nm = try loader.mtpNames(l.a, mtp_root);
        w.model.mtp = .{
            .e_norm = try l.scale(nm.norm_embedding),
            .h_norm = try l.scale(nm.norm_hidden),
            .fc_e = try l.proj(nm.fc_embedding),
            .fc_h = try l.proj(nm.fc_hidden),
            .layer = try l.layer(nm.layer0, c.mtp.kinds[0]),
            .mix = try l.hc(nm.mixer, false),
            .dims = try model.Dims.of(c, 1),
            .draft = try l.put(draft_rows),
        };
    }
    return w;
}

test "a stitched product's part views start at their rows' planes" {
    const whole: Projection = .{ .n = 10, .k = 256, .handle = .{ .mxfp8 = .{ .codes = 0x10000, .scales = 0x90000 } } };
    const v = try rowsOf(whole, 6, 4);
    try std.testing.expectEqual(@as(u32, 4), v.n);
    try std.testing.expectEqual(@as(u64, 0x10000 + 6 * 256), v.handle.mxfp8.codes);
    try std.testing.expectEqual(@as(u64, 0x90000 + 6 * 8), v.handle.mxfp8.scales);
    const dense: Projection = .{ .n = 10, .k = 64, .handle = .{ .dense = .{ .weight = 0x2000, .dtype = .bf16 } } };
    try std.testing.expectEqual(@as(u64, 0x2000 + 3 * 64 * 2), (try rowsOf(dense, 3, 7)).handle.dense.weight);
    try std.testing.expectError(error.UnexpectedTensor, rowsOf(.{ .n = 1, .k = 32, .handle = .{ .mlx = undefined } }, 0, 1));
}
