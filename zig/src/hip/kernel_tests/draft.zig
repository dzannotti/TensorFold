//! The DFlash2 drafter's kernels against host references: copies and the conv exact, norms and attention near fp64's.
const std = @import("std");
const rig = @import("rig.zig");
const dft = @import("../launches/draft.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const head_dim = dft.head_dim;

fn bfv(b: u16) f32 {
    return @floatCast(rig.bf16Value(b));
}

/// Round to nearest even, as the kernels do.
fn tobf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

fn rbf(x: f32) f32 {
    return bfv(tobf(x));
}

/// `n` bf16 values of [-scale, scale).
fn randoms(rng: *rig.Rng, n: usize, scale: f32) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*x| x.* = tobf(rng.unit() * scale);
    return out;
}

/// The reference of norm_rope4: rms norm with an fp32 weight, rotate-half at `pos`, both rounded.
fn normRope(comptime hd: usize, x: []const u16, w: []const f32, eps: f32, theta: f32, pos: i64, out: *[hd]f32) void {
    var sum: f32 = 0;
    for (0..hd) |d| sum += bfv(x[d]) * bfv(x[d]);
    const inv = 1.0 / @sqrt(sum / hd + eps);
    var n: [hd]f32 = undefined;
    for (0..hd) |d| n[d] = rbf(bfv(x[d]) * inv * w[d]);
    for (0..hd) |d| {
        const j = d % (hd / 2);
        // the angle in fp32, as every RoPE kernel of the engine forms it (far positions lose its low bits)
        const ang: f64 = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, theta, -@as(f32, @floatFromInt(j)) / (hd / 2));
        const c: f64 = @cos(ang);
        const s: f64 = @sin(ang);
        const y: f64 = if (d < hd / 2) n[d] * c - n[d + hd / 2] * s else n[d] * c + n[d - hd / 2] * s;
        out[d] = @floatCast(y);
    }
}

/// Within two bf16 steps of `want`, plus the angle's fp32 ulps at `pos` (the device's powf and sinf round their own way).
fn near(got: u16, want: f32, pos: i64) bool {
    const angle_ulps: f32 = @as(f32, @floatFromInt(pos)) * 0x1p-21;
    return @abs(bfv(got) - want) <= 2.0 / 128.0 * @max(@abs(want), 0.125) + angle_ulps;
}

test "gather: rows from anywhere, back to back" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd1 };
    const width = 96;
    const src = try randoms(&rng, 10 * width / 2, 1);
    defer gpa.free(src);
    var dsrc = try t.upload(src);
    defer dsrc.free();
    const picks = [_]usize{ 7, 2, 9 };
    var rows: [picks.len]dft.Row = undefined;
    for (picks, &rows) |p, *r| r.* = .{ .src = at(dsrc) + p * width, .ring = 0, .pos = 0 };
    var drows = try t.upload(&rows);
    defer drows.free();
    var out = try t.alloc(picks.len * width);
    defer out.free();
    try t.on.dftGather(at(drows), picks.len, width, at(out), t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    for (picks, 0..) |p, i| try std.testing.expectEqualSlices(u16, src[p * width / 2 ..][0 .. width / 2], got[i * width / 2 ..][0 .. width / 2]);
}

test "ring put: every layer's keys normed and rotated into slot pos % R, values transposed" {
    try ringPut(128);
    try ringPut(256);
}

fn ringPut(comptime hd: usize) !void {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd2 };
    const layers = 2;
    const kvh = 2;
    const kvw = kvh * hd;
    const R = 32;
    const n = 5;
    const stride = layers * 2 * kvw + 64; // a row wider than the keys and values it holds
    const eps: f32 = 1e-6;
    const theta: f32 = 10_000_000;
    const kv = try randoms(&rng, n * stride, 2);
    defer gpa.free(kv);
    var knorm: [layers * hd]f32 = undefined;
    for (&knorm) |*x| x.* = 1 + rng.unit() * 0.5;
    const ring_elems = layers * 2 * R * kvw;
    var rings = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(ring_elems * 2), try t.alloc(ring_elems * 2) };
    defer for (&rings) |*r| r.free();
    for (rings) |r| try r.fill8(0);
    // rows of two streams, one crossing its ring's end
    const where = [n]struct { s: usize, pos: i64 }{ .{ .s = 0, .pos = 30 }, .{ .s = 0, .pos = 31 }, .{ .s = 0, .pos = 32 }, .{ .s = 1, .pos = 3 }, .{ .s = 1, .pos = 70 } };
    var rows: [n]dft.Row = undefined;
    for (where, &rows) |w, *r| r.* = .{ .src = 0, .ring = at(rings[w.s]), .pos = w.pos };
    var dkv = try t.upload(kv);
    defer dkv.free();
    var drows = try t.upload(&rows);
    defer drows.free();
    var dnorm = try t.upload(&knorm);
    defer dnorm.free();
    try t.on.dftRingPut(at(dkv), stride, at(drows), n, at(dnorm), layers, kvh, R, eps, theta, hd, t.stream.handle);
    try t.stream.synchronize();
    const got = [2][]u16{ try rig.download(u16, rings[0]), try rig.download(u16, rings[1]) };
    defer for (got) |g| gpa.free(g);
    var want: [hd]f32 = undefined;
    for (where, 0..) |w, r| for (0..layers) |l| for (0..kvh) |h| {
        const slot: usize = @intCast(@mod(w.pos, R));
        const k = kv[r * stride + l * 2 * kvw + h * hd ..][0..hd];
        normRope(hd, k, knorm[l * hd ..][0..hd], eps, theta, w.pos, &want);
        const ring = got[w.s][l * 2 * R * kvw ..];
        for (0..hd) |d| try rig.expect(near(ring[slot * kvw + h * hd + d], want[d], w.pos), "key row {d} layer {d} head {d} dim {d}: {d} vs {d}", .{ r, l, h, d, bfv(ring[slot * kvw + h * hd + d]), want[d] });
        const vt = ring[R * kvw + h * hd * R ..];
        for (0..hd) |d| try std.testing.expectEqual(kv[r * stride + l * 2 * kvw + kvw + h * hd + d], vt[(slot / 16) * hd * 16 + d * 16 + slot % 16]);
    };
}

test "norm rope: block rows at their streams' positions, from strided rows" {
    const hd = head_dim;
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd3 };
    const block = 5;
    const streams = 2;
    const rows = block * streams;
    const heads = 3;
    const stride = heads * hd + 128;
    const eps: f32 = 1e-6;
    const theta: f32 = 10_000_000;
    const x = try randoms(&rng, rows * stride, 3);
    defer gpa.free(x);
    var w: [hd]f32 = undefined;
    for (&w) |*v| v.* = 1 + rng.unit() * 0.5;
    const desc = [streams]dft.Desc{ .{ .ring = 0, .begin = 0, .pos0 = 17 }, .{ .ring = 0, .begin = 0, .pos0 = 123456 } };
    var dx = try t.upload(x);
    defer dx.free();
    var dw = try t.upload(&w);
    defer dw.free();
    var dd = try t.upload(&desc);
    defer dd.free();
    var out = try t.alloc(rows * heads * hd * 2);
    defer out.free();
    try t.on.dftNormRope(at(dx), stride, at(dw), at(out), at(dd), block, rows, heads, eps, theta, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    var want: [hd]f32 = undefined;
    for (0..rows) |r| for (0..heads) |h| {
        const pos = desc[r / block].pos0 + @as(i64, @intCast(r % block));
        normRope(hd, x[r * stride + h * hd ..][0..hd], &w, eps, theta, pos, &want);
        for (0..hd) |d| try rig.expect(near(got[(r * heads + h) * hd + d], want[d], pos), "row {d} head {d} dim {d}: {d} vs {d}", .{ r, h, d, bfv(got[(r * heads + h) * hd + d]), want[d] });
    };
}

test "dynamic conv: both branches bit for bit, the lag stopping at each block's first row" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd4 };
    const block = 4;
    const rows = 3 * block;
    const D = 64;
    const group = 16;
    const groups = D / group;
    const x = try randoms(&rng, rows * D, 2);
    defer gpa.free(x);
    const dyn = try randoms(&rng, rows * 4 * groups, 1);
    defer gpa.free(dyn);
    const base = try randoms(&rng, 4 * D, 1);
    defer gpa.free(base);
    const res = try randoms(&rng, rows * D, 4);
    defer gpa.free(res);
    var dx = try t.upload(x);
    defer dx.free();
    var ddyn = try t.upload(dyn);
    defer ddyn.free();
    var dbase = try t.upload(base);
    defer dbase.free();
    var dres = try t.upload(res);
    defer dres.free();
    var out = try t.alloc(rows * D * 2);
    defer out.free();
    for ([_]u1{ 0, 1 }) |branch| {
        const with_res = branch == 1;
        try t.on.dftDynConv(at(dx), at(ddyn), at(dbase), if (with_res) at(dres) else 0, at(out), rows, D, group, branch, block, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        for (0..rows) |r| for (0..D) |c| {
            var total: f32 = 0;
            const lags: usize = if (r % block == 0) 1 else 2;
            for (0..lags) |l| {
                const xv = bfv(x[(r - l) * D + c]);
                total = rbf(total + rbf(bfv(base[(@as(usize, branch) * 2 + l) * D + c]) * xv));
                total = rbf(total + rbf(bfv(dyn[r * 4 * groups + (@as(usize, branch) * 2 + l) * groups + c / group]) * xv));
            }
            if (with_res) total = rbf(total + bfv(res[r * D + c]));
            try std.testing.expectEqual(tobf(total), got[r * D + c]);
        };
    }
}

test "block attention: each row's window over a wrapped ring, the context's start, and the whole block" {
    try blockAttn(128);
    try blockAttn(256);
}

fn blockAttn(comptime hd: usize) !void {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd5 };
    const layers = 2;
    const layer = 1;
    const streams = 3;
    const kvh = 2;
    const heads = 8;
    const G = heads / kvh;
    const block = 8;
    const rows = streams * block;
    const R = 48;
    const window = 40;
    const kvw = kvh * hd;
    const vstride = kvw + 32;
    const desc = [streams]dft.Desc{
        .{ .ring = 0, .begin = 0, .pos0 = 20 }, // short: no window cut
        .{ .ring = 0, .begin = 10, .pos0 = 100 }, // wrapped, the window cutting each row at its own place
        .{ .ring = 0, .begin = 5, .pos0 = 5 }, // nothing kept: the block alone
    };
    const ring_elems = layers * 2 * R * kvw;
    var ring_host: [streams][]u16 = undefined;
    for (&ring_host) |*r| r.* = try randoms(&rng, ring_elems, 1);
    defer for (ring_host) |r| gpa.free(r);
    var rings: [streams]@import("../memory.zig").DeviceBuffer = undefined;
    var descs = desc;
    for (&rings, ring_host, &descs) |*r, h, *d| {
        r.* = try t.upload(h);
        d.ring = at(r.*);
    }
    defer for (&rings) |*r| r.free();
    const q = try randoms(&rng, rows * heads * hd, 1);
    defer gpa.free(q);
    const kb = try randoms(&rng, rows * kvw, 1);
    defer gpa.free(kb);
    const vb = try randoms(&rng, rows * vstride, 1);
    defer gpa.free(vb);
    var dq = try t.upload(q);
    defer dq.free();
    var dkb = try t.upload(kb);
    defer dkb.free();
    var dvb = try t.upload(vb);
    defer dvb.free();
    var dd = try t.upload(&descs);
    defer dd.free();
    var out = try t.alloc(rows * heads * hd * 2);
    defer out.free();
    for ([_]usize{ 1, 3, 16 }) |splits| {
        var part = try t.alloc(dft.partBytes(streams, kvh, splits, hd));
        defer part.free();
        try t.on.dftAttn(.{ .q = at(dq), .kb = at(dkb), .vb = at(dvb), .vstride = vstride, .desc = at(dd), .layer = layer, .streams = streams, .block = block, .heads = heads, .kv_heads = kvh, .ring_rows = R, .window = window, .splits = splits, .part = at(part), .out = at(out), .hd = hd }, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        for (0..streams) |s| for (0..block) |m| for (0..heads) |h| {
            const d = desc[s];
            const kh = h / G;
            const ring = ring_host[s][layer * 2 * R * kvw ..];
            const qrow = q[((s * block + m) * heads + h) * hd ..][0..hd];
            var scores: [R + block]f64 = undefined;
            var vals: [R + block][]const u16 = undefined;
            var vt: [R + block]?usize = undefined; // a ring key's slot (values transposed); null: a block row
            var n: usize = 0;
            const first = @max(d.begin, d.pos0 + @as(i64, @intCast(m)) - window + 1);
            var p = first;
            while (p < d.pos0) : (p += 1) {
                const slot: usize = @intCast(@mod(p, R));
                var dot: f64 = 0;
                for (0..hd) |e| dot += bfv(qrow[e]) * bfv(ring[slot * kvw + kh * hd + e]);
                scores[n] = dot;
                vt[n] = slot;
                n += 1;
            }
            for (0..block) |j| {
                var dot: f64 = 0;
                for (0..hd) |e| dot += bfv(qrow[e]) * bfv(kb[((s * block + j) * kvh + kh) * hd + e]);
                scores[n] = dot;
                vt[n] = null;
                vals[n] = vb[(s * block + j) * vstride + kh * hd ..][0..hd];
                n += 1;
            }
            var mx: f64 = -std.math.inf(f64);
            for (scores[0..n]) |*x| {
                x.* /= @sqrt(@as(f64, hd));
                mx = @max(mx, x.*);
            }
            var sum: f64 = 0;
            for (scores[0..n]) |*x| {
                x.* = @exp(x.* - mx);
                sum += x.*;
            }
            for (0..hd) |e| {
                var acc: f64 = 0;
                for (0..n) |i| {
                    const v = if (vt[i]) |slot| ring[R * kvw + kh * hd * R + (slot / 16) * hd * 16 + e * 16 + slot % 16] else vals[i][e];
                    acc += scores[i] * bfv(v);
                }
                const want: f32 = @floatCast(acc / sum);
                const g = bfv(got[((s * block + m) * heads + h) * hd + e]);
                // P is rounded to bf16 before the values' product
                try rig.expect(@abs(g - want) <= 0.01 + 0.01 * @abs(want), "splits {d} stream {d} row {d} head {d} dim {d}: {d} vs {d}", .{ splits, s, m, h, e, g, want });
            }
        };
    }
}

test "bf16 rows: products of bf16 weights over every row count a hold takes, fp32 sums rounded once" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd6 };
    const n = 200; // not a whole number of 128-column blocks
    const k = 96;
    const w = try randoms(&rng, n * k, 1);
    defer gpa.free(w);
    var dw = try t.upload(w);
    defer dw.free();
    for ([_]usize{ 1, 8, 24, 64, 72 }) |rows| {
        const x = try randoms(&rng, rows * k, 2);
        defer gpa.free(x);
        var dx = try t.upload(x);
        defer dx.free();
        var out = try t.alloc(rows * n * 2);
        defer out.free();
        try t.on.dftBf16Rows(at(dx), at(dw), at(out), rows, n, k, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        for (0..rows) |r| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |e| acc += @as(f64, bfv(x[r * k + e])) * bfv(w[c * k + e]);
            const want: f32 = @floatCast(acc);
            try rig.expect(@abs(bfv(got[r * n + c]) - want) <= 1.0 / 128.0 * @max(@abs(want), 0.25), "rows {d}: ({d}, {d}) {d} vs {d}", .{ rows, r, c, bfv(got[r * n + c]), want });
        };
    }
}

test "q4 rows: a tiled affine-4 group-64 matrix's products over every row count a hold takes, each weight dequantized to bf16" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xd7 };
    const n = 224; // 7 blocks: the last wave of the second grid block has none
    const k = 192;
    const groups = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    for (words) |*w| w.* = @truncate(rng.next());
    const scales = try randoms(&rng, n * groups, 0.05);
    defer gpa.free(scales);
    const biases = try randoms(&rng, n * groups, 0.4);
    defer gpa.free(biases);
    // the row kernel's layout: 32-column blocks, a block's group 32 columns' 8 words back to back, tables likewise
    const tw = try gpa.alloc(u32, words.len);
    defer gpa.free(tw);
    const ts = try gpa.alloc(u16, scales.len);
    defer gpa.free(ts);
    const tb = try gpa.alloc(u16, biases.len);
    defer gpa.free(tb);
    for (0..n) |c| for (0..groups) |g| {
        const i = (c / 32 * groups + g) * 32 + c % 32;
        @memcpy(tw[i * 8 ..][0..8], words[c * (k / 8) + g * 8 ..][0..8]);
        ts[i] = scales[c * groups + g];
        tb[i] = biases[c * groups + g];
    };
    var dw = try t.upload(tw);
    defer dw.free();
    var ds = try t.upload(ts);
    defer ds.free();
    var db = try t.upload(tb);
    defer db.free();
    for ([_]usize{ 1, 8, 24, 64, 72 }) |rows| {
        const x = try randoms(&rng, rows * k, 2);
        defer gpa.free(x);
        var dx = try t.upload(x);
        defer dx.free();
        var out = try t.alloc(rows * n * 2);
        defer out.free();
        try t.on.dftQ4Rows(at(dx), at(dw), at(ds), at(db), at(out), rows, n, k, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        for (0..rows) |r| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |e| {
                const q: f32 = @floatFromInt((words[(c * k + e) / 8] >> @intCast(4 * (e % 8))) & 15);
                const g = c * groups + e / 64;
                acc += @as(f64, bfv(x[r * k + e])) * rbf(q * bfv(scales[g]) + bfv(biases[g]));
            }
            const want: f32 = @floatCast(acc);
            try rig.expect(@abs(bfv(got[r * n + c]) - want) <= 1.0 / 128.0 * @max(@abs(want), 0.25), "rows {d}: ({d}, {d}) {d} vs {d}", .{ rows, r, c, bfv(got[r * n + c]), want });
        };
    }
}

test "top-k merge: chunks' own top k merged give the whole row's, ties to the lower id, best first" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7c };
    const rows = 3;
    const chunks = 8;
    const chunk = 64;
    const k = 16;
    const n = chunks * chunk;
    // few distinct values: many ties, inside and across chunks
    const logits = try gpa.alloc(u16, rows * n);
    defer gpa.free(logits);
    for (logits) |*x| x.* = tobf(@round(rng.unit() * 6));
    const ks = [_]i32{ 0, k, k };
    var kc: [rows * chunks]i32 = undefined;
    for (&kc, 0..) |*x, i| x.* = ks[i / chunks];
    var dl = try t.upload(logits);
    defer dl.free();
    var dks = try t.upload(@as([]const i32, &ks));
    defer dks.free();
    var dkc = try t.upload(@as([]const i32, &kc));
    defer dkc.free();
    var ci = try t.alloc(rows * chunks * k * 4);
    defer ci.free();
    var cv = try t.alloc(rows * chunks * k * 2);
    defer cv.free();
    var ids = try t.alloc(rows * k * 4);
    defer ids.free();
    var vals = try t.alloc(rows * k * 2);
    defer vals.free();
    try t.on.tf_topk_rows(@ptrFromInt(at(dl)), rows * chunks, chunk, @ptrFromInt(at(dkc)), k, @ptrFromInt(at(ci)), @ptrFromInt(at(cv)), t.stream.handle);
    try t.on.dftTopkMerge(at(ci), at(cv), rows, chunks, k, chunk, at(dks), k, at(ids), at(vals), t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(i32, ids);
    defer gpa.free(got);
    for (1..rows) |r| {
        const row = logits[r * n ..][0..n];
        var order: [n]u32 = undefined;
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        const Desc = struct {
            row: []const u16,
            fn less(c: @This(), a: u32, b: u32) bool {
                const va = bfv(c.row[a]);
                const vb = bfv(c.row[b]);
                return va > vb or (va == vb and a < b);
            }
        };
        std.mem.sort(u32, &order, Desc{ .row = row }, Desc.less);
        for (0..k) |j| try std.testing.expectEqual(@as(i32, @intCast(order[j])), got[r * k + j]);
    }
}

test "mlx products: the drafter's tiles write the verify tile's bytes, alone, in pairs, padded and with the conv after" {
    const affine = @import("../launches/affine.zig");
    const Buf = @import("../memory.zig").DeviceBuffer;
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5eed };
    // ld: the drafter's copy stores rows of ld k, garbage past k (never read)
    const Case = struct { n: usize, k: usize, ld: usize = 0 };
    for ([_]Case{ .{ .n = 96, .k = 512 }, .{ .n = 80, .k = 768 }, .{ .n = 48, .k = 2560 }, .{ .n = 32, .k = 4096, .ld = 4352 }, .{ .n = 32, .k = 17408 } }) |cs| {
        const n = cs.n;
        const k = cs.k;
        const ld = if (cs.ld == 0) k else cs.ld;
        const groups = k / 64;
        var dw: [2]Buf = undefined;
        var ds: [2]Buf = undefined;
        var db: [2]Buf = undefined;
        var pw: [2]Buf = undefined;
        var ps: [2]Buf = undefined;
        var pb: [2]Buf = undefined;
        for (0..2) |i| {
            const words = try gpa.alloc(u32, n * ld / 8);
            defer gpa.free(words);
            for (words) |*w| w.* = @truncate(rng.next());
            const scales = try randoms(&rng, n * ld / 64, 0.05);
            defer gpa.free(scales);
            const biases = try randoms(&rng, n * ld / 64, 0.4);
            defer gpa.free(biases);
            pw[i] = try t.upload(words);
            ps[i] = try t.upload(scales);
            pb[i] = try t.upload(biases);
            // the verify tile's compact copy: each row's first k
            for (0..n) |r| {
                std.mem.copyForwards(u32, words[r * k / 8 ..][0 .. k / 8], words[r * ld / 8 ..][0 .. k / 8]);
                std.mem.copyForwards(u16, scales[r * groups ..][0..groups], scales[r * ld / 64 ..][0..groups]);
                std.mem.copyForwards(u16, biases[r * groups ..][0..groups], biases[r * ld / 64 ..][0..groups]);
            }
            dw[i] = try t.upload(words[0 .. n * k / 8]);
            ds[i] = try t.upload(scales[0 .. n * groups]);
            db[i] = try t.upload(biases[0 .. n * groups]);
        }
        defer for (0..2) |i| {
            dw[i].free();
            ds[i].free();
            db[i].free();
            pw[i].free();
            ps[i].free();
            pb[i].free();
        };
        for ([_]usize{ 1, 5, 16 }) |rows| {
            const x = try randoms(&rng, rows * k, 2);
            defer gpa.free(x);
            var dx = try t.upload(x);
            defer dx.free();
            var want: [2]Buf = undefined;
            for (0..2) |i| {
                want[i] = try t.alloc(rows * n * 2);
                const arg: affine.Arg = .{ .x = at(dx), .words = at(dw[i]), .scale = .{ .p = at(ds[i]), .kind = 1 }, .bias = .{ .p = at(db[i]), .kind = 1 }, .out = 0, .out16 = at(want[i]), .m = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .bits = 4, .group = 64, .fp16 = 0 };
                try t.on.affine.verifyGo(&t.r, arg, .{}, @intCast(n), t.stream.handle);
            }
            defer for (&want) |*w| w.free();
            try t.stream.synchronize();
            var out = try t.alloc(rows * n * 2);
            defer out.free();
            var out2 = try t.alloc(rows * n * 2);
            defer out2.free();
            const mats = [2]dft.Mat{
                .{ .words = at(pw[0]), .scale = at(ps[0]), .bias = at(pb[0]), .out = at(out), .n = @intCast(n), .ld = @intCast(ld) },
                .{ .words = at(pw[1]), .scale = at(ps[1]), .bias = at(pb[1]), .out = at(out2), .n = @intCast(n), .ld = @intCast(ld) },
            };
            const w1 = try rig.download(u16, want[1]);
            defer gpa.free(w1);
            const w0 = try rig.download(u16, want[0]);
            defer gpa.free(w0);
            // the conv variants: dynConv's branch 1 with a residual over the product
            const group = 16;
            const block: usize = if (rows == 16) 16 else 4;
            const dyn = try randoms(&rng, rows * 4 * (n / group), 1);
            defer gpa.free(dyn);
            const cbase = try randoms(&rng, 4 * n, 1);
            defer gpa.free(cbase);
            const res = try randoms(&rng, rows * n, 4);
            defer gpa.free(res);
            var ddyn = try t.upload(dyn);
            defer ddyn.free();
            var dbase = try t.upload(cbase);
            defer dbase.free();
            var dres = try t.upload(res);
            defer dres.free();
            var conv = try t.alloc(rows * n * 2);
            defer conv.free();
            try t.on.dftDynConv(at(want[0]), at(ddyn), at(dbase), at(dres), at(conv), rows, n, group, 1, block, t.stream.handle);
            try t.stream.synchronize();
            const wc = try rig.download(u16, conv);
            defer gpa.free(wc);
            const cv: dft.Conv = .{ .dyn = at(ddyn), .base = at(dbase), .res = at(dres), .group = group, .block = block };
            for (std.enums.values(dft.Kind)) |kind| {
                try out.fill8(0xff);
                try t.on.dftMlx(kind, at(dx), rows, k, 0, mats[0], mats[1], cv, t.stream.handle);
                try t.stream.synchronize();
                const got = try rig.download(u16, out);
                defer gpa.free(got);
                const ref = if (kind == .conv) wc else w0;
                for (got, ref, 0..) |g, r, i| try rig.expect(g == r, "{s} n {d} k {d} rows {d}: element {d} {x} vs {x}", .{ @tagName(kind), n, k, rows, i, g, r });
                if (kind == .pair) {
                    const got2 = try rig.download(u16, out2);
                    defer gpa.free(got2);
                    for (got2, w1, 0..) |g, r, i| try rig.expect(g == r, "pair b n {d} k {d} rows {d}: element {d} {x} vs {x}", .{ n, k, rows, i, g, r });
                }
            }
        }
    }
}

test "mlx products: the drafter's rms writes tf_rms's bytes" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7a };
    for ([_][2]usize{ .{ 16, 5120 }, .{ 3, 2048 }, .{ 32, 5000 } }) |rw| {
        const rows = rw[0];
        const width = rw[1];
        const x = try randoms(&rng, rows * width, 3);
        defer gpa.free(x);
        const w = try gpa.alloc(f32, width);
        defer gpa.free(w);
        for (w) |*v| v.* = rng.unit() * 2;
        var dx = try t.upload(x);
        defer dx.free();
        var dw = try t.upload(w);
        defer dw.free();
        var want = try t.alloc(rows * width * 2);
        defer want.free();
        var got = try t.alloc(rows * width * 2);
        defer got.free();
        try t.on.tf_rms(@ptrFromInt(at(dx)), @ptrFromInt(at(dw)), @ptrFromInt(at(want)), 2, @intCast(rows), @intCast(width), 1e-6, t.stream.handle);
        try t.on.dftRms(at(dx), at(dw), at(got), rows, width, 1e-6, t.stream.handle);
        try t.stream.synchronize();
        const a = try rig.download(u16, want);
        defer gpa.free(a);
        const b = try rig.download(u16, got);
        defer gpa.free(b);
        for (a, b, 0..) |p, q, i| try rig.expect(p == q, "rms {d}x{d}: element {d} {x} vs {x}", .{ rows, width, i, q, p });
    }
}

test "mlx product timings on the drafter's shapes at 16 rows (TF_DFT_BENCH=1)" {
    if (std.c.getenv("TF_DFT_BENCH") == null) return;
    const affine = @import("../launches/affine.zig");
    const t = try Rig.open(1 << 16);
    defer t.close();
    // ld: a row's k as stored (zero codes past k, never read)
    const Shape = struct { name: []const u8, n: usize, k: usize, ld: usize = 0 };
    const all = [_]Shape{
        .{ .name = "conv", .n = 1280, .k = 5120 },
        .{ .name = "conv/5376", .n = 1280, .k = 5120, .ld = 5376 },
        .{ .name = "qkv", .n = 6144, .k = 5120 },
        .{ .name = "qkv/5376", .n = 6144, .k = 5120, .ld = 5376 },
        .{ .name = "o/4096", .n = 5120, .k = 4096 },
        .{ .name = "o/4352", .n = 5120, .k = 4096, .ld = 4352 },
        .{ .name = "o+256", .n = 5120, .k = 4352 },
        .{ .name = "down", .n = 5120, .k = 17408 },
        .{ .name = "down/17664", .n = 5120, .k = 17408, .ld = 17664 },
        .{ .name = "gate", .n = 17408, .k = 5120 },
        .{ .name = "gate/5376", .n = 17408, .k = 5120, .ld = 5376 },
        .{ .name = "head", .n = 98592, .k = 5120 },
        .{ .name = "head/5376", .n = 98592, .k = 5120, .ld = 5376 },
        .{ .name = "selector", .n = 256, .k = 5120 },
    };
    const want: ?[]const u8 = if (std.c.getenv("TF_DFT_BENCH_SHAPE")) |e| std.mem.span(e) else null;
    const m = 16;
    const spread = 256 << 20;
    std.debug.print("\nshape        variant     us     GB/s\n", .{});
    for (all) |sh| {
        if (want) |w| if (std.mem.indexOf(u8, w, sh.name) == null) continue;
        const ld = if (sh.ld == 0) sh.k else sh.ld;
        const wbytes = sh.n * ld / 2;
        const tbytes = sh.n * ld / 64 * 2;
        const real = sh.n * sh.k / 2 + 2 * (sh.n * sh.k / 64 * 2);
        const copies = @max(2, spread / (wbytes + 2 * tbytes));
        var words = try t.alloc(copies * wbytes);
        defer words.free();
        var tables = try t.alloc(copies * 2 * tbytes);
        defer tables.free();
        var x = try t.alloc(m * sh.k * 2);
        defer x.free();
        var out = try t.alloc(m * sh.n * 2);
        defer out.free();
        try words.fill8(0x95);
        try tables.fill8(0x3c);
        try x.fill8(0x3f);
        // verify (the engine's), then each drafter tile alone and as a pair (twice the bytes, counted as two)
        const V = struct { own: bool, pair: bool };
        for ([_]V{ .{ .own = false, .pair = false }, .{ .own = true, .pair = false }, .{ .own = true, .pair = true } }) |vt| {
            if (!vt.own and ld != sh.k) continue;
            var times: [7]f64 = undefined;
            const launches = @max(copies, 8);
            for (0..times.len + 1) |r| {
                var ts: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                const t0 = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
                var i: usize = 0;
                while (i < launches) : (i += if (vt.pair) 2 else 1) {
                    const c = i % copies;
                    const c2 = (i + 1) % copies;
                    const w = at(words) + c * wbytes;
                    const sc = at(tables) + c * 2 * tbytes;
                    if (vt.own) {
                        const a: dft.Mat = .{ .words = w, .scale = sc, .bias = sc + tbytes, .out = at(out), .n = @intCast(sh.n), .ld = @intCast(ld) };
                        const b: dft.Mat = .{ .words = at(words) + c2 * wbytes, .scale = at(tables) + c2 * 2 * tbytes, .bias = at(tables) + c2 * 2 * tbytes + tbytes, .out = at(out), .n = @intCast(sh.n), .ld = @intCast(ld) };
                        try t.on.dftMlx(if (vt.pair) .pair else .plain, at(x), m, sh.k, 0, a, b, .{}, t.stream.handle);
                    } else {
                        const arg: affine.Arg = .{ .x = at(x), .words = w, .scale = .{ .p = sc, .kind = 1 }, .bias = .{ .p = sc + tbytes, .kind = 1 }, .out = 0, .out16 = at(out), .m = m, .n = @intCast(sh.n), .k = @intCast(sh.k), .bits = 4, .group = 64, .fp16 = 0 };
                        try t.on.affine.verifyGo(&t.r, arg, .{}, @intCast(sh.n), t.stream.handle);
                    }
                }
                try t.stream.synchronize();
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                const t1 = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
                if (r > 0) times[r - 1] = @as(f64, @floatFromInt(t1 - t0)) / 1e3 / @as(f64, @floatFromInt(launches));
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            const us = times[times.len / 2];
            std.debug.print("{s:<12} {s:<4}{s:<4} {d:>8.1} {d:>8.1}\n", .{ sh.name, if (vt.own) "own" else "verify", if (vt.pair) "+p" else "", us, @as(f64, @floatFromInt(real)) / us / 1e3 });
        }
    }
}

test "markov chain: each step the whole vocabulary's best of logit + w2 . w1[prev], bf16 and affine-4, the keys zeroed after" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x3a };
    const V = 2500; // a partial last workgroup
    const R = 256;
    const block = 3;
    const streams = 2;
    const logits = try randoms(&rng, streams * block * V, 4);
    defer gpa.free(logits);
    const w1 = try randoms(&rng, V * R, 0.3);
    defer gpa.free(w1);
    const w2 = try randoms(&rng, V * R, 0.3);
    defer gpa.free(w2);
    // affine-4: random codes, scales and biases (the value the kernel must read: s * code + b)
    const words = try gpa.alloc(u32, V * R / 8);
    defer gpa.free(words);
    for (words) |*w| w.* = @truncate(rng.next());
    const scales = try randoms(&rng, V * 4, 0.05);
    defer gpa.free(scales);
    const biases = try randoms(&rng, V * 4, 0.3);
    defer gpa.free(biases);
    const ids = [_]i32{ 7, -1, -1, 2499, -1, -1 };
    var dl = try t.upload(logits);
    defer dl.free();
    var d1 = try t.upload(w1);
    defer d1.free();
    var d2 = try t.upload(w2);
    defer d2.free();
    var dw = try t.upload(@as([]const u32, words));
    defer dw.free();
    var ds = try t.upload(scales);
    defer ds.free();
    var db = try t.upload(biases);
    defer db.free();
    var dids = try t.upload(@as([]const i32, &ids));
    defer dids.free();
    var keys = try t.alloc(streams * block * 8);
    defer keys.free();
    try keys.fill8(0);
    var out = try t.alloc(streams * block * 4);
    defer out.free();
    for ([_]bool{ false, true }) |q4| {
        const m: dft.Markov = if (q4) .{ .w1 = at(d1), .w2 = at(dw), .scales = at(ds), .biases = at(db), .vocab = V } else .{ .w1 = at(d1), .w2 = at(d2), .vocab = V };
        for (0..block) |i| try t.on.dftMarkovStep(at(dl), V, m, at(dids), at(keys), streams, block, i, t.stream.handle);
        try t.on.dftMarkovTake(at(keys), at(out), streams * block, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(i32, out);
        defer gpa.free(got);
        const left = try rig.download(u64, keys);
        defer gpa.free(left);
        for (left) |k| try std.testing.expectEqual(@as(u64, 0), k);
        for (0..streams) |s| {
            var prev: usize = @intCast(ids[s * block]);
            for (0..block) |i| {
                var best: f64 = -std.math.inf(f64);
                var score_of_got: f64 = 0;
                for (0..V) |tok| {
                    var dot: f64 = 0;
                    for (0..R) |r| {
                        const wv: f64 = if (q4) blk: {
                            const code: f64 = @floatFromInt((words[(tok * R + r) / 8] >> @intCast(4 * (r % 8))) & 15);
                            break :blk bfv(scales[tok * 4 + r / 64]) * code + bfv(biases[tok * 4 + r / 64]);
                        } else bfv(w2[tok * R + r]);
                        dot += wv * bfv(w1[prev * R + r]);
                    }
                    const sc = bfv(logits[(s * block + i) * V + tok]) + dot;
                    best = @max(best, sc);
                    if (tok == got[s * block + i]) score_of_got = sc;
                }
                // fp32 sums against fp64: the pick is the best up to rounding
                try rig.expect(score_of_got >= best - 1e-3, "q4 {} stream {d} step {d}: token {d} scores {d} of best {d}", .{ q4, s, i, got[s * block + i], score_of_got, best });
                prev = @intCast(got[s * block + i]);
            }
        }
    }
}
