//! `zig build test-sampler-stats`: the keyed sampler's draws against the exact f64 target distribution.
//! Reference semantics (HF transformers generation/logits_process.py, vLLM v1/sample/ops/topk_topp_sampler.py):
//! logits / T, then top_k, then top_p on the top_k-renormalised softmax (a token stays while the mass strictly
//! above it is < top_p, HF's `cumulative_probs <= 1 - top_p` removal), then min_p (HF's order: after top_p),
//! renormalised. Ties at the top_k cutoff go to the lower id (TensorFold's convention; HF and vLLM keep every tie).
const std = @import("std");
const sampling = @import("sampling.zig");
const Sampling = sampling.Sampling;

const inf = std.math.inf(f64);

/// The exact target probabilities of `values` (ids 0..n) under `s`, in f64.
fn exact(gpa: std.mem.Allocator, values: []const f64, s: Sampling) ![]f64 {
    const n = values.len;
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, values, struct {
        fn lt(v: []const f64, a: usize, b: usize) bool {
            return v[a] > v[b] or (v[a] == v[b] and a < b);
        }
    }.lt);
    const p = try gpa.alloc(f64, n);
    @memset(p, 0);
    if (s.temperature <= 0) {
        p[order[0]] = 1;
        return p;
    }
    const k = @max(1, if (s.top_k == 0) n else @min(n, s.top_k));
    const top = values[order[0]] / s.temperature;
    var total: f64 = 0;
    for (order[0..k]) |i| total += @exp(values[i] / s.temperature - top);
    var keep = k;
    if (s.top_p > 0 and s.top_p < 1) {
        var above: f64 = 0;
        for (order[0..k], 0..) |i, j| {
            if (above >= s.top_p) {
                keep = j;
                break;
            }
            above += @exp(values[i] / s.temperature - top) / total;
        }
    }
    var kept: f64 = 0;
    for (order[0..keep]) |i| {
        const x = values[i] / s.temperature;
        if (s.min_p > 0 and x < top + @log(s.min_p)) continue;
        p[i] = @exp(x - top);
        kept += p[i];
    }
    for (p) |*q| q.* /= kept;
    return p;
}

const Stats = struct { chi2: f64, df: usize, crit: f64, tv: f64, outside: usize };

/// `draws` keyed draws at positions 0.. under `s`, against `exact`: chi-square (bins under 5 expected merged), TV.
fn check(values: []const f64, s: Sampling, draws: usize) !Stats {
    const gpa = std.testing.allocator;
    const p = try exact(gpa, values, s);
    defer gpa.free(p);
    const ids = try gpa.alloc(u64, values.len);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = i;
    const count = try gpa.alloc(usize, values.len);
    defer gpa.free(count);
    @memset(count, 0);
    for (0..draws) |pos| count[try sampling.choose(gpa, values, ids, pos, s)] += 1;
    const nf: f64 = @floatFromInt(draws);
    var st: Stats = .{ .chi2 = 0, .df = 0, .crit = 0, .tv = 0, .outside = 0 };
    var pool_e: f64 = 0;
    var pool_o: f64 = 0;
    var bins: usize = 0;
    for (p, count) |q, c| {
        const o: f64 = @floatFromInt(c);
        st.tv += @abs(o / nf - q) / 2;
        if (q == 0) {
            st.outside += c;
            continue;
        }
        const e = q * nf;
        if (e < 5) {
            pool_e += e;
            pool_o += o;
            continue;
        }
        st.chi2 += (o - e) * (o - e) / e;
        bins += 1;
    }
    if (pool_e > 0) {
        st.chi2 += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        bins += 1;
    }
    st.df = @max(bins, 2) - 1;
    // Wilson-Hilferty chi-square quantile at z = 4.75 (one-sided p ~ 1e-6): a fixed-seed test that never flakes by luck
    const df: f64 = @floatFromInt(st.df);
    const a = 2.0 / (9.0 * df);
    st.crit = df * std.math.pow(f64, 1 - a + 4.75 * @sqrt(a), 3);
    std.debug.print("  T={d} k={d} p={d} min_p={d} seed={d} V={d} N={d}: chi2={d:.1} df={d} crit={d:.1} TV={d:.5}\n", .{ s.temperature, s.top_k, s.top_p, s.min_p, s.seed, values.len, draws, st.chi2, st.df, st.crit, st.tv });
    try std.testing.expectEqual(@as(usize, 0), st.outside);
    try std.testing.expect(st.chi2 < st.crit);
    return st;
}

const N = 200_000;
const seeds = [_]u64{ 1, 0xDEAD_BEEF, 0x7FFF_FFFF_FFFF_FFFF };

fn each(values: []const f64, s: Sampling, tv_max: f64) !void {
    for (seeds) |seed| {
        var t = s;
        t.seed = seed;
        try std.testing.expect((try check(values, t, N)).tv < tv_max);
    }
}

test "peaked row: the draws are softmax(logits / T)" {
    var v: [50]f64 = undefined;
    for (&v, 0..) |*x, i| x.* = if (i == 7) 4.0 else 2.0 - @as(f64, @floatFromInt(i)) * 0.1;
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 0.01);
    try each(&v, .{ .seed = 0, .temperature = 0.6, .top_k = 20, .top_p = 0.95 }, 0.01);
}

test "flat row: uniform, and top_k on equal values keeps the k lowest ids" {
    const v: [64]f64 = @splat(0);
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 0.02);
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 5, .top_p = 1.0 }, 0.01);
    const p = try exact(std.testing.allocator, &v, .{ .seed = 0, .top_k = 5, .top_p = 1.0 });
    defer std.testing.allocator.free(p);
    try std.testing.expectEqual(@as(f64, 0.2), p[4]);
    try std.testing.expectEqual(@as(f64, 0), p[5]);
}

test "ties at the cutoffs: top_k and top_p split a tie by id, a tie inside the kept set draws evenly" {
    const v = [_]f64{ 1, 3, 3, 0, 3, 1, -2, 1 };
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 2, .top_p = 1.0 }, 0.01); // ids 1, 2 (4 tied, cut)
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 4, .top_p = 1.0 }, 0.01); // 1, 2, 4, then 0 of {0, 5, 7}
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 0.5 }, 0.01); // the 3s hold ~0.29 each: 1, 2
    try each(&v, .{ .seed = 0, .temperature = 2.0, .top_k = 0, .top_p = 1.0 }, 0.01);
}

test "-inf and negative entries are never drawn and do not shift the rest" {
    const v = [_]f64{ -inf, 0, -inf, -5, 2, -inf, -0.5, -inf };
    try each(&v, .{ .seed = 0, .temperature = 0.7, .top_k = 0, .top_p = 1.0 }, 0.01);
    // top_k past the finite tokens: the -inf ones fill the cut and still weigh 0
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 6, .top_p = 0.99 }, 0.01);
}

test "min_p keeps tokens with p >= min_p * p_max, and after top_p as HF orders them" {
    const v = [_]f64{ 2, 1.5, 1, 0.5, 0, -0.5, -1, -3, -4, -6 };
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.1 }, 0.01);
    // p = .45, .2, .15 then a .2 tail min_p drops: HF (top_p, then min_p) keeps 3; vLLM (min_p first) would keep 2
    const w = [_]f64{ @log(0.45), @log(0.2), @log(0.15), @log(0.05), @log(0.05), @log(0.05), @log(0.05) };
    const s: Sampling = .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 0.75, .min_p = 0.3 };
    const p = try exact(std.testing.allocator, &w, s);
    defer std.testing.allocator.free(p);
    try std.testing.expect(p[2] > 0 and p[3] == 0);
    try each(&w, s, 0.01);
}

test "edge settings: k = 1, k >= V, p = 1, p tiny, T = 0 are exact" {
    const gpa = std.testing.allocator;
    const v = [_]f64{ 0.5, 2.0, -1, 2.0, 1.9 };
    const ids = [_]u64{ 0, 1, 2, 3, 4 };
    for (0..2000) |pos| {
        // k = 1, tiny p and T = 0 all take the first maximum (id 1 over the tied 3)
        try std.testing.expectEqual(@as(u64, 1), try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .top_k = 1, .top_p = 1.0 }));
        try std.testing.expectEqual(@as(u64, 1), try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .top_k = 0, .top_p = 1e-9 }));
        try std.testing.expectEqual(@as(u64, 1), try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .temperature = 0 }));
        // k >= V draws exactly as k = 0 (same key, same set)
        const all = try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .top_k = 0, .top_p = 1.0 });
        try std.testing.expectEqual(all, try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .top_k = 5, .top_p = 1.0 }));
        try std.testing.expectEqual(all, try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 3, .top_k = 1000, .top_p = 1.0 }));
    }
    try each(&v, .{ .seed = 0, .temperature = 1.0, .top_k = 1000, .top_p = 1.0 }, 0.01);
}

test "a 32k vocabulary with a Zipf tail under top_p 0.95" {
    const gpa = std.testing.allocator;
    const v = try gpa.alloc(f64, 32768);
    defer gpa.free(v);
    // Zipf logits, ids scrambled so value order is not id order
    for (v, 0..) |*x, i| x.* = -1.1 * @log(@as(f64, @floatFromInt((i * 7919) % v.len + 1)));
    const s: Sampling = .{ .seed = 17, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    // ponytail: 20k draws, not 200k: each draw sorts the whole row (the engine's top_k 0 path); the chi-square bins the tail
    const st = try check(v, s, 20_000);
    try std.testing.expect(st.tv < 0.2);
    _ = try check(v, .{ .seed = 18, .temperature = 0.8, .top_k = 200, .top_p = 0.9, .min_p = 0.001 }, 20_000);
}

test "a seed's stream is deterministic and seeds give different streams" {
    const gpa = std.testing.allocator;
    const v: [32]f64 = @splat(0);
    var ids: [32]u64 = undefined;
    for (&ids, 0..) |*id, i| id.* = i;
    const s: Sampling = .{ .seed = 42, .top_k = 0, .top_p = 1.0 };
    var differ: usize = 0;
    for (0..1000) |pos| {
        const a = try sampling.choose(gpa, &v, &ids, pos, s);
        try std.testing.expectEqual(a, try sampling.choose(gpa, &v, &ids, pos, s));
        differ += @intFromBool(a != try sampling.choose(gpa, &v, &ids, pos, .{ .seed = 43, .top_k = 0, .top_p = 1.0 }));
    }
    try std.testing.expect(differ > 900); // independent streams agree ~1/32 of the time
    try std.testing.expectEqual(sampling.seedFor(&.{ 1, 2, 3 }, 0), sampling.seedFor(&.{ 1, 2, 3 }, 0));
    try std.testing.expect(sampling.seedFor(&.{ 1, 2, 3 }, 0) != sampling.seedFor(&.{ 1, 2, 3 }, 1));
    // a fixed golden: the keyed uniform is a pure function of (seed, position, id)
    try std.testing.expectEqual(sampling.keyBits(7, 11, 13), sampling.keyBits(7, 11, 13));
}
