//! ggml's block quantizations as a GGUF file stores them (ggml-org/llama.cpp ggml/src/ggml-common.h, the block_*
//! structs): each projection row [K] is K / block blocks, read in place from the mapped file. The reference is
//! ggml's dequantize_row_* (ggml/src/ggml-quants.c), reproduced in fp32 bit for bit and widened.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const container = @import("../gguf.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const Type = container.Type;

pub const id: types.Format = .gguf;
pub const decoder: types.Decoder = .gguf;

/// The types the kernels decode, as a projection's key suffix names them (`<key>.ggml.<type>`).
pub const kinds = [_]Type{ .q4_k, .q5_k, .q6_k, .q8_0, .iq4_xs, .iq4_nl, .q3_k, .iq3_s };

/// A GGUF checkpoint declares nothing in config.json: its files are the declaration (the loader sets it).
pub const Config = struct {};

pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = .{ a, src };
    return null;
}

/// A projection as stored: its rows' blocks, u8 [N, K / block * block bytes].
pub const Host = struct {
    blocks: Tensor,
    type: Type,
    k: usize,
};

/// The key a projection of type `t` is stored under.
pub fn keyOf(buf: []u8, key: []const u8, t: Type) []const u8 {
    return join(buf, &.{ key, ".ggml.", @tagName(t) });
}

pub fn matches(t: anytype, key: []const u8) bool {
    return kindOf(t, key) != null;
}

fn kindOf(t: anytype, key: []const u8) ?Type {
    var buf: [256]u8 = undefined;
    inline for (kinds) |k| if (t.has(keyOf(&buf, key, k))) return k;
    return null;
}

pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    _ = a;
    const kind = kindOf(t, key) orelse return refuse(key, "no GGUF type the kernels decode");
    var buf: [256]u8 = undefined;
    const b = try t.get(keyOf(&buf, key, kind));
    const blk = kind.block().?;
    if (b.dtype != .u8 or b.rank != 2 or b.shape[1] % blk.bytes != 0) return refuse(key, "GGUF blocks must be u8 [N, whole blocks]");
    const k = b.shape[1] / blk.bytes * blk.elems;
    if (k % 256 != 0) return refuse(key, "GGUF projections need K a multiple of 256");
    return .{ .blocks = b, .type = kind, .k = k };
}

pub fn rows(h: Host) usize {
    return h.blocks.shape[0];
}

pub fn bytes(h: Host) usize {
    return h.blocks.bytes.len;
}

/// Both halves along K: each row's first and last half of its blocks, copied.
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const row = p.blocks.shape[1];
    if ((p.k / 2) % 256 != 0) return refuse("fc", "GGUF halves must keep whole 256-element runs");
    var out: [2]Host = .{ p, p };
    for (&out, 0..) |*o, i| {
        o.blocks = try slice.takeCols(a, p.blocks, i * row / 2, (i + 1) * row / 2);
        o.k = p.k / 2;
    }
    return out;
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    var h = p;
    h.blocks = try slice.takeRows(a, p.blocks, spans);
    return h;
}

fn noCut(what: []const u8) slice.Error {
    std.log.err("{s}: GGUF projections have no column or expert cut here", .{what});
    return error.UnevenSplit;
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    _ = .{ a, p, r };
    return noCut(what);
}

pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    _ = .{ a, p, r, part };
    return noCut("experts");
}

pub const Device = struct { blocks: types.Buf, type: Type, k: usize };

pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{ .blocks = try u.tensor(h.blocks), .type = h.type, .k = h.k };
}

/// What the kernels read: the rows' blocks, ggml's type number, and the stored row's bytes.
pub const Matrix = struct { blocks: u64, type: Type, row_bytes: u32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    if (stacked) return error.UnsupportedTables;
    return .{ .n = @intCast(d.blocks.dim(0)), .k = @intCast(d.k), .matrix = .{ .blocks = d.blocks.ptr, .type = d.type, .row_bytes = @intCast(d.blocks.dim(1)) } };
}

pub fn elements(h: Host) usize {
    return rows(h) * h.k;
}

/// fp64 W as rows of outputs [N, K]: ggml's dequantize_row_* in fp32, widened.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const k = h.k;
    if (out.len != rows(h) * k) return error.UnexpectedTensor;
    const row_bytes = h.blocks.shape[1];
    var tmp: [256]f32 = undefined;
    const blk = h.type.block().?;
    for (0..rows(h)) |r| {
        var at: usize = 0;
        while (at < k) : (at += blk.elems) {
            const src = h.blocks.bytes[r * row_bytes + at / blk.elems * blk.bytes ..][0..blk.bytes];
            dequantBlock(h.type, src, tmp[0..blk.elems]);
            for (tmp[0..blk.elems], 0..) |v, i| out[r * k + at + i] = v;
        }
    }
}

fn half(b: []const u8, at: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[at..][0..2], .little))));
}

/// get_scale_min_k4: sub-block j's 6-bit scale and min of a q4_K / q5_K block's 12 scale bytes.
pub fn scaleMinK4(j: usize, q: []const u8) [2]u8 {
    if (j < 4) return .{ q[j] & 63, q[j + 4] & 63 };
    return .{ (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4) };
}

/// q3_K's 16 signed 6-bit scales (the `aux` unpacking of dequantize_row_q3_K), before the -32.
pub fn scalesQ3K(sc: []const u8) [16]i8 {
    const k1: u32 = 0x03030303;
    const k2: u32 = 0x0f0f0f0f;
    const a0 = std.mem.readInt(u32, sc[0..4], .little);
    const a1 = std.mem.readInt(u32, sc[4..8], .little);
    const tmp = std.mem.readInt(u32, sc[8..12], .little);
    const aux = [4]u32{
        (a0 & k2) | (((tmp >> 0) & k1) << 4),
        (a1 & k2) | (((tmp >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
    };
    var out: [16]i8 = undefined;
    for (aux, 0..) |w, i| for (0..4) |j| {
        out[4 * i + j] = @bitCast(@as(u8, @truncate(w >> @intCast(8 * j))));
    };
    return out;
}

/// ggml's kvalues_iq4nl.
pub const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

/// One block of type `t` as dequantize_row_* writes it.
pub fn dequantBlock(t: Type, b: []const u8, y: []f32) void {
    switch (t) {
        .q8_0 => {
            const d = half(b, 0);
            for (0..32) |j| y[j] = @as(f32, @floatFromInt(@as(i8, @bitCast(b[2 + j])))) * d;
        },
        .iq4_nl => {
            const d = half(b, 0);
            for (0..16) |j| {
                y[j] = d * @as(f32, @floatFromInt(kvalues_iq4nl[b[2 + j] & 0xf]));
                y[j + 16] = d * @as(f32, @floatFromInt(kvalues_iq4nl[b[2 + j] >> 4]));
            }
        },
        .q4_k, .q5_k => {
            const d = half(b, 0);
            const min = half(b, 2);
            const sc = b[4..16];
            const five = t == .q5_k;
            const qh = b[16..48];
            const qs = if (five) b[48..176] else b[16..144];
            for (0..4) |c| {
                const s1 = scaleMinK4(2 * c, sc);
                const s2 = scaleMinK4(2 * c + 1, sc);
                const d1 = d * @as(f32, @floatFromInt(s1[0]));
                const m1 = min * @as(f32, @floatFromInt(s1[1]));
                const d2 = d * @as(f32, @floatFromInt(s2[0]));
                const m2 = min * @as(f32, @floatFromInt(s2[1]));
                for (0..32) |l| {
                    const q = qs[32 * c + l];
                    var lo: u32 = q & 0xF;
                    var hi: u32 = q >> 4;
                    if (five) {
                        if (qh[l] & (@as(u8, 1) << @intCast(2 * c)) != 0) lo += 16;
                        if (qh[l] & (@as(u8, 2) << @intCast(2 * c)) != 0) hi += 16;
                    }
                    y[64 * c + l] = d1 * @as(f32, @floatFromInt(lo)) - m1;
                    y[64 * c + 32 + l] = d2 * @as(f32, @floatFromInt(hi)) - m2;
                }
            }
        },
        .q6_k => {
            const ql = b[0..128];
            const qh = b[128..192];
            const d = half(b, 208);
            for (0..2) |n| for (0..32) |l| {
                const is = l / 16;
                const l0 = ql[64 * n + l];
                const l1 = ql[64 * n + l + 32];
                const h = qh[32 * n + l];
                const q = [4]i32{ (l0 & 0xF) | (((h >> 0) & 3) << 4), (l1 & 0xF) | (((h >> 2) & 3) << 4), (l0 >> 4) | (((h >> 4) & 3) << 4), (l1 >> 4) | (((h >> 6) & 3) << 4) };
                for (0..4) |i| {
                    const sc: i8 = @bitCast(b[192 + 8 * n + is + 2 * i]);
                    y[128 * n + l + 32 * i] = d * @as(f32, @floatFromInt(sc)) * @as(f32, @floatFromInt(q[i] - 32));
                }
            };
        },
        .q3_k => {
            const hm = b[0..32];
            const qs = b[32..96];
            const scales = scalesQ3K(b[96..108]);
            const d = half(b, 108);
            var is: usize = 0;
            var m: u8 = 1;
            for (0..2) |n| {
                for (0..4) |j| {
                    for (0..2) |hf| {
                        const dl = d * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                        is += 1;
                        for (0..16) |l0| {
                            const l = 16 * hf + l0;
                            const q: i32 = @as(i32, (qs[32 * n + l] >> @intCast(2 * j)) & 3) - @as(i32, if (hm[l] & m != 0) 0 else 4);
                            y[128 * n + 32 * j + l] = dl * @as(f32, @floatFromInt(q));
                        }
                    }
                    m <<= 1;
                }
            }
        },
        .iq4_xs => {
            const d = half(b, 0);
            const sh = std.mem.readInt(u16, b[2..4], .little);
            for (0..8) |ib| {
                const ls: i32 = ((b[4 + ib / 2] >> @intCast(4 * (ib % 2))) & 0xf) | @as(i32, @intCast(((sh >> @intCast(2 * ib)) & 3) << 4));
                const dl = d * @as(f32, @floatFromInt(ls - 32));
                for (0..16) |j| {
                    const q = b[8 + 16 * ib + j];
                    y[32 * ib + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0xf]));
                    y[32 * ib + j + 16] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4]));
                }
            }
        },
        .iq3_s => {
            const d = half(b, 0);
            const qs = b[2..66];
            const qh = b[66..74];
            const signs = b[74..106];
            const sc = b[106..110];
            for (0..8) |ib| {
                const db = d * @as(f32, @floatFromInt(1 + 2 * @as(u32, (sc[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf)));
                for (0..4) |l| for (0..2) |g| {
                    const idx: usize = qs[8 * ib + 2 * l + g] | ((@as(usize, qh[ib]) << @intCast(8 - 2 * l - g)) & 256);
                    const grid = iq3s_grid[idx];
                    for (0..4) |j| {
                        const v: f32 = @floatFromInt(@as(u8, @truncate(grid >> @intCast(8 * j))));
                        const neg = signs[4 * ib + l] & (@as(u8, 1) << @intCast(j + 4 * g)) != 0;
                        y[32 * ib + 8 * l + 4 * g + j] = db * v * @as(f32, if (neg) -1.0 else 1.0);
                    }
                };
            }
        },
        else => unreachable,
    }
}

/// ggml's iq3s_grid (ggml/src/ggml-common.h): four 8-bit magnitudes a word.
pub const iq3s_grid = [512]u32{
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

/// A checkpoint as `read` sees it: tensors by name.
const Fixture = struct {
    names: []const []const u8,
    tensors: []const Tensor,

    pub fn has(f: Fixture, name: []const u8) bool {
        return f.find(name) != null;
    }

    pub fn get(f: Fixture, name: []const u8) error{MissingTensor}!Tensor {
        return f.tensors[f.find(name) orelse return error.MissingTensor];
    }

    fn find(f: Fixture, name: []const u8) ?usize {
        for (f.names, 0..) |x, i| if (std.mem.eql(u8, x, name)) return i;
        return null;
    }
};

test "a q4_K projection reads its type and K, halves its rows' blocks, and dequantizes as ggml" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // two rows of two blocks: d = 1, dmin = 0.5, scale j and min 1 for every sub-block j < 4 (scales[j] = j, [j+4] = 1)
    var raw: [2 * 2 * 144]u8 = @splat(0);
    for (0..4) |blk| {
        const b = raw[blk * 144 ..][0..144];
        std.mem.writeInt(u16, b[0..2], @bitCast(@as(f16, 1.0)), .little);
        std.mem.writeInt(u16, b[2..4], @bitCast(@as(f16, 0.5)), .little);
        for (0..4) |j| {
            b[4 + j] = @intCast(j);
            b[8 + j] = 1;
        }
        for (b[16..144], 0..) |*q, i| q.* = @intCast((i + blk) % 256);
    }
    const fx: Fixture = .{ .names = &.{"p.ggml.q4_k"}, .tensors = &.{.{ .dtype = .u8, .rank = 2, .shape = .{ 2, 288, 1, 1, 1 }, .bytes = &raw }} };
    try std.testing.expect(matches(fx, "p"));
    const h = try read(a, fx, "p");
    try std.testing.expectEqual(@as(usize, 512), h.k);
    const w = try a.alloc(f64, 2 * 512);
    try reference(h, w);
    // block 0, element 1: sub-block 0, low nibble of qs[1] = 1, scale 0 -> -0.5; element 32 + 1: high nibble of
    // qs[1] (0), scale 1 -> -0.5; element 64 + 3: sub-block 2 (scale 2), low nibble of qs[35] = 3 -> 6 - 0.5
    try std.testing.expectEqual(@as(f64, -0.5), w[1]);
    try std.testing.expectEqual(@as(f64, -0.5), w[33]);
    try std.testing.expectEqual(@as(f64, 5.5), w[67]);
    const pair = try halves(a, h);
    const part = try a.alloc(f64, 2 * 256);
    try reference(pair[1], part);
    for (0..2) |r| for (0..256) |c| try std.testing.expectEqual(w[r * 512 + 256 + c], part[r * 256 + c]);
}

test "the q3_K scale unpacking and q6_K's signed codes" {
    var sc: [12]u8 = undefined;
    for (&sc, 0..) |*v, i| v.* = @intCast(17 * i + 3);
    const s = scalesQ3K(&sc);
    // scale 0: low nibble of byte 0 (3) | bits 0-1 of byte 8 (139 & 3 = 3) << 4
    try std.testing.expectEqual(@as(i8, 3 | (3 << 4)), s[0]);
    var b: [210]u8 = @splat(0);
    b[192] = 2;
    std.mem.writeInt(u16, b[208..210], @bitCast(@as(f16, 0.25)), .little);
    var y: [256]f32 = undefined;
    dequantBlock(.q6_k, &b, &y);
    try std.testing.expectEqual(@as(f32, 0.25 * 2 * -32), y[0]);
}
