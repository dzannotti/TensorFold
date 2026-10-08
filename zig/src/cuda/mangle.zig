//! CUDA C++ kernel symbols as hipcc mangles them: CUDA's bf16, fp8 and vector types renamed to HIP's, and the
//! Itanium substitutions (S_, S0_, ...) recounted, since HIP_vector_type<T, N> adds candidates CUDA's uint4 lacks.
//! Covers what kernel signatures use: nested and template names, literals, qualifiers, builtins and template params.

const std = @import("std");

const Kind = enum { builtin, source, nested, template, qual, literal, tparam };

/// A name or type; `a` is the prefix/name/inner node, `args` a range of `Tree.list` (index 0 is no node).
const Node = struct { kind: Kind, text: []const u8 = "", a: u16 = 0, args: [2]u16 = .{ 0, 0 } };

const Bad = error{Unsupported};

const Tree = struct {
    src: []const u8,
    pos: usize = 0,
    nodes: [512]Node = undefined,
    n: u16 = 1,
    list: [512]u16 = undefined,
    nl: u16 = 0,
    subs: [256]u16 = undefined,
    ns: u16 = 0,
    fn_name: u16 = 0,
    params: [2]u16 = .{ 0, 0 },

    fn add(t: *Tree, node: Node) Bad!u16 {
        if (t.n == t.nodes.len) return error.Unsupported;
        t.nodes[t.n] = node;
        t.n += 1;
        return t.n - 1;
    }

    fn sub(t: *Tree, i: u16) Bad!void {
        if (t.ns == t.subs.len) return error.Unsupported;
        t.subs[t.ns] = i;
        t.ns += 1;
    }

    fn peek(t: *const Tree) u8 {
        return if (t.pos < t.src.len) t.src[t.pos] else 0;
    }

    fn eat(t: *Tree, c: u8) bool {
        if (t.peek() != c) return false;
        t.pos += 1;
        return true;
    }

    fn number(t: *Tree) Bad!usize {
        const start = t.pos;
        while (std.ascii.isDigit(t.peek())) t.pos += 1;
        return std.fmt.parseInt(usize, t.src[start..t.pos], 10) catch error.Unsupported;
    }

    fn source(t: *Tree) Bad!u16 {
        const len = try t.number();
        if (len == 0 or t.pos + len > t.src.len) return error.Unsupported;
        defer t.pos += len;
        return t.add(.{ .kind = .source, .text = t.src[t.pos..][0..len] });
    }

    /// S_ is the first candidate, S<base 36>_ the ones after it.
    fn substitution(t: *Tree) Bad!u16 {
        if (!t.eat('S') or std.ascii.isLower(t.peek())) return error.Unsupported;
        var i: usize = 0;
        if (!t.eat('_')) {
            while (true) {
                const c = t.peek();
                t.pos += 1;
                if (c == '_') break;
                const v: usize = if (std.ascii.isDigit(c)) c - '0' else if (std.ascii.isUpper(c)) c - 'A' + 10 else return error.Unsupported;
                i = i * 36 + v;
            }
            i += 1;
        }
        if (i >= t.ns) return error.Unsupported;
        return t.subs[i];
    }

    fn templateArgs(t: *Tree) Bad![2]u16 {
        if (!t.eat('I')) return error.Unsupported;
        var tmp: [64]u16 = undefined;
        var k: usize = 0;
        while (!t.eat('E')) {
            if (k == tmp.len) return error.Unsupported;
            if (t.eat('L')) {
                const ty = try t.typ();
                const start = t.pos;
                while (t.peek() != 'E' and t.peek() != 0) t.pos += 1;
                if (!t.eat('E')) return error.Unsupported;
                tmp[k] = try t.add(.{ .kind = .literal, .a = ty, .text = t.src[start .. t.pos - 1] });
            } else tmp[k] = try t.typ();
            k += 1;
        }
        return t.store(tmp[0..k]);
    }

    fn store(t: *Tree, items: []const u16) Bad![2]u16 {
        if (t.nl + items.len > t.list.len) return error.Unsupported;
        const start = t.nl;
        @memcpy(t.list[start..][0..items.len], items);
        t.nl += @intCast(items.len);
        return .{ start, t.nl };
    }

    /// A function's own name (`top`) is not a candidate, nor its specialization; its template name is.
    fn name(t: *Tree, top: bool) Bad!u16 {
        const nested = t.eat('N');
        var node: u16 = 0;
        while (true) {
            const first = node == 0;
            if (t.peek() == 'S') {
                node = try t.substitution();
            } else if (std.ascii.isDigit(t.peek())) {
                const s = try t.source();
                node = if (first) s else try t.add(.{ .kind = .nested, .a = node, .text = t.nodes[s].text });
                if (t.peek() == 'I' or !(top and (!nested or t.peek() == 'E'))) try t.sub(node);
            } else return error.Unsupported;
            if (t.peek() == 'I') {
                const args = try t.templateArgs();
                node = try t.add(.{ .kind = .template, .a = node, .args = args });
                if (!(top and (!nested or t.peek() == 'E'))) try t.sub(node);
            }
            if (!nested or t.eat('E')) return node;
        }
    }

    fn typ(t: *Tree) Bad!u16 {
        const c = t.peek();
        if (std.mem.indexOfScalar(u8, "vbchastijlmxynofdegzw", c) != null) {
            t.pos += 1;
            return t.add(.{ .kind = .builtin, .text = t.src[t.pos - 1 .. t.pos] });
        }
        if (c == 'D') {
            const start = t.pos;
            t.pos += 2;
            if (t.pos > t.src.len) return error.Unsupported;
            if (t.src[t.pos - 1] == 'F') {
                while (std.ascii.isDigit(t.peek())) t.pos += 1;
                if (!t.eat('_') and !t.eat('b')) return error.Unsupported;
            } else if (std.mem.indexOfScalar(u8, "hnisu", t.src[t.pos - 1]) == null) return error.Unsupported;
            return t.add(.{ .kind = .builtin, .text = t.src[start..t.pos] });
        }
        if (c == 'P' or c == 'R' or c == 'O' or c == 'K' or c == 'V' or c == 'r') {
            const start = t.pos;
            t.pos += 1;
            if (c != 'P' and c != 'R' and c != 'O') while (t.peek() == 'K' or t.peek() == 'V' or t.peek() == 'r') {
                t.pos += 1;
            };
            const text = t.src[start..t.pos];
            const inner = try t.typ();
            const q = try t.add(.{ .kind = .qual, .text = text, .a = inner });
            try t.sub(q);
            return q;
        }
        if (c == 'T') {
            const start = t.pos;
            t.pos += 1;
            while (t.peek() != '_' and t.peek() != 0) t.pos += 1;
            if (!t.eat('_')) return error.Unsupported;
            const p = try t.add(.{ .kind = .tparam, .text = t.src[start..t.pos] });
            try t.sub(p);
            return p;
        }
        if (c == 'S') {
            var node = try t.substitution();
            if (t.peek() == 'I') {
                node = try t.add(.{ .kind = .template, .a = node, .args = try t.templateArgs() });
                try t.sub(node);
            }
            return node;
        }
        if (c == 'N' or std.ascii.isDigit(c)) return t.name(false);
        return error.Unsupported;
    }

    fn parse(t: *Tree) Bad!void {
        if (!std.mem.startsWith(u8, t.src, "_Z")) return error.Unsupported;
        t.pos = 2;
        t.fn_name = try t.name(true);
        var tmp: [64]u16 = undefined;
        var k: usize = 0;
        while (t.pos < t.src.len) : (k += 1) {
            if (k == tmp.len) return error.Unsupported;
            tmp[k] = try t.typ();
        }
        t.params = try t.store(tmp[0..k]);
    }

    /// CUDA's names for types HIP spells differently, rewritten in place (substitutions share the node).
    fn rename(t: *Tree) Bad!void {
        const n = t.n;
        for (1..n) |i| {
            if (t.nodes[i].kind != .source) continue;
            const text = t.nodes[i].text;
            if (renamed(text)) |r| {
                t.nodes[i].text = r;
            } else if (vector(text)) |v| {
                const hvt = try t.add(.{ .kind = .source, .text = "HIP_vector_type" });
                const elem = try t.add(.{ .kind = .builtin, .text = v[0] });
                const ty = try t.add(.{ .kind = .builtin, .text = "j" });
                const lit = try t.add(.{ .kind = .literal, .a = ty, .text = v[1] });
                t.nodes[i] = .{ .kind = .template, .a = hvt, .args = try t.store(&.{ elem, lit }) };
            }
        }
    }
};

fn renamed(text: []const u8) ?[]const u8 {
    const map = [_][2][]const u8{
        .{ "__nv_bfloat16", "__hip_bfloat16" },  .{ "__nv_bfloat162", "__hip_bfloat162" },
        .{ "__nv_fp8_e4m3", "__hip_fp8_e4m3" },  .{ "__nv_fp8_e5m2", "__hip_fp8_e5m2" },
    };
    for (map) |m| if (std.mem.eql(u8, text, m[0])) return m[1];
    return null;
}

/// CUDA's vector structs (uint4, float2, ...) as HIP_vector_type's element builtin and width.
fn vector(text: []const u8) ?[2][]const u8 {
    const elems = [_][2][]const u8{
        .{ "ulonglong", "y" }, .{ "longlong", "x" }, .{ "double", "d" }, .{ "ushort", "t" }, .{ "short", "s" },
        .{ "float", "f" },     .{ "uchar", "h" },    .{ "ulong", "m" },  .{ "char", "c" },   .{ "long", "l" },
        .{ "uint", "j" },      .{ "int", "i" },
    };
    if (text.len < 2) return null;
    const w = text[text.len - 1];
    if (w < '1' or w > '4') return null;
    for (elems) |e| if (std.mem.eql(u8, text[0 .. text.len - 1], e[0])) return .{ e[1], text[text.len - 1 ..] };
    return null;
}

/// Writes nodes back with fresh substitutions; a candidate's key is its uncompressed spelling.
const Emitter = struct {
    t: *const Tree,
    out: []u8,
    len: usize = 0,
    keys: [256][]const u8 = undefined,
    nk: usize = 0,
    key_buf: [16384]u8 = undefined,
    kb: usize = 0,

    fn put(e: *Emitter, s: []const u8) Bad!void {
        if (e.len + s.len > e.out.len) return error.Unsupported;
        @memcpy(e.out[e.len..][0..s.len], s);
        e.len += s.len;
    }

    fn key(e: *Emitter, i: u16) Bad![]const u8 {
        var w: std.Io.Writer = .fixed(e.key_buf[e.kb..]);
        e.plain(&w, i) catch return error.Unsupported;
        const k = w.buffered();
        e.kb += k.len;
        return k;
    }

    fn plain(e: *Emitter, w: *std.Io.Writer, i: u16) std.Io.Writer.Error!void {
        const n = e.t.nodes[i];
        switch (n.kind) {
            .builtin, .tparam => try w.writeAll(n.text),
            .source => try w.print("{d}{s}", .{ n.text.len, n.text }),
            .nested => {
                try e.plain(w, n.a);
                try w.print("{d}{s}", .{ n.text.len, n.text });
            },
            .template => {
                try e.plain(w, n.a);
                try w.writeByte('I');
                for (e.t.list[n.args[0]..n.args[1]]) |a| try e.plain(w, a);
                try w.writeByte('E');
            },
            .qual => {
                try w.writeAll(n.text);
                try e.plain(w, n.a);
            },
            .literal => {
                try w.writeByte('L');
                try e.plain(w, n.a);
                try w.print("{s}E", .{n.text});
            },
        }
    }

    /// Emits the substitution for `i` when it is a candidate already; false otherwise.
    fn reuse(e: *Emitter, i: u16) Bad!bool {
        const k = try e.key(i);
        defer e.kb -= k.len;
        for (e.keys[0..e.nk], 0..) |old, j| if (std.mem.eql(u8, old, k)) {
            if (j == 0) {
                try e.put("S_");
                return true;
            }
            var buf: [8]u8 = undefined;
            var v = j - 1;
            var p: usize = buf.len;
            while (true) {
                p -= 1;
                buf[p] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[v % 36];
                v /= 36;
                if (v == 0) break;
            }
            try e.put("S");
            try e.put(buf[p..]);
            try e.put("_");
            return true;
        };
        return false;
    }

    fn candidate(e: *Emitter, i: u16) Bad!void {
        if (e.nk == e.keys.len) return error.Unsupported;
        e.keys[e.nk] = try e.key(i);
        e.nk += 1;
    }

    fn scoped(e: *const Emitter, i: u16) bool {
        const n = e.t.nodes[i];
        return n.kind == .nested or (n.kind == .template and e.t.nodes[n.a].kind == .nested);
    }

    /// A name's components; `last` of a function name (`top`) adds no candidate.
    fn prefix(e: *Emitter, i: u16, top: bool, last: bool) Bad!void {
        if (!(top and last) and try e.reuse(i)) return;
        const n = e.t.nodes[i];
        switch (n.kind) {
            .source => {},
            .nested => {
                try e.prefix(n.a, top, false);
            },
            .template => {
                try e.prefix(n.a, top, false);
                try e.templateArgs(n.args);
            },
            else => return error.Unsupported,
        }
        if (n.kind != .template) {
            var buf: [8]u8 = undefined;
            try e.put(std.fmt.bufPrint(&buf, "{d}", .{n.text.len}) catch unreachable);
            try e.put(n.text);
        }
        if (!(top and last)) try e.candidate(i);
    }

    fn entity(e: *Emitter, i: u16, top: bool) Bad!void {
        const sc = e.scoped(i);
        if (sc) try e.put("N");
        try e.prefix(i, top, true);
        if (sc) try e.put("E");
    }

    fn templateArgs(e: *Emitter, args: [2]u16) Bad!void {
        try e.put("I");
        for (e.t.list[args[0]..args[1]]) |a| {
            const n = e.t.nodes[a];
            if (n.kind == .literal) {
                try e.put("L");
                try e.typ(n.a);
                try e.put(n.text);
                try e.put("E");
            } else try e.typ(a);
        }
        try e.put("E");
    }

    fn typ(e: *Emitter, i: u16) Bad!void {
        const n = e.t.nodes[i];
        switch (n.kind) {
            .builtin => try e.put(n.text),
            .tparam => if (!try e.reuse(i)) {
                try e.put(n.text);
                try e.candidate(i);
            },
            .qual => if (!try e.reuse(i)) {
                try e.put(n.text);
                try e.typ(n.a);
                try e.candidate(i);
            },
            .source, .nested, .template => try e.entity(i, false),
            .literal => return error.Unsupported,
        }
    }
};

fn translate(name: []const u8, out: []u8) Bad![]const u8 {
    var t: Tree = .{ .src = name };
    try t.parse();
    try t.rename();
    var e: Emitter = .{ .t = &t, .out = out };
    try e.put("_Z");
    try e.entity(t.fn_name, true);
    for (t.list[t.params[0]..t.params[1]]) |p| try e.typ(p);
    return out[0..e.len];
}

/// The HIP symbol of a CUDA C++ mangled kernel name; extern "C" names, and any this cannot parse, come back as given.
pub fn toHip(name: []const u8, out: []u8) []const u8 {
    if (!std.mem.startsWith(u8, name, "_Z")) return name;
    return translate(name, out) catch name;
}

fn expectHip(cuda: []const u8, hip: []const u8) !void {
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings(hip, try translate(cuda, &buf));
    // a name with nothing to rename comes back byte-equal: the recompression reproduces the mangler's
    try std.testing.expectEqualStrings(hip, try translate(hip, &buf));
}

// HIP spellings: llvm-nm of hipcc 7.15 --genco --offload-arch=gfx1151 code objects (docs/rocm/notes/rt.md).
test "CUDA kernel names become hipcc's" {
    try expectHip("_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxiPKiiS5_iPfiii", "_ZN6tf_gdn13replay_kernelI14__hip_bfloat16Li8ELi4EEEvPKxiPKiiS5_iPfiii");
    try expectHip("_ZN10tf_fn_int411int4_kernelILi128ELi4ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", "_ZN10tf_fn_int411int4_kernelILi128ELi4ELi1ELi2ELi2ELi4EEEvPK14__hip_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi");
    try expectHip("_ZN10tf_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif", "_ZN10tf_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK14__hip_bfloat16iiPK15HIP_vector_typeIjLj4EEiiPKiS9_S9_Pvif");
    try expectHip("_ZN10tf_fn_roce6gatherI5uint4EEvPKT_PS2_ijS5_S4_PKjPjyPNS_5StateEiiyiPKcy", "_ZN10tf_fn_roce6gatherI15HIP_vector_typeIjLj4EEEEvPKT_PS3_ijS6_S5_PKjPjyPNS_5StateEiiyiPKcy");
    try expectHip("_ZN10tf_fn_roce6gatherIjEEvPKT_PS1_ijS4_S3_PKjPjyPNS_5StateEiiyiPKcy", "_ZN10tf_fn_roce6gatherIjEEvPKT_PS1_ijS4_S3_PKjPjyPNS_5StateEiiyiPKcy");
    try expectHip("_ZN10tf_fn_roce8prefetchEPKcy", "_ZN10tf_fn_roce8prefetchEPKcy");
    try expectHip("_ZN14tf_fn_gdn_tree11tree_kernelIfLi0ELi8ELi4ELb1EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PKxPKiSC_iPS4_iiiNS_7PendingIS1_EEPfSA_b", "_ZN14tf_fn_gdn_tree11tree_kernelIfLi0ELi8ELi4ELb1EEEvPKT_S3_PK14__hip_bfloat16PKfS8_S8_PKxPKiSC_iPS4_iiiNS_7PendingIS1_EEPfSA_b");
    try expectHip("_ZN4tf_x3vecI5uint4EEvPT_6float2S1_PKS1_PS4_7__half213__nv_bfloat1614__nv_bfloat1624int4", "_ZN4tf_x3vecI15HIP_vector_typeIjLj4EEEEvPT_S1_IfLj2EES2_PKS2_PS5_7__half214__hip_bfloat1615__hip_bfloat162S1_IiLj4EE");
}

test "extern C names and unparsable names pass through" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("tf_probe_fill", toHip("tf_probe_fill", &buf));
    try std.testing.expectEqualStrings("_ZSt4cout", toHip("_ZSt4cout", &buf));
    try std.testing.expectEqualStrings("_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxi", toHip("_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxi", buf[0..8]));
}
