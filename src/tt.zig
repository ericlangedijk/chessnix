// zig fmt: off

//! TranspositionTable for search.

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf();

const Value = types.Value;
const Move = types.Move;

pub const Bound = enum(u2) { None, Exact, Alpha, Beta };

/// Must be 16 bytes.
pub const Entry = packed struct {
    /// The kind (exact, alpha or beta) with which this entry was stored..
    bound: Bound,
    /// The position hash key. If key == 0 we assume this entry is empty.
    key: u64,
    /// The search depth when this entry was stored,
    depth: u8,
    /// The best move according to search.
    move: Move,
    /// The evaluation according to search.
    score: i16,
    // The age
    // age: u8,

    const empty: Entry = .{ .bound = .None, .key = 0, .depth = 0, .move = .empty, .score = 0 };
};

// 64 MB = ok, 256 MB = very good.
pub const TranspositionTable = struct {
    /// Array on heap.
    data: []Entry,
    /// The number of entries.
    len: u64,
    /// Used megabytes.
    mb: u64,
    /// Number of filled entries.
    filled: u64,
    /// Probes.
    probes: u64,
    /// Hits
    hits: u64,

    // age: u8,

    pub fn init(size_in_megabytes: u64) !TranspositionTable {
        comptime if (@sizeOf(Entry) != 16) @compileError("TT Entry must be 16 bytes");
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;
        const len: u64 = (size_in_megabytes * 1024 * 1024) / 16;
        const data: []Entry = try ctx.galloc.alloc(Entry, len);
        @memset(data, Entry.empty);
        return .{ .data = data, .len = len, .mb = size_in_megabytes, .filled = 0, .probes = 0, .hits = 0, };
    }

    pub fn deinit(self: *TranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *TranspositionTable) void {
        @memset(self.data, Entry.empty);
        self.filled = 0;
        self.probes = 0;
        self.hits = 0;
        // self.uci_offsetkey = 0;
        // self.age = 0;
    }

    // pub fn increase_age(self: *TranspositionTable) void {
    //     self.age +%= 1;
    // }

    pub fn permille(self: *const TranspositionTable) usize {
        return funcs.permille(self.len, self.filled);
    }

    /// TODO: atomic store when threading.
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        const entry: *Entry = self.get_mut(key);
        const was_empty: bool =  entry.bound == .None;
        const adjusted_score = get_adjusted_score_for_tt_store(score, ply);
        if (entry.key != key or entry.bound == .None or depth >= entry.depth) {
            entry.* = .{
                .bound = bound,
                .key = key,
                .depth = depth,
                .move = move,
                .score = @truncate(adjusted_score),
            };
        }
        if (was_empty) self.filled += 1;
    }

    // The score of the entry is adjusted for the ply when mating distance is there.
    pub fn probe(self: *TranspositionTable, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        self.probes += 1;
        const entry: Entry = self.get(key);
        if (entry.bound == .None or entry.key != key) return null;
        tt_move.* = entry.move;
        if (entry.depth < depth) return null;
        const adjusted_score = get_adjusted_score_for_tt_probe(entry.score, ply);
        switch (entry.bound) {
            .None => {
                unreachable;
            },
            .Exact => {
                self.hits += 1;
                return adjusted_score;
            },
            .Alpha => {
                if (adjusted_score <= alpha) {
                    self.hits += 1;
                    return adjusted_score;
                }
            },
            .Beta => {
                if (adjusted_score >= beta) {
                    self.hits += 1;
                     return adjusted_score;
                }
            },
        }
        return null;
    }

    /// Just return the entry if not empty and key matches. Ignoring age.
    pub fn probe_original(self: *TranspositionTable, key: u64) ?Entry {
        const entry: Entry = self.get(key);
        if (entry.bound == .None or entry.key != key) return null;
        return entry;
    }

    fn index_of(self: *const TranspositionTable, key: u64) u64 {
        return key & (self.len - 1);
    }

    fn get_mut(self: *TranspositionTable, key: u64) *Entry {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    fn get(self: *TranspositionTable, key: u64) Entry {
        const idx = self.index_of(key);
        return self.data[idx];
    }
};

/// Adjust score for mate in X when storing.
pub fn get_adjusted_score_for_tt_store(score: Value, ply: u16) Value {
    if (score >= types.mate_threshold) return score + ply
    else if (score <= -types.mate_threshold) return score - ply;
    return score;
}

/// Adjust score for mate in X when probing.
pub fn get_adjusted_score_for_tt_probe(score: Value, ply: u16) Value {
    if (score >= types.mate_threshold) return score - ply
    else if (score <= -types.mate_threshold) return score + ply;
    return score;
}

const Error = error {
    TTSizeMustBeAPowerOfTwo,
};

test "transpositiontable" {
    const position = @import("position.zig");
    try lib.initialize();

    const megabyte: usize = 1024 * 1024;

    var tt: TranspositionTable = try .init(8);
    defer tt.deinit();

    try std.testing.expectEqual((8 * megabyte) / 16, tt.len);

    var pos: position.Position = .empty;
    var st: position.StateInfo = undefined;
    try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const move: types.Move = .create(types.Square.E2, types.Square.A6);
    var e: ?Entry = null;

    // Probing must succeed.
    const eq_entry: Entry = .{ .bound = .Exact, .key = pos.state.key, .depth = 1, .move = move, .score = 42, .age = 0 };
    tt.store(.Exact, pos.state.key, 1, move, 42);
    e = tt.probe(pos.state.key);
    try std.testing.expect(e != null);
    try std.testing.expect(eq_entry == e.?);

    // Probing must fail.
    var newstate: position.StateInfo = undefined;
    pos.do_move(types.Color.WHITE, &newstate, move);
    e = tt.probe(pos.state.key);
    try std.testing.expect(e == null);

    // We should have 1 entry.
    try std.testing.expectEqual(1, tt.filled);

    // We should have 2 entries.
    tt.store(.Lower, pos.state.key, 1, move, 144);
    try std.testing.expectEqual(2, tt.filled);

    // Probing must succeed
    e = tt.probe(pos.state.key);
    try std.testing.expect(e != null);
    try std.testing.expectEqual(e.?.score, 144);
}
