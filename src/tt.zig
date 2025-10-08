// zig fmt: off

//! TranspositionTable for search.

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf;

const Value = types.Value;
const Move = types.Move;

pub const Bound = enum(u2) { None, Exact, Alpha, Beta };


// typedef struct {

//   uint64_t hash;
//   move_t move;
//   uint8_t flags;
//   uint8_t depth;
//   int16_t score;
//   int16_t ev;

// } TT;

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
    /// == len - 1.
    mask: u64,
    probes: u64,
    hits: u64,

    pub fn init(size_in_megabytes: u64) !TranspositionTable {
        comptime if (@sizeOf(Entry) != 16) @compileError("TT Entry must be 16 bytes");
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;
        const len: u64 = (size_in_megabytes * 1024 * 1024) / 16;
        const data: []Entry = try ctx.galloc.alloc(Entry, len);
        @memset(data, Entry.empty);
        return .{ .data = data, .len = len, .mb = size_in_megabytes, .mask = len - 1, .probes = 0, .hits = 0 };
    }

    pub fn deinit(self: *TranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *TranspositionTable) void {
        @memset(self.data, Entry.empty);
    }

    pub fn age(self: *TranspositionTable) void {
        self.probes = 0;
        self.hits = 0;
    }

    /// TODO: atomic store when multiple threads.
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        const entry: *Entry = self.get_mut(key);
        const adjusted_score = get_adjusted_score_for_tt_store(score, ply);
        if (entry.bound != .None and entry.key == key and entry.depth > depth) {
            return;
        }
        entry.* = .{ .bound = bound, .key = key, .depth = depth, .move = move, .score = @truncate(adjusted_score) };
    }

    // The score of the entry is adjusted for the ply when mating distance is there.
    pub fn probe(self: *TranspositionTable, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        self.probes += 1;
        const entry: Entry = self.get(key);
        if (entry.bound == .None or entry.key != key) return null;
        tt_move.* = entry.move; // usable.
        self.hits += 1;
        if (entry.depth < depth) return null;
        const adjusted_score = get_adjusted_score_for_tt_probe(entry.score, ply);
        switch (entry.bound) {
            .None => {
                unreachable;
            },
            .Exact => {
                return adjusted_score;
            },
            .Alpha => {
                if (adjusted_score <= alpha) {
                    return adjusted_score; // alpha??
                }
            },
            .Beta => {
                if (adjusted_score >= beta) {
                     return adjusted_score; // beta??
                }
            },
        }
        return null;
    }

    fn index_of(self: *const TranspositionTable, key: u64) u64 {
        // return key & (self.len - 1);
        return key & self.mask;
    }

    fn get_mut(self: *TranspositionTable, key: u64) *Entry {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    pub fn get(self: *TranspositionTable, key: u64) Entry {
        const idx = self.index_of(key);
        return self.data[idx];
    }
};


/// Must be 16 bytes.
pub const PawnEntry = packed struct {
    /// The position pawn key.
    key: u64,
    /// The evaluation according to search.
    score: i16,

    const empty: PawnEntry = .{ .key = 0, .score = 0, };
};

/// ### Deprecated for now.
/// A simple cache for pawn evaluation speedup.
pub const PawnTranspositionTable = struct {
    /// Array on heap.
    data: []PawnEntry,
    /// The number of entries.
    len: u64,
    /// Used megabytes.
    mb: u64,
    /// Mask for indexing.
    mask: u64,
    /// Filled at this age.
    filled: u64,
    /// Probes.
    probes: u64,
    /// Hits
    hits: u64,

    pub fn init(size_in_megabytes: u64) !PawnTranspositionTable {
        comptime if (@sizeOf(PawnEntry) != 16) @compileError("TT PawnEntry must be 16 bytes");
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;
        const len: u64 = (size_in_megabytes * 1024 * 1024) / 16;
        const data: []PawnEntry = try ctx.galloc.alloc(PawnEntry, len);
        @memset(data, PawnEntry.empty);
        return .{ .data = data, .len = len, .mb = size_in_megabytes, .mask = len - 1, .filled = 0, .probes = 0, .hits = 0, };
    }

    pub fn deinit(self: *PawnTranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *PawnTranspositionTable) void {
        @memset(self.data, PawnEntry.empty);
        self.filled = 0;
        self.probes = 0;
        self.hits = 0;
    }

    /// TODO: atomic store when multiple threads.
    pub fn store(self: *PawnTranspositionTable, key: u64, score: Value) void {
        assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        const entry: *PawnEntry = self.get_mut(key);
        const was_empty: bool = entry.key == 0;
        entry.* = .{
            .key = key,
            .score = @truncate(score),
        };
        if (was_empty) self.filled += 1;
    }

    pub fn probe(self: *PawnTranspositionTable, key: u64) ?Value {
        self.probes += 1;
        const entry: PawnEntry = self.get(key);
        if (entry.key != key) return null;
        self.hits += 1;
        return entry.score;
    }

    fn index_of(self: *const PawnTranspositionTable, key: u64) u64 {
        // return key & (self.len - 1);
        return key & self.mask;
    }

    fn get_mut(self: *PawnTranspositionTable, key: u64) *PawnEntry {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    pub fn get(self: *PawnTranspositionTable, key: u64) PawnEntry {
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


/// Must be 16 bytes.
pub const EvalEntry = packed struct {
    /// The position pawn key.
    key: u64,
    /// The evaluation according to search.
    score: i16,
    const empty: EvalEntry = .{ .key = 0, .score = 0, };
};

/// ### Deprecated for now.
/// A simple cache for pawn evaluation speedup.
pub const EvalTranspositionTable = struct {
    /// Array on heap.
    data: []EvalEntry,
    /// The number of entries.
    len: u64,
    /// Used megabytes.
    mb: u64,
    /// Mask for indexing.
    mask: u64,
    /// Filled at this age.
    filled: u64,
    /// Probes.
    probes: u64,
    /// Hits
    hits: u64,

    pub fn init(size_in_megabytes: u64) !EvalTranspositionTable {
        comptime if (@sizeOf(EvalEntry) != 16) @compileError("TT EvalEntry must be 16 bytes");
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;
        const len: u64 = (size_in_megabytes * 1024 * 1024) / 16;
        const data: []EvalEntry = try ctx.galloc.alloc(EvalEntry, len);
        @memset(data, EvalEntry.empty);
        return .{ .data = data, .len = len, .mb = size_in_megabytes, .mask = len - 1, .filled = 0, .probes = 0, .hits = 0, };
    }

    pub fn deinit(self: *EvalTranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *EvalTranspositionTable) void {
        @memset(self.data, EvalEntry.empty);
        self.filled = 0;
        self.probes = 0;
        self.hits = 0;
    }

    /// TODO: atomic store when multiple threads.
    pub fn store(self: *EvalTranspositionTable, key: u64, score: Value) void {
        assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        const entry: *EvalEntry = self.get_mut(key);
        const was_empty: bool = entry.key == 0;
        entry.* = .{
            .key = key,
            .score = @truncate(score),
        };
        if (was_empty) self.filled += 1;
    }

    pub fn probe(self: *EvalTranspositionTable, key: u64) ?Value {
        self.probes += 1;
        const entry: EvalEntry = self.get(key);
        if (entry.key != key) return null;
        self.hits += 1;
        return entry.score;
    }

    fn index_of(self: *const EvalTranspositionTable, key: u64) u64 {
        // return key & (self.len - 1);
        return key & self.mask;
    }

    fn get_mut(self: *EvalTranspositionTable, key: u64) *EvalEntry {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    pub fn get(self: *EvalTranspositionTable, key: u64) EvalEntry {
        const idx = self.index_of(key);
        return self.data[idx];
    }
};






const Error = error {
    TTSizeMustBeAPowerOfTwo,
};











test "transpositiontable" {
    const position = @import("position.zig");
    try lib.initialize();

    {
        var pt: PawnTranspositionTable = try .init(2);
        defer pt.deinit();
        var pos: position.Position = .empty;
        var st: position.StateInfo = undefined;
        try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        pt.store(st.pawnkey, 42);
        const e = pt.probe(pos.state.pawnkey);
        try std.testing.expect(e != null);
        try std.testing.expect(e.? == 42);
    }

    // {
    //     const megabyte: usize = 1024 * 1024;

    //     var tt: TranspositionTable = try .init(8);
    //     defer tt.deinit();

    //     try std.testing.expectEqual((8 * megabyte) / 16, tt.len);

    //     var pos: position.Position = .empty;
    //     var st: position.StateInfo = undefined;
    //     try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    //     const move: types.Move = .create(types.Square.E2, types.Square.A6);
    //     var e: ?Entry = null;

    //     // Probing must succeed.
    //     const eq_entry: Entry = .{ .bound = .Exact, .key = pos.state.key, .depth = 1, .move = move, .score = 42, .age = 0 };
    //     tt.store(.Exact, pos.state.key, 1, move, 42);
    //     e = tt.probe(pos.state.key);
    //     try std.testing.expect(e != null);
    //     try std.testing.expect(eq_entry == e.?);

    //     // Probing must fail.
    //     var newstate: position.StateInfo = undefined;
    //     pos.do_move(types.Color.WHITE, &newstate, move);
    //     e = tt.probe(pos.state.key);
    //     try std.testing.expect(e == null);

    //     // We should have 1 entry.
    //     //try std.testing.expectEqual(1, tt.filled);

    //     // We should have 2 entries.
    //     tt.store(.Lower, pos.state.key, 1, move, 144);
    //     //try std.testing.expectEqual(2, tt.filled);

    //     // Probing must succeed
    //     e = tt.probe(pos.state.key);
    //     try std.testing.expect(e != null);
    //     try std.testing.expectEqual(e.?.score, 144);
    // }
}


