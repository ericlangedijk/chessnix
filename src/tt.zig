// zig fmt: off

//! TranspositionTable for search.

// TODO: atomics when multiple threads.
// TODO: paranoid asserts for too small sizes.
// Alpha / UPPER bound → real score ≤ stored score
// Beta / LOWER bound → real score ≥ stored score
// Exact → real score == stored score (PV result)


const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf;

const Color = types.Color;
const SmallValue = types.SmallValue;
const Value = types.Value;
const Move = types.Move;
const ScorePair = types.ScorePair;

/// Simple struct for calculating the 3 tt sizes.
pub const TTSizes = struct {
    /// Space in bytes for transposition table.
    tt: usize,
    /// Space in bytes for eval table.
    eval: usize,

    // Space in bytes for pawn eval table.
    //pawneval: usize,
};

/// We need to divide the total hash space over the 3 used hash tables. Minimum is 16 MB.
pub fn compute_tt_sizes(mb: usize) TTSizes {
    const required_mb: usize = @max(16, mb);
    const size: f64 = @floatFromInt(required_mb * types.megabyte);
    var sizes: TTSizes = undefined;
    sizes.tt = @intFromFloat(size * 0.80);
    sizes.eval = @intFromFloat(size * 0.20);
    //sizes.pawneval = @intFromFloat(size * 0.05);
    return sizes;
}

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

////////////////////////////////////////////////////////////////
// Elements.
////////////////////////////////////////////////////////////////
pub const Bound = enum(u2) { None, Exact, Alpha, Beta };

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

    pub const empty: Entry = .{ .bound = .None, .key = 0, .depth = 0, .move = .empty, .score = 0 };
};

pub const EvalEntry = packed struct {
    /// The position key.
    key: u64,
    /// The evaluation according to search.
    score: i16,

    const empty: EvalEntry = .{ .key = 0, .score = 0 };
};

pub const PawnEntry = struct {
    /// The position pawn key.
    key: u64,
    /// The evaluation of both white and black according to static evaluation.
    scores: [2]ScorePair,

    const empty: PawnEntry = .{ .key = 0, .scores = @splat(.empty) };
};

/// The main transposition table.
pub const TranspositionTable = struct {
    hash: HashTable(Entry),

    pub fn init(size_in_bytes: usize) !TranspositionTable {
        return .{
            .hash = try .init(size_in_bytes)
        };
    }

    pub fn deinit(self: *TranspositionTable) void {
        self.hash.deinit();
    }

    pub fn resize(self: *TranspositionTable, size_in_bytes: usize) !void {
        try self.hash.resize(size_in_bytes);
    }

    pub fn clear(self: *TranspositionTable) void {
        self.hash.clear();
    }

    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        if (comptime lib.is_paranoid) {
             assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        }
        const entry: *Entry = self.get(key);

        // Don't overwrite
        if (entry.key == key and entry.depth > depth) {
            return;
        }

        const adjusted_score = get_adjusted_score_for_tt_store(score, ply);
        entry.* = .{ .bound = bound, .key = key, .depth = depth, .move = move, .score = @truncate(adjusted_score) };
    }

    /// The score of the entry is adjusted for the ply when mating distance is there.
    pub fn probe(self: *TranspositionTable, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        // We should return a copy with adjusted score.
        const entry: *const Entry = self.get(key);

        if (entry.key != key) {
            return null;
        }

        // Usable move.
        tt_move.* = entry.move;

        // Check score is usable for this depth.
        if (entry.depth < depth) {
            return null;
        }

        const adjusted_score = get_adjusted_score_for_tt_probe(entry.score, ply);

        switch (entry.bound) {
            .None => {
               unreachable;
            },
            .Exact => {
                return adjusted_score;
            },
            .Alpha => {
                return if (adjusted_score <= alpha) adjusted_score else null;
            },
            .Beta => {
                return if (adjusted_score >= beta) adjusted_score else null;
            },
        }
    }

    fn get(self: *TranspositionTable, key: u64) *Entry {
        return self.hash.get(key);
    }
};

/// A simple cache for evaluation speedup.
pub const EvalTranspositionTable = struct {
    /// Array on heap.
    hash: HashTable(EvalEntry),

    pub fn init(size_in_bytes: u64) !EvalTranspositionTable {
        return .{
            .hash = try .init(size_in_bytes),
        };
    }

    pub fn deinit(self: *EvalTranspositionTable) void {
        self.hash.deinit();
    }

    pub fn resize(self: *EvalTranspositionTable, size_in_bytes: usize) !void {
        try self.hash.resize(size_in_bytes);
    }

    pub fn clear(self: *EvalTranspositionTable) void {
        self.hash.clear();
    }

    pub fn store(self: *EvalTranspositionTable, key: u64, score: Value) void {
        if (comptime lib.is_paranoid) {
            assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        }
        const entry: *EvalEntry = self.get(key);
        entry.key = key;
        entry.score = @truncate(score);
    }

    pub fn probe(self: *EvalTranspositionTable, key: u64) ?Value {
        const entry: *const EvalEntry = self.get(key);
        if (entry.key != key) {
            return null;
        }
        return entry.score;
    }

    pub fn get(self: *EvalTranspositionTable, key: u64) *EvalEntry {
        return self.hash.get(key);
    }
};

/// A simple hash for pawn evaluation speedup.
pub const PawnTranspositionTable = struct {
    /// Array allocated on the heap.
    hash: HashTable(PawnEntry),

    pub fn init(size_in_bytes: u64) !PawnTranspositionTable {
        return .{
            .hash = try .init(size_in_bytes)
        };
        // const len: u64 = size_in_bytes / @sizeOf(PawnEntry);
        // const data: []PawnEntry = try ctx.galloc.alloc(PawnEntry, len);
        // @memset(data, PawnEntry.empty);
        // return .{ .data = data };
    }

    pub fn deinit(self: *PawnTranspositionTable) void {
        //ctx.galloc.free(self.data);
        self.hash.deinit();
    }

    pub fn resize(self: *PawnTranspositionTable, size_in_bytes: usize) !void {
        try self.hash.resize(size_in_bytes);
        // const len: usize = size_in_bytes / @sizeOf(PawnEntry);
        // //self.data = try ctx.galloc.resize(ctx.galloc, PawnEntry, len);
        // //try ctx.galloc.realloc(self.data, len);
        // self.data = try ctx.galloc.realloc(self.data, len);
        // self.clear();
    }

    pub fn clear(self: *PawnTranspositionTable) void {
        self.hash.clear();
        //@memset(self.data, PawnEntry.empty);
    }

    // pub fn probe(self: *EvalTranspositionTable, pawnkey: u64) ?Value {
    //     const entry: *const EvalEntry = self.get(pawnkey);
    //     if (entry.key != pawnkey) {
    //         return null;
    //     }
    //     return entry.score;
    // }

    // pub fn store(self: *EvalTranspositionTable, key: u64, score: Value) void {
    //     if (comptime lib.is_paranoid) {
    //         assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
    //     }
    //     const entry: *EvalEntry = self.get(key);
    //     entry.key = key;
    //     entry.score = @truncate(score);
    // }

    /// Get writable entry. Check key when retrieving!
    pub fn get(self: *PawnTranspositionTable, pawnkey: u64) *PawnEntry {
        return self.hash.get(pawnkey);
        // return if (e.key == pawnkey) e else null;
        //return &self.data[key % self.data.len];
    }
};

fn HashTable(Element: type) type {
    return struct {
        const Self = @This();

        const elementsize: usize = @sizeOf(Element);
        /// The `empty` must be an implemented const of the struct (Entry, EvalEntry, PawnEntry).
        //const empty_element: Element = Element.empty;

        /// Array allocated on the heap.
        data: []Element,

        fn init(size_in_bytes: usize) !Self {
            const len: usize = size_in_bytes / elementsize;
            return .{
                .data = try create_data(len),
            };
        }

        fn deinit(self: *Self) void {
            ctx.galloc.free(self.data);
        }

        pub fn percentage_filled(self: *const Self) usize {
            var filled: usize = 0;
            for (self.data) |e| {
                if (!std.meta.eql(e, Element.empty)) {
                    filled += 1;
                }
            }
            return funcs.percent(self.data.len, filled);
        }

        fn clear(self: *Self) void {
            clear_data(self.data);
        }

        fn resize(self: *Self, size_in_bytes: usize) !void {
            const len: usize = size_in_bytes / elementsize;
            self.data = try ctx.galloc.realloc(self.data, len);
            self.clear();
        }

        fn get(self: *Self, key: u64) *Element {
            return &self.data[key % self.data.len];
        }

        fn create_data(len: usize) ![]Element {
            const data: []Element = try ctx.galloc.alloc(Element, len);
            clear_data(data);
            return data;
        }

        fn clear_data(data: []Element) void {
            @memset(data, Element.empty);
        }
    };
}
