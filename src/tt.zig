// zig fmt: off

//! TranspositionTable for search.

// TODO: atomic store when multiple threads.
// TODO: paranoid asserts for too small sizes.

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
    /// Space in bytes for pawn eval table.
    pawneval: usize,
};

/// We need to divide the total hash space over the 3 used hash tables. Minimum is 16 MB.
pub fn compute_tt_sizes(mb: usize) TTSizes {
    const required_mb: usize = @max(16, mb);
    const size: f64 = @floatFromInt(required_mb * types.megabyte);
    var sizes: TTSizes = undefined;
    sizes.tt = @intFromFloat(size * 0.75);
    sizes.eval = @intFromFloat(size * 0.20);
    sizes.pawneval = @intFromFloat(size * 0.05);
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

    const empty: Entry = .{ .bound = .None, .key = 0, .depth = 0, .move = .empty, .score = 0 };
};

pub const TranspositionTable = struct {
    /// Array allocated on the heap.
    data: []Entry,

    pub fn init(size_in_bytes: usize) !TranspositionTable {
        const len: usize = size_in_bytes / @sizeOf(Entry);
        const data: []Entry = try ctx.galloc.alloc(Entry, len);
        @memset(data, Entry.empty);
        return .{ .data = data };
    }

    pub fn deinit(self: *TranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn resize(self: *TranspositionTable, size_in_bytes: usize) !void {
        const len: usize = size_in_bytes / @sizeOf(Entry);
        self.data = try ctx.galloc.realloc(self.data, len);
        self.clear();
    }

    pub fn clear(self: *TranspositionTable) void {
        @memset(self.data, Entry.empty);
    }


// Alpha / UPPER bound → real score ≤ stored score
// Beta / LOWER bound → real score ≥ stored score
// Exact → real score == stored score (PV result)
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        if (comptime lib.is_paranoid) {
             assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        }
        const entry: *Entry = self.get(key);

        // // Don't overwrite
        // if (entry.bound != .None and entry.key == key and entry.depth > depth) {
        //     return;
        // }

        if (entry.key == key and entry.depth > depth) {
            return;
        }


        const adjusted_score = get_adjusted_score_for_tt_store(score, ply);
        entry.* = .{ .bound = bound, .key = key, .depth = depth, .move = move, .score = @truncate(adjusted_score) };
    }

    /// The score of the entry is adjusted for the ply when mating distance is there.
    pub fn probe(self: *TranspositionTable, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        // We should return a copy with adjusted score.
        const entry: Entry = self.get(key).*;

        // This is nothing or a collision.
        if (entry.bound == .None or entry.key != key) {
            return null;
        }

        // Usable move.
        tt_move.* = entry.move;

        // Check score is usable at this depth.
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

    fn index_of(self: *const TranspositionTable, key: u64) u64 {
        return key % self.data.len;
    }

    fn get(self: *TranspositionTable, key: u64) *Entry {
        const idx = self.index_of(key);
        return &self.data[idx];
    }
};

pub const PawnEntry = struct {
    /// The position pawn key.
    key: u64,
    /// The evaluation of both white and black according to static evaluation.
    scores: [2]ScorePair,

    const empty: PawnEntry = .{ .key = 0, .scores = @splat(.empty) };
};

/// A simple hash for pawn evaluation speedup.
pub const PawnTranspositionTable = struct {
    /// Array allocated on the heap.
    data: []PawnEntry,

    pub fn init(size_in_bytes: u64) !PawnTranspositionTable {
        const len: u64 = size_in_bytes / @sizeOf(PawnEntry);
        const data: []PawnEntry = try ctx.galloc.alloc(PawnEntry, len);
        @memset(data, PawnEntry.empty);
        return .{ .data = data };
    }

    pub fn deinit(self: *PawnTranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn resize(self: *PawnTranspositionTable, size_in_bytes: usize) !void {
        const len: usize = size_in_bytes / @sizeOf(PawnEntry);
        //self.data = try ctx.galloc.resize(ctx.galloc, PawnEntry, len);
        //try ctx.galloc.realloc(self.data, len);
        self.data = try ctx.galloc.realloc(self.data, len);
        self.clear();
    }

    pub fn clear(self: *PawnTranspositionTable) void {
        @memset(self.data, PawnEntry.empty);
    }

    /// Get writable entry.
    pub fn get(self: *PawnTranspositionTable, key: u64) *PawnEntry {
        return &self.data[key % self.data.len];
    }
};

pub const EvalEntry = packed struct {
    /// The position pawn key.
    key: u64,
    /// The evaluation according to search.
    score: i16,

    const empty: EvalEntry = .{ .key = 0, .score = 0, };
};

/// A simple cache for evaluations.
pub const EvalTranspositionTable = struct {
    /// Array on heap.
    data: []EvalEntry,

    pub fn init(size_in_bytes: u64) !EvalTranspositionTable {
        const len: usize = size_in_bytes / @sizeOf(EvalEntry);
        const data: []EvalEntry = try ctx.galloc.alloc(EvalEntry, len);
        @memset(data, EvalEntry.empty);
        return .{ .data = data };
    }

    pub fn deinit(self: *EvalTranspositionTable) void {
        ctx.galloc.free(self.data);
    }

    pub fn resize(self: *EvalTranspositionTable, size_in_bytes: usize) !void {
        const len: usize = size_in_bytes / @sizeOf(EvalEntry);
        self.data = try ctx.galloc.realloc(self.data, len);
        self.clear();
    }

    pub fn clear(self: *EvalTranspositionTable) void {
        @memset(self.data, EvalEntry.empty);
    }

    pub fn store(self: *EvalTranspositionTable, key: u64, score: Value) void {
        if (comptime lib.is_paranoid) {
            assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        }
        const entry: *EvalEntry = &self.data[key % self.data.len];
        entry.* = .{
            .key = key,
            .score = @truncate(score),
        };
    }

    pub fn probe(self: *EvalTranspositionTable, key: u64) ?Value {
        const entry: *const EvalEntry = &self.data[key % self.data.len];
        if (entry.key != key) {
            return null;
        }
        return entry.score;
    }
};

/// Generic hashtable logic here.
fn HashTable(Element: type) type {
    return struct {
        const Self = @This();
        const elementsize: usize = @sizeOf(Element);
        //const empty_element: Element = std.mem.zeroes(Element);

        data: []Element,

        fn init(size_in_bytes: usize) !Self {
            const len: usize = size_in_bytes / elementsize;
            return .{
                .data = ctx.galloc(Element, len),
            };
        }

        pub fn deinit(self: *Self) void {
            ctx.galloc.free(self.data);
        }

        fn create_data(len: usize) ![]Element {
            const data = ctx.galloc(Element, len);
            clear_data(data);
            return data;
        }

        fn clear_data(data: []Element) void {
            const ptr: []u8 = @bitCast(data);
            @memset(ptr, 0);
        }

        fn resize(self: *Self, size_in_bytes: usize) !void {
            const len: usize = size_in_bytes / elementsize;
            self.data = try ctx.galloc.realloc(self.data, len);
            self.clear();
        }

        fn get(self: *Self, key: u64) *Element {
            return *self.data[key % self.data.len];
        }
    };
}
