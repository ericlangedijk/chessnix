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
};

/// We need to divide the total hash space over the 2 used hash tables. Minimum is 16 MB.
pub fn compute_tt_sizes(megabytes: usize) TTSizes {
    const required_mb: usize = @max(16, megabytes);
    const size: f64 = @floatFromInt(required_mb * types.megabyte);
    var sizes: TTSizes = undefined;
    sizes.tt = @intFromFloat(size * 0.80);
    sizes.eval = @intFromFloat(size * 0.20);
    return sizes;
}

/// Adjust score for mate in X when storing.
pub fn get_adjusted_score_for_tt_store(tt_score: Value, ply: u16) Value {
    if (tt_score >= types.mate_threshold) return tt_score + ply
    else if (tt_score <= -types.mate_threshold) return tt_score - ply;
    return tt_score;
}

/// Adjust score for mate in X when probing.
pub fn get_adjusted_score_for_tt_probe(tt_score: Value, ply: u16) Value {
    if (tt_score >= types.mate_threshold) return tt_score - ply
    else if (tt_score <= -types.mate_threshold) return tt_score + ply;
    return tt_score;
}

pub const Bound = enum(u2) { None, Exact, Alpha, Beta };

pub const Entry = packed struct {
    /// The kind (exact, alpha or beta) with which this entry was stored.
    bound: Bound,
    /// The position hash key. If key == 0 we assume this entry is empty.
    key: u64,
    /// The search depth when this entry was stored,
    depth: u8,
    /// The best move according to search.
    move: Move,
    /// The evaluation according to search.
    score: i16,
    /// Stored during principal variation search?
    was_pv: bool,

    pub const empty: Entry = .{ .bound = .None, .key = 0, .depth = 0, .move = .empty, .score = 0, .was_pv = false };

    pub fn is_score_usable(self: *const Entry, alpha: Value, beta: Value, depth: i32) bool {
        if (self.depth < depth) {
            return false;
        }
        switch (self.bound) {
            .None  => return false,
            .Exact => return true,
            .Alpha => return self.score <= alpha,
            .Beta  => return self.score >= beta,
        }
    }
};

pub const EvalEntry = packed struct {
    /// The position key.
    key: u64,
    /// The evaluation according to search.
    score: i16,

    const empty: EvalEntry = .{ .key = 0, .score = 0 };
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

    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: i32, ply: u16, move: Move, score: Value, pv: bool) void {
        if (comptime lib.is_paranoid) {
             assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
             assert(depth >= 0);
        }
        const entry: *Entry = self.get(key);

        // Don't overwrite
        if (entry.key == key and entry.depth > depth) {
            return;
        }

        const adjusted_score = get_adjusted_score_for_tt_store(score, ply);
        entry.* = .{ .bound = bound, .key = key, .depth = @intCast(depth), .move = move, .score = @intCast(adjusted_score), .was_pv = pv };
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

    /// TODO: we can just return the entry for overwrite. this saves another lookup.
    pub fn probe_raw(self: *TranspositionTable, key: u64) ?*const Entry {
        const entry: *const Entry = self.get(key);
        if (entry.key != key) {
            return null;
        }
        return entry;
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

/// Simple internal generic hash.
fn HashTable(Element: type) type {
    return struct {
        const Self = @This();

        /// The comptime Element must implement a `empty` const.
        const elementsize: usize = @sizeOf(Element);

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

        /// Slow! just for state output.
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
