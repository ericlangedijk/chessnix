// zig fmt: off

//! Transposition Table for search.

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

/// 16 bytes, 123 bits. We could still stuff a 5 bits age in here.
pub const Entry = struct {
    /// The kind (exact, alpha or beta) with which this entry was stored.
    bound: Bound,
    /// The position hash key.
    key: u64,
    /// The search depth when this entry was stored,
    depth: u8,
    /// The best move according to search.
    move: Move,
    /// The evaluation according to search.
    score: SmallValue,
    /// Stored during principal variation search?
    was_pv: bool,

    pub const empty: Entry = .{
        .bound = .None,
        .key = 0,
        .depth = 0,
        .move = .empty,
        .score = -types.infinity,
        // .raw_static_eval = -types.infinity,
        .was_pv = false
    };

    pub fn is_score_usable_for_depth(self: *const Entry, alpha: Value, beta: Value, depth: i32) bool {
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

    pub fn is_score_usable(self: *const Entry, alpha: Value, beta: Value) bool {
        switch (self.bound) {
            .None  => return false,
            .Exact => return true,
            .Alpha => return self.score <= alpha,
            .Beta  => return self.score >= beta,
        }
    }
};

pub const Bucket = struct {
    e0: Entry,
    e1: Entry,

    pub const empty: Bucket = .{ .e0 = .empty, .e1 = .empty };
};

/// The main transposition table.
pub const TranspositionTable = struct {
    hash: HashTable(Bucket),

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

    /// Store the search score. Does not overwrite the static_eval.
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: i32, ply: u16, move: Move, score: Value, pv: bool) void {
        if (comptime lib.is_paranoid) {
            assert(depth >= 0 and depth <= 128);
            assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
        }

        const bucket: *Bucket = self.hash.get(key);
        var entry: *Entry = undefined;

        if (bucket.e0.key == key) {
            entry = &bucket.e0;
        }
        else if (bucket.e1.key == key) {
            entry = &bucket.e1;
        }
        // No match, choose entry.
        else {
            if (bucket.e0.depth < bucket.e1.depth) {
                entry = &bucket.e0;
            }
            else if (bucket.e1.depth < bucket.e0.depth) {
                entry = &bucket.e1;
            }
            else {
                // Same depth → prefer replacing a non-exact bound
                const v0_cost: u1 = if (bucket.e0.bound == .Exact) 1 else 0;
                const v1_cost: u1 = if (bucket.e1.bound == .Exact) 1 else 0;
                entry = if (v0_cost < v1_cost) &bucket.e0 else &bucket.e1;
            }
        }

        entry.bound = bound;
        entry.key = key;
        entry.depth = @intCast(depth);
        entry.move = move;
        entry.score = @intCast(get_adjusted_score_for_tt_store(score, ply));
        entry.was_pv = pv;
    }

    pub fn probe(self: *TranspositionTable, key: u64) ?*const Entry {
        const bucket: *const Bucket = self.hash.get(key);
        if (bucket.e0.key == key) {
            return &bucket.e0;
        }
        if (bucket.e1.key == key) {
            return &bucket.e1;
        }
        return null;
    }
};

pub const EvalEntry = packed struct {
    /// The position key.
    key: u64,
    /// The evaluation according to search.
    score: i16,

    const empty: EvalEntry = .{ .key = 0, .score = 0 };
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
