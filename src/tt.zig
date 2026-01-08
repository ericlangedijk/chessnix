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

/// Returns number of bytes. Minimum is 16 MB.
pub fn compute_tt_size(megabytes: usize) usize {
    const required_mb: usize = @max(16, megabytes);
    return required_mb * types.megabyte;
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

pub const Bound = enum(u2) {
    /// Empty or static eval only.
    none,
    /// Exact score.
    exact,
    /// Upper bound.
    alpha,
    /// Lower bound.
    beta
};

/// 16 bytes, 123 bits. We could still stuff a 5 bits age in here.
pub const Entry = packed struct {
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
    /// Indicates wether this entry stored during principal variation search.
    was_pv: bool,
    /// The raw static eval of the evaluation function. Never when in check.
    raw_static_eval: SmallValue,

    pub const empty: Entry = .{
        .bound = .none,
        .key = 0,
        .depth = 0,
        .move = .empty,
        .score = -types.infinity,
        .raw_static_eval = -types.infinity,
        .was_pv = false
    };

    pub fn is_score_usable_for_depth(self: *const Entry, alpha: Value, beta: Value, depth: i32) bool {
        if (self.score == -types.infinity or self.depth < depth) {
            return false;
        }
        return switch (self.bound) {
            .none  => false,
            .exact => true,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
        };
    }

    pub fn is_score_usable(self: *const Entry, alpha: Value, beta: Value) bool {
        if (self.score == -types.infinity) {
            return false;
        }
        return switch (self.bound) {
            .none  => false,
            .exact => true,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
        };
    }

    pub fn is_raw_static_eval_usable(self: *const Entry) bool {
        return self.raw_static_eval != -types.infinity;
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
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: i32, ply: u16, move: Move, score: Value, pv: bool, raw_static_eval: ?Value) void {
        if (comptime lib.is_paranoid) {
            assert(depth >= 0 and depth <= 128);
            assert(score < std.math.maxInt(i16) and score > std.math.minInt(i16));
            assert(raw_static_eval == null or (raw_static_eval.? < std.math.maxInt(i16) and raw_static_eval.? > std.math.minInt(i16)));
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
                // Same depth: prefer replacing a non-exact bound
                const v0_cost: u1 = if (bucket.e0.bound == .exact) 1 else 0;
                const v1_cost: u1 = if (bucket.e1.bound == .exact) 1 else 0;
                entry = if (v0_cost < v1_cost) &bucket.e0 else &bucket.e1;
                // TODO: if this equal use age.
            }
        }

        entry.bound = bound;
        entry.key = key;
        entry.depth = @intCast(depth);
        entry.move = move;
        entry.score = @intCast(get_adjusted_score_for_tt_store(score, ply));
        entry.was_pv = pv;
        entry.raw_static_eval = if (raw_static_eval != null) @intCast(raw_static_eval.?) else -types.infinity;
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

        /// Slow! just for state debug output.
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
