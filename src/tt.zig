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

const no_score = types.no_score;

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
    /// Upper bound.
    alpha,
    /// Lower bound.
    beta,
    /// Exact score.
    exact,
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
    /// The evaluation according to search. no_score == null.
    score: SmallValue,
    /// The raw static eval of the evaluation function. Never when in check. no_score == null.
    raw_static_eval: SmallValue,

    //age: u5,

    pub const empty: Entry = .{
        .bound = .none,
        .key = 0,
        .depth = 0,
        .move = .empty,
        .score = no_score,
        .raw_static_eval = no_score,
    };

    pub fn is_score_usable_for_depth(self: *const Entry, alpha: Value, beta: Value, depth: i32) bool {
        if (self.score == no_score or self.depth < depth) {
            return false;
        }
        return switch (self.bound) {
            .none  => false,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
            .exact => true,
        };
    }

    pub fn is_score_usable(self: *const Entry, alpha: Value, beta: Value) bool {
        if (self.score == no_score) {
            return false;
        }
        return switch (self.bound) {
            .none  => false,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
            .exact => true,
        };
    }

    pub fn is_raw_static_eval_usable(self: *const Entry) bool {
        return self.raw_static_eval != no_score;
    }

    fn bias(self: *const Entry) u2 {
        return switch(self.bound) {
            .none => 0,
            .alpha, .beta => 1,
            .exact => 2,
        };
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

    /// Only store the raw static eval.
    pub fn store_static_eval(self: *TranspositionTable, key: u64, raw_static_eval: Value) void {
        self.store(.none, key, 0, 0, .empty, no_score, raw_static_eval);
    }

    /// Store the search score and the raw static eval, if any.
    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: i32, ply: u16, move: Move, score: Value, raw_static_eval: Value) void {

        if (lib.bughunt) {
            verify_args(depth, score, raw_static_eval);
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
                // Same depth: prefer replacing a less valuable bound.
                entry = if (bucket.e0.bias() < bucket.e1.bias()) &bucket.e0 else &bucket.e1;
            }
        }

        entry.bound = bound;
        entry.key = key;
        entry.depth = @intCast(depth);
        entry.move = move;
        entry.score = if (score != no_score) @intCast(get_adjusted_score_for_tt_store(score, ply)) else no_score;
        entry.raw_static_eval = @intCast(raw_static_eval);
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

    fn verify_args(depth: i32, score: Value, raw_static_eval: Value) void {
        lib.not_in_release();
        if (depth < 0 or depth > 128) lib.crash("tt invalid depth {}", .{ depth });
        if (score > std.math.maxInt(i16) or score < std.math.minInt(i16)) lib.crash("tt invalid score {}", .{ score });
        if (raw_static_eval > std.math.maxInt(i16) or raw_static_eval < std.math.minInt(i16)) lib.crash("tt invalid raw static eval {}", .{ raw_static_eval });
    }
};

/// Simple internal generic hash, with % indexing.
/// Although we only have 1 type of hashtable (the TranspositionTable), this thing is still hanging around here from earlier days.
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
