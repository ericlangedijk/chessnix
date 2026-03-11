// zig fmt: off

//! Transposition Table for search.

// Attempts to use three entries per bucket (aligning a bucket to 32 bytes) failed miserably in playing strength.
// The indexing of a bucket is a simple % length, thus the stored u16 entry key uses the 16 highest bits.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const consts = @import("consts.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf;

const Color = types.Color;
const SmallScore = types.SmallScore;
const Score = types.Score;
const Move = types.Move;
const ScorePair = types.ScorePair;

const no_score = types.no_score;
const tunables = consts.tunables;

// Using two entries per bucket we need to assert these sizes.
comptime {
    assert(Entry.STRUCTSIZE == 10);
    assert(Bucket.STRUCTSIZE == 20);
}

/// Returns number of bytes. Minimum is 16 MB.
pub fn compute_tt_size(megabytes: usize) usize {
    const required_mb: usize = @max(16, megabytes);
    return required_mb * types.megabyte;
}

/// Adjust score for mate in X when storing.
pub fn score_to_tt(tt_score: Score, ply: u16) Score {
    if (tt_score >= types.mate_threshold) return tt_score + ply
    else if (tt_score <= -types.mate_threshold) return tt_score - ply;
    return tt_score;
}

/// Adjust score for mate in X when probing.
pub fn score_from_tt(tt_score: Score, ply: u16) Score {
    if (tt_score >= types.mate_threshold) return tt_score - ply
    else if (tt_score <= -types.mate_threshold) return tt_score + ply;
    return tt_score;
}

pub const Bound = enum(u2) {
    /// No bound. Empty or raw static eval only.
    none,
    /// Upper bound.
    alpha,
    /// Lower bound.
    beta,
    /// Exact score.
    exact,
};

pub const Entry = struct {
    /// Must be 10 bytes
    pub const STRUCTSIZE: usize = @sizeOf(Entry);

    /// Making the Entry as small as possible.
    pub const Flags = packed struct {
        /// The bound kind with which this entry was stored.
        bound: Bound,
        /// Aging used for replacement strategy.
        age: u5,
    };

    /// The compressed position hash key (the 16 highest bits are used, because the bucket indexing uses %).
    key: u16,
    /// The evaluation according to search. no_score == null.
    score: SmallScore,
    /// The raw static eval of the evaluation function. Never when in check. no_score == null.
    raw_static_eval: SmallScore,
    /// The best move according to search.
    move: Move,
    /// The search depth when this entry was stored,
    depth: u8,
    /// Compressing the Entry to 10 bytes.
    flags: Flags,

    pub const empty: Entry = .{
        .key = 0,
        .score = no_score,
        .raw_static_eval = no_score,
        .move = .empty,
        .depth = 0,
        .flags = .{ .bound = .none, .age = 0 },
    };

    pub fn is_score_usable_for_depth(self: *const Entry, alpha: Score, beta: Score, depth: i32) bool {
        if (self.score == no_score or self.depth < depth) {
            return false;
        }
        return switch (self.flags.bound) {
            .none  => false,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
            .exact => true,
        };
    }

    pub fn is_score_usable(self: *const Entry, alpha: Score, beta: Score) bool {
        if (self.score == no_score) {
            return false;
        }
        return switch (self.flags.bound) {
            .none  => false,
            .alpha => self.score <= alpha,
            .beta  => self.score >= beta,
            .exact => true,
        };
    }

    pub fn is_raw_static_eval_usable(self: *const Entry) bool {
        return self.raw_static_eval != no_score;
    }

    pub fn is_empty(self: Entry) bool {
        return self.key == 0;
    }

    /// Score for TT entry replacement. Lower is less valuable.
    fn get_replacement_value(self: *const Entry, current_age: i32) i32 {
        const age_penalty: i32 = (current_age - self.flags.age) & 31;
        const score: i32 = tunables.tt_entry_depth_weight * self.depth - tunables.tt_entry_age_weight * age_penalty;
        return score;
    }

    inline fn compress_key(key: u64) u16 {
        return @intCast(key >> 48);
    }
};

pub const Bucket = struct {
    /// Must be 20 bytes
    pub const STRUCTSIZE: usize = @sizeOf(Bucket);

    entries: [2]Entry,

    pub const empty: Bucket = .{ .entries = @splat(.empty) };
};

/// The main transposition table.
pub const TranspositionTable = struct {
    hash: HashTable,
    age: u5,

    pub fn init(size_in_bytes: usize) !TranspositionTable {
        return .{
            .hash = try .init(size_in_bytes),
            .age = 0,
        };
    }

    pub fn deinit(self: *TranspositionTable) void {
        self.hash.deinit();
    }

    pub fn resize(self: *TranspositionTable, size_in_bytes: usize) !void {
        try self.hash.resize(size_in_bytes);
        self.age = 0;
    }

    pub fn clear(self: *TranspositionTable) void {
        self.hash.clear();
        self.age = 0;
    }

    pub fn decay(self: *TranspositionTable) void {
        self.age +%= 1;
    }

    /// Only store the raw static eval.
    pub fn store_raw_static_eval(self: *TranspositionTable, key: u64, raw_static_eval: Score) void {
        self.store(0, key, no_score, raw_static_eval, Move.empty, 0, Bound.none);
    }

    /// Store the search score and the raw static eval.
    pub fn store(self: *TranspositionTable, ply: u16, key: u64, score: Score, raw_static_eval: Score, move: Move, depth: i32, bound: Bound) void {
        if (comptime lib.bughunt) {
            verify_args(depth, score, raw_static_eval);
        }

        const ck: u16 = Entry.compress_key(key);

        const entry: *Entry = blk: {
            const bucket: *Bucket = self.hash.get(key);
            var best_value: i32 = std.math.maxInt(i32);
            var best_entry: *Entry = &bucket.entries[0];

            inline for (&bucket.entries) |*e| {
                if (e.key == ck) {
                    break :blk e;
                }
                const value = e.get_replacement_value(self.age);
                if (value < best_value) {
                    best_value = value;
                    best_entry = e;
                }
            }

            break :blk best_entry;
        };

        const overwrite: bool = entry.key != ck or bound == .exact or depth + 4 > entry.depth or entry.flags.age != self.age;

        if (!overwrite) {
            entry.flags.age = self.age;
            if (!move.is_empty()) {
                entry.move = move;
            }
            if (entry.raw_static_eval == no_score and raw_static_eval != no_score) {
                entry.raw_static_eval = @intCast(raw_static_eval);
            }
            return;
        }

        entry.key = ck;
        entry.score = if (score != no_score) @intCast(score_to_tt(score, ply)) else no_score;
        entry.raw_static_eval = if (raw_static_eval != no_score) @intCast(raw_static_eval) else no_score;
        entry.move = move;
        entry.depth = @intCast(depth);
        entry.flags.bound = bound;
        entry.flags.age = if (bound != .none) self.age else 0;
    }

    pub fn probe(self: *TranspositionTable, key: u64) Entry {
        const bucket: *const Bucket = self.hash.get(key);
        const ck: u16 = Entry.compress_key(key);
        inline for (bucket.entries) |e| {
            if (e.key == ck) {
                return e;
            }
        }
        return .empty;
    }

    fn verify_args(depth: i32, score: Score, raw_static_eval: Score) void {
        lib.not_in_release();
        if (depth < 0 or depth > types.max_search_depth + 16) lib.wtf("tt invalid depth", .{});
        if (score > std.math.maxInt(i16) or score < std.math.minInt(i16)) lib.wtf("tt invalid score", .{});
        if (raw_static_eval > std.math.maxInt(i16) or raw_static_eval < std.math.minInt(i16)) lib.wtf("tt invalid raw static eval", .{});
    }
};

/// Simple hash to handle the details for allocation and indexing.
const HashTable = struct {
    const elementsize: usize = @sizeOf(Bucket);

    /// data is allocated on the heap.
    data: []Bucket,

    fn init(size_in_bytes: usize) !HashTable {
        const len: usize = size_in_bytes / elementsize;
        return .{
            .data = try create_data(len),
        };
    }

    fn deinit(self: *HashTable) void {
        ctx.galloc.free(self.data);
    }

    fn clear(self: *HashTable) void {
        clear_data(self.data);
    }

    fn resize(self: *HashTable, size_in_bytes: usize) !void {
        const len: usize = size_in_bytes / elementsize;
        self.data = try ctx.galloc.realloc(self.data, len);
        self.clear();
    }

    inline fn index_of(self: *const HashTable, key: u64) usize {
        return key % self.data.len;
    }

    /// Public for prefetch in search.
    pub fn get(self: *HashTable, key: u64) *Bucket {
        const idx: usize = self.index_of(key);
        return &self.data[idx];
    }

    fn create_data(len: usize) ![]Bucket {
        const data: []Bucket = try ctx.galloc.alloc(Bucket, len);
        clear_data(data);
        return data;
    }

    fn clear_data(data: []Bucket) void {
        @memset(data, Bucket.empty);
    }
};
