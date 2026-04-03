// zig fmt: off

//! An attempt to centralize scoring logic, which is quite tricky.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");

const assert = std.debug.assert;

const max_search_depth = types.max_search_depth;

/// A score which means 'nothing' and should be treated as such.
pub const null_score: i32 = -32002;
/// Used for alpha beta.
pub const infinity: i32 = 32000;
/// Mate in 0.
pub const mate: i32 = 30000;
/// Mate in 128.
pub const mate_threshold = mate - max_search_depth;
/// Absolute win.
pub const win = 27000;
/// Static eval should never exceed this.
pub const static_eval_threshold = 20000;
pub const draw: i32 = 0;
pub const stalemate: i32 = 0;

/// Generates a 'random' draw score between -1 and 1.
/// - Only used in search.
pub fn drawscore(seed: u64) i32 {
    const r: i32 = @intCast(seed % 3);
    return 1 - r;
}

pub fn is_drawscore(score: i32) bool {
    return score >= -1 and score <= 1;
}

pub fn is_nullscore(score: i32) bool {
    return score == null_score;
}

/// TODO: maybe use 27000 (win) as threshold.
pub fn is_matescore(score: i32) bool {
    if (lib.is_paranoid) {
        assert(score > -infinity and score < infinity);
    }
    return @abs(score) >= mate_threshold;
}

pub fn is_normalscore(score: i32) bool {
    return @abs(score) <= static_eval_threshold;
}

/// Adjust score of transposition table when storing. Mate in X scores are adjusted using ply.
pub fn score_to_tt(score: i32, ply: u16) i32 {
    // assert valid
    if (score <= -win) {
        return score - ply;
    }
    if (score >= win) {
        return score + ply;
    }
    return score;
}

/// Adjust score of transposition table when probing. Mate in X scores are adjusted using ply.
pub fn score_from_tt(tt_score: i32, ply: u16) i32 {
    // assert valid
    if (tt_score <= -win) {
        return tt_score + ply;
    }
    if (tt_score >= win) {
        return tt_score - ply;
    }
    return tt_score;
}

/// Outputs  "cp n" for a normal score and "mate n" or "-mate n" for matescores. Random drawscores become 0.
pub fn format_score(score: i32) utils.BoundedArray(u8, 16) {
    var result: utils.BoundedArray(u8, 16) = .empty;

    if (is_matescore(score)) {
        const plies_to_mate = if (score > 0) mate - score else mate + score;
        const whole_moves_to_mate = @divTrunc(plies_to_mate + 1, 2);
        result.print_assume_capacity("mate ", .{});
        if (score < 0) {
            result.print_assume_capacity("-", .{});
        }
        result.print_assume_capacity("{}", .{ whole_moves_to_mate });
    }
    else {
        result.print_assume_capacity("cp ", .{});
        result.print_assume_capacity("{}", .{ if (!is_drawscore(score)) score else 0 });
    }

    return result;
}

/// Debug only.
pub fn verifyscore(score: i32, depth: i32, ply: i32, info: []const u8) void {
    lib.not_in_release();
    if (score > -mate_threshold and score < -20000) lib.wtf("{s} invalid score {} at depth {} ply {}", .{ info, score, depth, ply });
    if (score <  mate_threshold and score >  20000) lib.wtf("{s} invalid score {} at depth {} ply {}", .{ info, score, depth, ply });
}
