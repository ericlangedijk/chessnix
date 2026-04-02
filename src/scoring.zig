// zig fmt: off

//! An attempt to centralize scoring logic, which is quite tricky.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");

const assert = std.debug.assert;

const max_search_depth = types.max_search_depth;

// A score which means 'nothing' and should be treated as such.
pub const null_score: i32 = -32002;
pub const infinity: i32 = 32000;
/// Mate in 0.
pub const mate: i32 = 30000;
/// Mate in 128.
pub const mate_threshold = mate - max_search_depth;
/// Absolute win. Normal eval should never exceed this.
pub const win = 27000;
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

// TODO: contemplate outputting 0 for 'random' drawscores (-1, 0, 1).
/// Outputs  "cp n" for a normal score and "mate n" or "-mate n" for matescores.
pub fn format_score(score: i32) utils.BoundedArray(u8, 16) {
    var print_buf: [16]u8 = undefined;
    var result: utils.BoundedArray(u8, 16) = .{};
    if (is_matescore(score)) {
        const plies_to_mate = if (score > 0) mate - score else mate + score;
        const whole_moves_to_mate = @divTrunc(plies_to_mate + 1, 2);
        result.append_slice_assume_capacity("mate ");
        if (score < 0)
            result.append_assume_capacity('-');
        result.append_slice_assume_capacity(std.fmt.bufPrint(&print_buf, "{}", .{ whole_moves_to_mate }) catch lib.wtf("print", .{}));
    }
    else {
        result.append_slice_assume_capacity("cp ");
        result.append_slice_assume_capacity(std.fmt.bufPrint(&print_buf, "{}", .{ score }) catch lib.wtf("print", .{}));
    }
    return result;
}

/// Debug only.
pub fn verifyscore(score: i32, depth: i32, ply: i32, info: []const u8) void {
    lib.not_in_release();
    if (score > -mate_threshold and score < -20000) lib.wtf("{s} invalid score {} at depth {} ply {}", .{ info, score, depth, ply });
    if (score <  mate_threshold and score >  20000) lib.wtf("{s} invalid score {} at depth {} ply {}", .{ info, score, depth, ply });
}
