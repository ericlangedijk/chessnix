// zig fmt: off

///! History Heuristics for search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const Move = types.Move;
const Position = position.Position;
const ExtMove = search.ExtMove;
const Node = search.Node;
const MovePicker = search.MovePicker;


fn get_bonus(depth: i32, comptime max_bonus: SmallValue) SmallValue {
    return @intCast(@min(16 * depth * depth + 32 * depth + 16, max_bonus));
}

fn apply_bonus(entry: *SmallValue, bonus: SmallValue, comptime max_score: SmallValue) void {
    const abs_bonus: u31 = @intCast(@abs(bonus));
    var e: Value = entry.*;
    e += bonus - @divTrunc(e * abs_bonus, max_score);
    entry.* = @intCast(e);
}

/// Wrapper around all history heuristics.
pub const History = struct {
    quiet: QuietHistory,
    capture: CaptureHistory,
    continuation: ContinuationHistory,
    correction: CorrectionHistory,

    pub fn init() History {
        return std.mem.zeroes(History);
    }

    pub fn clear(self: *History) void {
        self.* = std.mem.zeroes(History);
    }

    pub fn record_beta_cutoff(self: *History, parentnode: ?*const Node, node: *Node, depth: i32, prev_moves: []const ExtMove) void {
        //assert(node.current_move.move == move)
        const current: ExtMove = node.current_move;

        if (current.move.is_quiet()) {

            // Update killers of this node.
            if (node.killers[0] != current.move) {
                node.killers[1] = node.killers[0];
                node.killers[0] = current.move;
            }

            // if (depth <= 1) { // TESTING #
            //     return;
            // }

            // Quiet history.
            self.quiet.update(depth, current, prev_moves);

            // Continuation history.
            if (parentnode != null) {
                const parent: ExtMove = parentnode.?.current_move;
                if (!parent.move.is_empty() and parent.move.is_quiet()) {
                    self.continuation.update(depth, parent, current, prev_moves);
                }
            }
        }
        else if (current.move.is_capture()) {
            // Capture history.
            self.capture.update(depth, current, prev_moves);
        }
    }

    //pub fn get_quiet_score()
    // pub fn get_capture_score()
};

/// Heuristics for quiet moves.
pub const QuietHistory = struct {

    const max_bonus: SmallValue = 1300;
    const max_score: SmallValue = 8000;

    /// Quiet move scores. Indexing: [piece][from-square][to-square]
    table: [12][64][64]SmallValue,

    fn update(self: *QuietHistory, depth: i32, ex: ExtMove, prev_moves: []const ExtMove) void {
        const bonus: SmallValue = get_bonus(depth, max_bonus);

        // Increase score for this move.
        const v: *SmallValue = self.get_score_ptr(ex);
        apply_bonus(v, bonus, max_score);

        // Decrease score of previous moves. These did not cause a beta cutoff.
        for (prev_moves) |prev| {
            if (prev.is_seen_by_search and prev.move.is_quiet()) {
                const p: *SmallValue = self.get_score_ptr(prev);
                apply_bonus(p, -bonus, max_score);
            }
        }
    }

    fn get_score_ptr(self: *QuietHistory, ex: ExtMove) *SmallValue {
        return &self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }

    pub fn get_score(self: *const QuietHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }
};

/// Hearistics for captures moves.
pub const CaptureHistory = struct {

    const max_bonus: SmallValue = 1300;
    const max_score: SmallValue = 8000;

    /// Capture move scores. Indexing: [piece][to-square][captured-piecetype]
    table: [12][64][6]SmallValue,

    fn update(self: *CaptureHistory, depth: i32, ex: ExtMove, prev_moves: []const ExtMove) void {
        const bonus: SmallValue = get_bonus(depth, max_bonus);

        // Increase score for this move.
        const v: *SmallValue = self.get_score_ptr(ex);
        apply_bonus(v, bonus, max_bonus);

        // Decrease score of previous capture moves. These did not cause a beta cutoff.
        for (prev_moves) |prev| {
            if (prev.is_seen_by_search and prev.move.is_capture()) {
                const p: *SmallValue = self.get_score_ptr(prev);
                apply_bonus(p, -bonus, max_score);
            }
        }
    }

    fn get_score_ptr(self: *CaptureHistory, ex: ExtMove) *SmallValue {
        return &self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }

    pub fn get_score(self: *const CaptureHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }
};

/// Heuristics for quiet continuation moves.
pub const ContinuationHistory = struct {

    const max_bonus: SmallValue = 1300;
    const max_score: SmallValue = 8000; //6000;

    // Move pair scores. Indexing: [prevpiece][to-square][piece][to-square]
    table: [12][64][12][64]SmallValue,

    fn update(self: *ContinuationHistory, depth: i32, parent: ExtMove, current: ExtMove, prev_moves: []const ExtMove) void {
        const bonus: SmallValue = get_bonus(depth, max_bonus);

        // Increase score for this move pair.
        const v: *SmallValue = self.get_score_ptr(parent, current);
        apply_bonus(v, bonus, max_score);

        // Decrease score of previous quiet moves. These did not cause a beta cutoff.
        for (prev_moves) |prev| {
            if (prev.is_seen_by_search and prev.move.is_quiet()) {
                const p: *SmallValue = self.get_score_ptr(parent, prev);
                apply_bonus(p, -bonus, max_score);
            }
        }
    }

    fn get_score_ptr(self: *ContinuationHistory, parent: ExtMove, current: ExtMove) *SmallValue {
        return &self.table[parent.piece.u][parent.move.to.u][current.piece.u][current.move.to.u];
    }

    pub fn get_score(self: *const ContinuationHistory, parent: ExtMove, current: ExtMove) SmallValue {
        return self.table[parent.piece.u][parent.move.to.u][current.piece.u][current.move.to.u];
    }

};

/// Heuristics for learning difference between static evaluation and search result.
pub const CorrectionHistory = struct {
    const CORRECTION_HISTORY_SIZE: usize = 16384 * 2;
    const MAX_CORRECTION_HISTORY: SmallValue = 16384;
    const CORRECTION_HISTORY_GRAIN: SmallValue = 256;
    const CORRECTION_HISTORY_WEIGHT_SCALE: SmallValue = 1024;

    const CorrectionTable = [2][CORRECTION_HISTORY_SIZE]SmallValue;

    /// Small hash. Used for correction of static evaluation. Indexing: [color][pawnkey]
    pawn_correction: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][white-nonpawnkey]
    non_pawn_white: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][black-nonpawnkey]
    non_pawn_black: CorrectionTable,

    pub fn update(self: *CorrectionHistory, pos: *const Position, static_eval: Value, score: Value, depth: Value, comptime us: Color) void {
        const u: u1 = us.u;
        const err : Value = (score - static_eval) * CORRECTION_HISTORY_GRAIN;
        const weight: Value = @min(depth * depth + 2 * depth + 1, 128);
        set_correction(&self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE], err, weight);
        set_correction(&self.non_pawn_white[u][pos.nonpawnkeys[0] % CORRECTION_HISTORY_SIZE], err, weight);
        set_correction(&self.non_pawn_black[u][pos.nonpawnkeys[1] % CORRECTION_HISTORY_SIZE], err, weight);
    }

    fn set_correction(corr_entry: *SmallValue, err: Value, weight: Value) void {
        const interpolated: Value = (corr_entry.* * (CORRECTION_HISTORY_WEIGHT_SCALE - weight) + err * weight) >> 10;
        const clamped: Value = std.math.clamp(interpolated, -MAX_CORRECTION_HISTORY, MAX_CORRECTION_HISTORY);
        corr_entry.* = @intCast(clamped);
    }

    pub fn get_correction(self: *CorrectionHistory, pos: *const Position, comptime us: Color) SmallValue {
        const u: u1 = us.u;
        var corr_eval: Value = 0;
        corr_eval += self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.non_pawn_white[u][pos.nonpawnkeys[0] % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.non_pawn_black[u][pos.nonpawnkeys[1] % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval >>= 9;
        //corr_eval = std.math.clamp(corr_eval, -127, 127);
        return @intCast(corr_eval);
    }
};
