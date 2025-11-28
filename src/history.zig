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

/// Wrapper around all history heuristics.
pub const History = struct {
    quiet: QuietHistory,
    continuation: ContinuationHistory,
    correction: CorrectionHistory,

    pub fn init() History {
        return std.mem.zeroes(History);
    }

    pub fn clear(self: *History) void {
        self.* = std.mem.zeroes(History);
    }

    /// Call after each move in a game.
    pub fn decay(self: *History) void {
        self.quiet.decay();
        self.continuation.decay();
    }

    pub fn record_beta_cutoff(self: *History, parentnode: ?*const Node, node: *Node, depth: i32, movepicker: *const MovePicker, move_idx: usize) void {
        const current: ExtMove = node.current_move;

        if (current.move.is_quiet()) {

            // Update killers of this node.
            if (node.killers[0] != current.move) {
                node.killers[1] = node.killers[0];
                node.killers[0] = current.move;
            }

            // if (depth <= 1) {
            //     return;
            // }

            // Quiet history.
            self.quiet.update(depth, current, movepicker, move_idx);

            // Continuation history.
            if (parentnode != null) {
                const parent: ExtMove = parentnode.?.current_move;
                if (!parent.move.is_empty() and parent.move.is_quiet()) {
                    self.continuation.update(depth, parent, current);
                }
            }
        }
    }
};

pub const QuietHistory = struct {
    const max_bonus: SmallValue = 16384;

    /// Quiet move scores. Indexing: [piece][from-square][to-square]
    piece_from_to: [12][64][64]SmallValue,

    fn decay(self: *QuietHistory) void {
        const all: []SmallValue = @ptrCast(&self.piece_from_to);
        for (all) |*v| {
            v.* >>= 2;
        }
    }

    /// Only call for quiet move.
    fn update(self: *QuietHistory, depth: i32, ex: ExtMove, movepicker: *const MovePicker, move_idx: usize) void {
        const bonus: SmallValue = @intCast(depth * depth);

        // Increase score for this move.
        const v: *SmallValue = self.get_value_ptr(ex);
        v.* = std.math.clamp(v.* + bonus, -max_bonus, max_bonus);

        // Decrease score of previous quiet moves. These did not cause a beta cutoff.
        if (move_idx > 0) {
            for (movepicker.extmoves[0..move_idx]) |prev| {
                if (prev.move.is_quiet()) {
                    const p: *SmallValue = self.get_value_ptr(prev);
                    p.* = std.math.clamp(p.* - bonus, -max_bonus, max_bonus);
                }
            }
        }
    }

    /// Only call for quiet move.
    fn get_value_ptr(self: *QuietHistory, ex: ExtMove) *SmallValue {
        return &self.piece_from_to[ex.moved_piece.u][ex.move.from.u][ex.move.to.u];
    }

    /// Only call for quiet move.
    pub fn get_score(self: *const QuietHistory, ex: ExtMove) SmallValue {
        return self.piece_from_to[ex.moved_piece.u][ex.move.from.u][ex.move.to.u];
    }
};

pub const ContinuationHistory = struct {

    // Move pair scores. Indexing: [prevpiece][to-square][piece][to-square]
    table: [12][64][12][64]SmallValue,

    fn decay(self: *ContinuationHistory) void {
        const all: []SmallValue = @ptrCast(&self.table);
        for (all) |*v| {
            v.* >>= 1;
        }
    }

    /// Only call for quiet moves.
    fn update(self: *ContinuationHistory, depth: i32, parent: ExtMove, current: ExtMove) void {
        // Increase score for this move pair.
        const ch_bonus: SmallValue = @intCast(depth * 2);
        const c: *SmallValue = self.get_value_ptr(parent, current);
        c.* = std.math.clamp(c.* + ch_bonus, -4096, 4096);
    }

    /// Only call for quiet moves.
    fn get_value_ptr(self: *ContinuationHistory, parent: ExtMove, current: ExtMove) *SmallValue {
        return &self.table[parent.moved_piece.u][parent.move.to.u][current.moved_piece.u][current.move.to.u];
    }

    /// Only call for quiet move.
    pub fn get_score(self: *const ContinuationHistory, parent: ExtMove, current: ExtMove) SmallValue {
        return self.table[parent.moved_piece.u][parent.move.to.u][current.moved_piece.u][current.move.to.u];
    }

};

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

    pub fn get_correction(self: *const CorrectionHistory, pos: *const Position, comptime us: Color) SmallValue {
        const u: u1 = us.u;
        var corr_eval: Value = 0;
        corr_eval += self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.non_pawn_white[u][pos.nonpawnkeys[0] % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.non_pawn_black[u][pos.nonpawnkeys[1] % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval >>= 9;
        corr_eval = std.math.clamp(corr_eval, -127, 127);
        return @intCast(corr_eval);
    }
};