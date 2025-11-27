// zig fmt: off

///! History Heuristics for search.

const std = @import("std");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Move = types.Move;
const Position = position.Position;
const ExtMove = search.ExtMove;
const Node = search.Node;
const MovePicker = search.MovePicker;

pub const History = struct {
    const MAX_BETA_BONUS: SmallValue = 16384;

    const CORRECTION_HISTORY_SIZE: usize = 16384 * 2;
    const MAX_CORRECTION_HISTORY: SmallValue = 16384;
    const CORRECTION_HISTORY_GRAIN: SmallValue = 256;
    const CORRECTION_HISTORY_WEIGHT_SCALE: SmallValue = 1024;

    const CorrectionTable = [2][CORRECTION_HISTORY_SIZE]SmallValue;

    /// Used for quiet moves. Indexing: [piece][to-square]
    piece_to: [12][64]SmallValue,
    // Indexing: [prevpiece][to-square][piece][to-square]
    continuation: [12][64][12][64]SmallValue,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][pawnkey]
    pawn_correction: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][white-nonpawnkey]
    non_pawn_white: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][black-nonpawnkey]
    non_pawn_black: CorrectionTable,

    pub fn init() History {
        return std.mem.zeroes(History);
    }

    pub fn clear(self: *History) void {
        self.* = std.mem.zeroes(History);
    }

    pub fn decay(self: *History) void {
        const to_hist: []SmallValue = @ptrCast(&self.piece_to);
        for (to_hist) |*v| {
            v.* >>= 3;
        }

        const cont: []SmallValue = @ptrCast(&self.continuation);
        for (cont) |*v| {
            v.* >>= 1; // TESTING
        }
    }

    /// TODO: optimize this. and put more info in node (like position etc.) ex == node.currentmove.
    pub fn record_beta_cutoff(self: *History, parentnode: ?*const Node, node: *Node, ex: ExtMove, depth: i32, movepicker: *const MovePicker, move_idx: usize, is_quiet: bool) void {
        const move: Move = ex.move;
        if (is_quiet) {
            // Update killers of this node.
            if (node.killers[0] != move) {
                node.killers[1] = node.killers[0];
                node.killers[0] = move;
            }

            if (depth <= 1) {
                return;
            }

            const bonus: SmallValue = @intCast(depth * depth);
            const malus: SmallValue = bonus;
            const ch_bonus: SmallValue = @intCast(depth * 2);

            // Increase score for this move.
            const v: *SmallValue = &self.piece_to[ex.moved_piece.u][move.to.u];
            v.* = std.math.clamp(v.* + bonus, -MAX_BETA_BONUS, MAX_BETA_BONUS);

            // Decrease score of previous quiet moves. These did not cause a beta cutoff.
            if (move_idx > 0) {
                for (movepicker.extmoves[0..move_idx]) |prev| {
                    if (prev.move.is_quiet()) {
                        const p: *SmallValue = &self.piece_to[prev.moved_piece.u][prev.move.to.u];
                        p.* = std.math.clamp(p.* - malus, -MAX_BETA_BONUS, MAX_BETA_BONUS);
                    }
                }
            }

            // Increase continuation history score.
            if (parentnode) |prev| {
                if (!prev.current_move.move.is_empty() and prev.current_move.move.is_quiet()) {
                    const c: *SmallValue = &self.continuation[prev.current_move.moved_piece.u][prev.current_move.move.to.u][ex.moved_piece.u][ex.move.to.u];
                    c.* = std.math.clamp(c.* + ch_bonus, -4096, 4096);
                }
            }
        }

        // NOTE: capture history was a disaster.
    }

    pub fn update_correction_history(self: *History, pos: *const Position, static_eval: Value, score: Value, depth: Value, comptime us: Color) void {
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

    pub fn get_correction(self: *History, pos: *const Position, comptime us: Color) SmallValue {
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
