// zig fmt: off

///! History Heuristics for search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const search = @import("search.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const Move = types.Move;
const Position = position.Position;
const Searcher = search.Searcher;
const Nodes = search.Nodes;
const ExtMove = search.ExtMove;
const Node = search.Node;
const MovePicker = search.MovePicker;

const hist_scale: SmallValue = 135;
const hist_max_bonus: SmallValue = 1188;
const hist_max_score: SmallValue = 15176;

/// The 3 functions to handle default history bonus.
fn HistoryBonus(comptime scale: SmallValue, comptime max_bonus: SmallValue, comptime max_score: SmallValue) type {
    return struct {
        //const this = @This();

        /// `depth * scale` clamped.
        fn get_bonus(depth: i32) SmallValue {
            return @intCast(clamp(depth * scale, -max_bonus, max_bonus)); 
        }

        /// Add scaled bonus to the entry.
        fn apply_bonus(entry: *SmallValue, bonus: SmallValue) void {
            entry.* += scale_bonus(entry.*, bonus);
        }

        /// `bonus - score * abs(bonus) / max_score`.
        fn scale_bonus(score: SmallValue, bonus: SmallValue) SmallValue {
            const s: Value = score;
            return @intCast(bonus - @divTrunc(s * @abs(bonus), max_score));
        }
    };
}

// /// The 3 functions to handle correction history bonus.
// fn CorrectionBonus(comptime scale: SmallValue, comptime max_bonus: SmallValue, comptime max_score: SmallValue) type {
//     return struct {
//         const this = @This();

//         /// `depth * scale` clamped.
//         fn get_bonus(depth: i32) SmallValue {
//             return @intCast(clamp(depth * scale, -max_bonus, max_bonus)); 
//         }

//         /// Add scaled bonus to the entry.
//         fn apply_bonus(entry: *SmallValue, bonus: SmallValue) void {
//             entry.* += this.scale_bonus(entry.*, bonus);
//         }

//         /// `bonus - score * abs(bonus) / max_score`.
//         fn scale_bonus(score: SmallValue, bonus: SmallValue) SmallValue {
//             const s: Value = score;
//             return @intCast(bonus - @divTrunc(s * @abs(bonus), max_score));
//         }
//     };
// }


const hist_calc = HistoryBonus(hist_scale, hist_max_bonus, hist_max_score);

// TODO: instead of passing nodes paramaters we could use @FieldParentPtr and retrieve the nodes.

/// Container for all history heuristics.
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

    /// Increase the score of the node's move. If the node move is quiet, punish the quiets.
    pub fn record_beta_cutoff(self: *History, depth: i32, ply: u16, ex: ExtMove, nodes: []const Node, bad_quiets: []const ExtMove) void {        
        if (ex.move.is_quiet()) {
            self.quiet.update(depth, ex, bad_quiets);
            self.continuation.update(depth, ex, ply, nodes, bad_quiets);
        }
        else if (ex.move.is_capture()) {
            self.capture.update(depth, ex);
        }
    }
};

/// Heuristics for quiet moves.
pub const QuietHistory = struct {

    /// Quiet move scores. Indexing: [piece][from-square][to-square]
    table: [12][64][64]SmallValue,

    fn update(self: *QuietHistory, depth: i32, ex: ExtMove, quiets: []const ExtMove) void {
        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        const v: *SmallValue = self.get_score_ptr(ex);
        hist_calc.apply_bonus(v, bonus);

        // Decrease score of previous moves. These did not cause a beta cutoff.
        for (quiets) |prev| {
            const p: *SmallValue = self.get_score_ptr(prev);
            hist_calc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *QuietHistory, ex: ExtMove) *SmallValue {
        return &self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }

    pub fn get_score(self: *const QuietHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }

    /// EXPERIMENTAL
    pub fn owner(self: *QuietHistory) *History {
        return @fieldParentPtr("quiet", self);
    }

    /// EXPERIMENTAL
    pub fn searcher(self: *History) *Searcher {
        return self.owner().owner();
    }

};

/// Hearistics for captures moves.
pub const CaptureHistory = struct {

    /// Capture move scores. Indexing: [piece][to-square][captured-piecetype]
    table: [12][64][6]SmallValue,

    fn update(self: *CaptureHistory, depth: i32, ex: ExtMove) void {
        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        const v: *SmallValue = self.get_score_ptr(ex);
        hist_calc.apply_bonus(v, bonus);
    }

    pub fn punish(self: *CaptureHistory, depth: i32, captures: []const ExtMove) void {
        const bonus: SmallValue = hist_calc.get_bonus(depth);
        // Decrease score of previous capture moves. These did not cause a beta cutoff.
        for (captures) |prev| {
            const p: *SmallValue = self.get_score_ptr(prev);
            hist_calc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *CaptureHistory, ex: ExtMove) *SmallValue {
        return &self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }

    pub fn get_score(self: *const CaptureHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }
};

/// Heuristics for quiet continuations. 
pub const ContinuationHistory = struct {
    // #testing
    // The current scheme is:
    // - update: ply - 1
    // - retrieve: sum ply - 1, ply - 2, ply - 4

    const depths_delta: [3]u16 = .{ 1, 2, 4 };

    // Move pair scores. Indexing: [prevpiece][to-square][piece][to-square]
    table: [12][64][12][64]SmallValue,

    pub fn update(self: *ContinuationHistory, depth: i32, ex: ExtMove, ply: u16, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ply == 0) {
            return;
        }

        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        if (ply >= 1) {
            self.update_single_score(ex, ply - 1, nodes, bonus);
        }

        // Decrease score of the bad quiet moves. These did not cause a beta cutoff.
        for (bad_quiets) |bad| {
            if (ply >= 1) self.update_single_score(bad, ply - 1, nodes, -bonus);
        }
    }

    fn update_single_score(self: *ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node, bonus: SmallValue) void {
        const node: *const Node = &nodes[ply];
        if (node.current_move.move.is_empty()) {
            return;
        }
        const v: *SmallValue = &self.table[node.current_move.piece.u][node.current_move.move.to.u][ex.piece.u][ex.move.to.u];
        hist_calc.apply_bonus(v, bonus);
    }

    fn get_single_score(self: *const ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node) Value {
        const prev: *const Node = &nodes[ply];
        if (prev.current_move.move.is_empty()) {
            return 0;
        }
        return self.table[prev.current_move.piece.u][prev.current_move.move.to.u][ex.piece.u][ex.move.to.u];
    }

    pub fn get_score(self: *const ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node) Value {
        var v: Value = 0;
        inline for (depths_delta) |d|{
            if (ply >= d) {
                v += self.get_single_score(ex, ply - d, nodes);
            }
        }
        return v;
    }
};

pub const CorrectionHistory = struct {
    const SIZE: usize = 16384;
    const MASK: usize = SIZE - 1;

    const CorrectionTable = [2][SIZE]SmallValue;

    /// Small hash. Used for correction of static evaluation. Indexing: [color][pawnkey]
    pawn_correction: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][white-nonpawnkey]
    non_pawn_white: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][black-nonpawnkey]
    non_pawn_black: CorrectionTable,

    pub fn update(self: *CorrectionHistory, comptime us: Color, pos: *const Position, static_eval: Value, search_score: Value, depth: Value) void {
        
        //if (depth < 8) return;
        const u: u1 = us.u;


        // max_score = 16384 => we divide that by 64 to get the real correction (max 256)
        const bonus: SmallValue = get_bonus(depth, static_eval, search_score);
        
        // 

        update_single_score(&self.pawn_correction[u][pos.pawnkey & MASK], bonus);
        update_single_score(&self.non_pawn_white[u][pos.nonpawnkeys[Color.WHITE.u] & MASK], bonus);
        update_single_score(&self.non_pawn_black[u][pos.nonpawnkeys[Color.BLACK.u] & MASK], bonus);
        //update_single_score(&self.pawn_correction[u][pos.pawnkey & MASK], bonus);

        const get = self.get_correction(us, pos);
        lib.io.debugprint("d {} s {} e {} c {}\n", .{ depth, static_eval, search_score, get });

        // const weight: Value = @min(depth * depth + 2 * depth + 1, 128);
        // set_correction(&self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE], err, weight);
        // set_correction(&self.non_pawn_white[u][pos.nonpawnkeys[0] % CORRECTION_HISTORY_SIZE], err, weight);
        // set_correction(&self.non_pawn_black[u][pos.nonpawnkeys[1] % CORRECTION_HISTORY_SIZE], err, weight);
    }

    pub fn get_correction(self: *CorrectionHistory, comptime us: Color, pos: *const Position) SmallValue {
        const u: u1 = us.u;
        var v: Value = 0;
        v += self.pawn_correction[u][pos.pawnkey & MASK];
        v += self.non_pawn_white[u][pos.nonpawnkeys[0] & MASK];
        v += self.non_pawn_black[u][pos.nonpawnkeys[1] & MASK];
        v = @divTrunc(v, 3);

        // corr_eval >>= 9;
        // corr_eval = std.math.clamp(corr_eval, -300, 300);
        return @intCast(v);
    }

    fn update_single_score(entry: *SmallValue, bonus: SmallValue) void {
        apply_bonus(entry, bonus);
    }

    fn get_bonus(depth: i32, static_eval: Value, search_score: Value) SmallValue {
        return @intCast(clamp((search_score - static_eval) * @divTrunc(depth, 8), -256, 256));
    }

    /// Add scaled bonus to the entry.
    fn apply_bonus(entry: *SmallValue, bonus: SmallValue) void {
        entry.* += scale_bonus(entry.*, bonus);
    }

    /// `bonus - score * abs(bonus) / max_score`.
    fn scale_bonus(score: SmallValue, bonus: SmallValue) SmallValue {
        const s: Value = score;
        return @intCast(bonus - @divTrunc(s * @abs(bonus), 1024));
    }

    // fn set_correction(corr_entry: *SmallValue, err: Value, weight: Value) void {
    //     const interpolated: Value = (corr_entry.* * (CORRECTION_HISTORY_WEIGHT_SCALE - weight) + err * weight) >> 10;
    //     const clamped: Value = std.math.clamp(interpolated, -MAX_CORRECTION_HISTORY, MAX_CORRECTION_HISTORY);
    //     corr_entry.* = @intCast(clamped);
    // }


    // An update to correction history happens when the following conditions are satisfied:

    //     Side-to-move is not in check
    //     Best move either does not exist or is quiet
    //     If score type is Lower bound, then score should not be below static evaluation
    //     If score type is Upper bound, then score should not be above static evaluation    
    // In engines with history gravity updates, the correction value is applied by summing the unadjusted static evaluation with a fraction of correction history value.
    // Value to_corrected_static_eval(Value v, const Worker& w, const Position& pos) {
    //     const auto cv =
    //       w.pawnCorrectionHistory[pos.side_to_move()][pawn_structure_index<Correction>(pos)];
    //     v += 66 * cv / 512;
    //     return std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
    // }
};


