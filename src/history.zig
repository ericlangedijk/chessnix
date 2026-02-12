// zig fmt: off

///! History Heuristics for search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;
const Searcher = search.Searcher;
const Nodes = search.Nodes;
const Node = search.Node;
const MovePicker = search.MovePicker;


// const hist_scale: SmallValue = 135;
// const hist_max_bonus: SmallValue = 1188;
// const hist_max_score: SmallValue = 15176;

// #hist
const hist_scale: SmallValue = 146; // try 145
const hist_max_bonus: SmallValue = 1282; // try 1280
const hist_max_score: SmallValue = 16384;

// 15176  * 1.0796 ≈ 16384 (hist_max)
//  1188  * 1.0796 ≈ 1282  (hist_max_bonus)
// 13000  * 1.0796 ≈ 14035 (lmr divider)
//  1695  * 1.0796 ≈ 1830  (lmp)
//  480   * 1.0796 ≈ 518   (lmp base)
//  135   * 1.0796 ≈ 146   (hist_scale)

const hist_calc = HistoryBonus(hist_scale, hist_max_bonus, hist_max_score);

/// The 3 functions to handle history bonus.
fn HistoryBonus(comptime scale: SmallValue, comptime max_bonus: SmallValue, comptime max_score: SmallValue) type {
    return struct {
        /// `depth * scale`.
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
            //return @intCast(bonus - @divTrunc(s * @abs(bonus), max_score));
            return @intCast(bonus - @divFloor(s * @abs(bonus), max_score)); // #testing
        }
    };
}

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
        if (lib.bughunt) {
            self.continuation.verify_node(&nodes[ply]);
        }

        if (ex.move.is_quiet()) {
            self.quiet.update(depth, ex, bad_quiets);
            ContinuationHistory.update(depth, ex, ply, nodes, bad_quiets);
        }
        else if (ex.move.is_capture()) {
            self.capture.update(depth, ex);
        }
    }

    /// Returns the history score of a quiet move. Used for move ordering and pruning decisions.
    pub fn get_quiet_score(self: *const History, ex: ExtMove, ply: u16, nodes: []const Node) Value {
        var v: Value = self.quiet.get_score(ex);
        if (ply >= 1) v += ContinuationHistory.get_single_score(&nodes[ply - 1], ex);
        if (ply >= 2) v += ContinuationHistory.get_single_score(&nodes[ply - 2], ex);
        if (ply >= 4) v += ContinuationHistory.get_single_score(&nodes[ply - 4], ex);
        return v;
    }

    /// Returns the history score of a capture move. Used for move ordering and pruning decisions.
    pub fn get_capture_score(self: *const History, ex: ExtMove) Value {
        return self.capture.get_score(ex);
    }
};

/// Heuristics for quiet moves.
pub const QuietHistory = struct {

    /// Quiet move scores. Indexing: [piece][from][to]
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

    fn get_score(self: *const QuietHistory, ex: ExtMove) Value {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }
};

/// Hearistics for capture moves.
pub const CaptureHistory = struct {

    /// Capture move scores. Indexing: [piece][to][captured-piecetype]
    table: [12][64][6]SmallValue,

    pub fn update(self: *CaptureHistory, depth: i32, ex: ExtMove) void {
        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        const v: *SmallValue = self.get_score_ptr(ex);
        hist_calc.apply_bonus(v, bonus);
    }

    pub fn punish(self: *CaptureHistory, depth: i32, captures: []const ExtMove) void {
        const bonus: SmallValue = hist_calc.get_bonus(depth);
        // Decrease score of capture moves (that did not raise alpha).
        for (captures) |prev| {
            const p: *SmallValue = self.get_score_ptr(prev);
            hist_calc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *CaptureHistory, ex: ExtMove) *SmallValue {
        return &self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }

    fn get_score(self: *const CaptureHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }
};

/// Heuristics for quiet continuation moves.
pub const ContinuationHistory = struct {

    /// The previous plies of which we update the score.
    const depths_delta: [3]u16 = .{ 1, 2, 4 }; // #testing

    /// Move pair scores. Indexing: [prevpiece][to][piece][to]
    table: [12][64][12][64]SmallValue,

    /// The node's continuation_entry is used, so we do not need Self.
    pub fn update(depth: i32, ex: ExtMove, ply: u16, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ply == 0) {
            return;
        }

        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        inline for (depths_delta) |d| {
            if (ply >= d) {
                update_single_score(&nodes[ply - d], ex, bonus);
            }
        }

        // Decrease score of the bad quiet moves (that did not cause a beta cutoff).
        for (bad_quiets) |bad| {
            inline for (depths_delta) |d| {
                if (ply >= d) {
                    update_single_score(&nodes[ply - d], bad, -bonus);
                }
            }
        }
    }

    /// The node's continuation_entry is used, so we do not need Self.
    fn update_single_score(node: *const Node, ex: ExtMove, bonus: SmallValue) void {
        if (node.continuation_entry) |entry| {
            const v: *SmallValue = &entry[ex.piece.u][ex.move.to.u];
            hist_calc.apply_bonus(v, bonus);
        }
    }

    /// The node's continuation_entry is used, so we do not need Self.
    fn get_single_score(node: *const Node, ex: ExtMove) Value {
        if (node.continuation_entry) |entry| {
            return entry[ex.piece.u][ex.move.to.u];
        }
        return 0;
    }

    /// Chess programming is crazy. In the node we store a pointer to an entry in the table.
    pub fn get_node_entry(self: *ContinuationHistory, ex: ExtMove) *[12][64]SmallValue {
        return &self.table[ex.piece.u][ex.move.to.u];
    }

    /// bughunt function.
    fn verify_node(self: *const ContinuationHistory, node: *const Node) void {
        lib.not_in_release();
        if (node.current_move.move.is_empty()) {
            lib.verify(node.continuation_entry == null, "continuation history verify_node() error. currentmove is empty but continuation_entry is not null", .{});
        }
        if (!node.current_move.move.is_empty()) {
            lib.verify(node.continuation_entry != null, "continuation history verify_node() error. currentmove is not empty but continuation_entry is null", .{});
        }
        if (node.continuation_entry == null) {
            return;
        }
        lib.verify(node.continuation_entry == &self.table[node.current_move.piece.u][node.current_move.move.to.u], "continuation history verify_node() error. continuation_entry address mismatch", .{});
    }
};

var MIN: Value = 1000;
var MAX: Value = -1000;

pub const CorrectionHistory = struct {
    // I chose a big size here. It tested better. Probably less collisions.
    const SIZE: usize = 16384 * 2;

    const corr_scale: Value = 256;
    const corr_max: Value = 64;

    /// Entries for pawns
    pawn_table: [2][SIZE]SmallValue,
    /// Entries for white pieces
    white_table: [2][SIZE]SmallValue,
    /// Entries for black pieces.
    black_table: [2][SIZE]SmallValue,

    /// Updates the error values: the difference between the search score and the raw static eval.
    pub fn update(self: *CorrectionHistory, comptime us: Color, pos: *const Position, depth: i32, search_score: Value, raw_static_eval: Value) void {
        const err: Value = search_score - raw_static_eval;
        const scaled_bonus: Value = err * corr_scale;
        const weight: Value = @min(1 + depth, 16);

        const pawn_entry: *SmallValue = &self.pawn_table[us.u][pos.pawnkey % SIZE];
        const white_entry: *SmallValue = &self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % SIZE];
        const black_entry: *SmallValue = &self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % SIZE];

        update_entry(pawn_entry, scaled_bonus, weight);
        update_entry(white_entry, scaled_bonus, weight);
        update_entry(black_entry, scaled_bonus, weight);
    }

    /// Returns a corrected raw static eval.
    pub fn apply(self: *const CorrectionHistory, comptime us: Color, pos: *const Position, raw_static_eval: Value) Value {

        const p: Value = self.pawn_table[us.u][pos.pawnkey % SIZE];
        const w: Value = self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % SIZE];
        const b: Value = self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % SIZE];

        var correction: Value = 0;
        //correction += @divFloor(p * 150, 100); //p * 2;
        correction += p * 2;
        correction += w;
        correction += b;

        correction = @divFloor(correction, corr_scale * 3); // #testing 3

        const adjusted: Value = raw_static_eval + correction;
        const result: Value = clamp(adjusted, -types.mate_threshold + 1, types.mate_threshold - 1);
        return result;
    }

    fn update_entry(entry: *SmallValue, scaled_bonus: Value, weight: Value) void {
        var score: Value = entry.*;
        score = @divFloor(score * (corr_scale - weight) + scaled_bonus * weight, corr_scale);
        score = clamp(score, corr_scale * -corr_max, corr_scale * corr_max);
        entry.* = @intCast(score);
    }
};
