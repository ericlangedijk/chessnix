// zig fmt: off

///! History Heuristics guiding search and move ordering.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;

const Score = types.Score;
const SmallScore = types.SmallScore;
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

const hist_scale: SmallScore = 146;
const hist_max_bonus: SmallScore = 1282;
const hist_max_score: SmallScore = 16384;

const hist_calc = HistoryBonus(hist_scale, hist_max_bonus, hist_max_score);

/// The 3 functions to handle history bonus.
fn HistoryBonus(comptime scale: SmallScore, comptime max_bonus: SmallScore, comptime max_score: SmallScore) type {
    return struct {
        /// `depth * scale`.
        fn get_bonus(depth: i32) SmallScore {
            return @intCast(clamp(depth * scale, -max_bonus, max_bonus));
        }

        /// Add scaled bonus to the entry.
        fn apply_bonus(entry: *SmallScore, bonus: SmallScore) void {
            entry.* += scale_bonus(entry.*, bonus);
        }

        /// `bonus - score * abs(bonus) / max_score`.
        fn scale_bonus(score: SmallScore, bonus: SmallScore) SmallScore {
            const s: Score = score;
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
    pub fn get_quiet_score(self: *const History, ex: ExtMove, ply: u16, nodes: []const Node) Score {
        var v: Score = self.quiet.get_score(ex);
        if (ply >= 1) v += ContinuationHistory.get_single_score(&nodes[ply - 1], ex);
        if (ply >= 2) v += ContinuationHistory.get_single_score(&nodes[ply - 2], ex);
        if (ply >= 4) v += ContinuationHistory.get_single_score(&nodes[ply - 4], ex);
        return v;
    }

    /// Returns the history score of a capture move. Used for move ordering and pruning decisions.
    pub fn get_capture_score(self: *const History, ex: ExtMove) Score {
        return self.capture.get_score(ex);
    }
};

/// Heuristics for quiet moves.
pub const QuietHistory = struct {

    /// Quiet move scores. Indexing: [piece][from][to]
    table: [12][64][64]SmallScore,

    fn update(self: *QuietHistory, depth: i32, ex: ExtMove, quiets: []const ExtMove) void {
        const bonus: SmallScore = hist_calc.get_bonus(depth);

        // Increase score for this move.
        const v: *SmallScore = self.get_score_ptr(ex);
        hist_calc.apply_bonus(v, bonus);

        // Decrease score of previous moves. These did not cause a beta cutoff.
        for (quiets) |prev| {
            const p: *SmallScore = self.get_score_ptr(prev);
            hist_calc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *QuietHistory, ex: ExtMove) *SmallScore {
        return &self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }

    fn get_score(self: *const QuietHistory, ex: ExtMove) Score {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }
};

/// Hearistics for capture moves.
pub const CaptureHistory = struct {

    /// Capture move scores. Indexing: [piece][to][captured-piecetype]
    table: [12][64][6]SmallScore,

    pub fn update(self: *CaptureHistory, depth: i32, ex: ExtMove) void {
        const bonus: SmallScore = hist_calc.get_bonus(depth);

        // Increase score for this move.
        const v: *SmallScore = self.get_score_ptr(ex);
        hist_calc.apply_bonus(v, bonus);
    }

    pub fn punish(self: *CaptureHistory, depth: i32, captures: []const ExtMove) void {
        const bonus: SmallScore = hist_calc.get_bonus(depth);
        // Decrease score of capture moves (that did not raise alpha).
        for (captures) |prev| {
            const p: *SmallScore = self.get_score_ptr(prev);
            hist_calc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *CaptureHistory, ex: ExtMove) *SmallScore {
        return &self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }

    fn get_score(self: *const CaptureHistory, ex: ExtMove) SmallScore {
        return self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }
};

/// Heuristics for quiet continuation moves.
pub const ContinuationHistory = struct {

    /// The previous plies of which we update the score.
    const depths_delta: [3]u16 = .{ 1, 2, 4 }; // #testing

    /// Move pair scores. Indexing: [prevpiece][to][piece][to]
    table: [12][64][12][64]SmallScore,

    /// The node's continuation_entry is used, so we do not need Self.
    pub fn update(depth: i32, ex: ExtMove, ply: u16, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ply == 0) {
            return;
        }

        const bonus: SmallScore = hist_calc.get_bonus(depth);

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
    fn update_single_score(node: *const Node, ex: ExtMove, bonus: SmallScore) void {
        if (node.continuation_entry) |entry| {
            const v: *SmallScore = &entry[ex.piece.u][ex.move.to.u];
            hist_calc.apply_bonus(v, bonus);
        }
    }

    /// The node's continuation_entry is used, so we do not need Self.
    fn get_single_score(node: *const Node, ex: ExtMove) Score {
        if (node.continuation_entry) |entry| {
            return entry[ex.piece.u][ex.move.to.u];
        }
        return 0;
    }

    /// Chess programming is crazy. In the node we store a pointer to an entry in the table.
    pub fn get_node_entry(self: *ContinuationHistory, ex: ExtMove) *[12][64]SmallScore {
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

/// Heuristics for evaluation correction.
///
/// The math is quite confusing to me and the result is based on trial and error and probably mathematically incorrect.
/// 1) We get a raw static evaluation from hce.
/// 2) During search we call `update`, comparing the search score with the raw static evaluation score and storing the difference it in the 3 tables.
/// 3) After retrieving a new raw static eval we adjust it using `apply`.
/// 4) the size of the table is twice as big as I usually see in other engines.
pub const CorrectionHistory = struct {
    const table_size: usize = 16384 * 2;

    const corr_scale: Score = 256;
    const corr_max: Score = 64;

    /// Entries for pawns
    pawn_table: [2][table_size]SmallScore,
    /// Entries for white pieces
    white_table: [2][table_size]SmallScore,
    /// Entries for black pieces.
    black_table: [2][table_size]SmallScore,

    /// Updates the error values: the difference between the search score and the raw static eval.
    pub fn update(self: *CorrectionHistory, comptime us: Color, pos: *const Position, depth: i32, search_score: Score, raw_static_eval: Score) void {
        const err: Score = search_score - raw_static_eval;
        const scaled_err: Score = err * corr_scale;
        const weight: Score = @min(1 + depth, 16);

        const pawn_entry: *SmallScore = &self.pawn_table[us.u][pos.pawnkey % table_size];
        const white_entry: *SmallScore = &self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % table_size];
        const black_entry: *SmallScore = &self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % table_size];

        update_entry(pawn_entry, scaled_err, weight);
        update_entry(white_entry, scaled_err, weight);
        update_entry(black_entry, scaled_err, weight);
    }

    /// Returns a corrected raw static eval.
    pub fn apply(self: *const CorrectionHistory, comptime us: Color, pos: *const Position, raw_static_eval: Score) Score {
        const p: Score = self.pawn_table[us.u][pos.pawnkey % table_size];
        const w: Score = self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % table_size];
        const b: Score = self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % table_size];

        // The 2 is no typo. Maybe it is 'wrong' but it performed quite ok.
        const correction: Score = @divFloor(p + w + b, corr_scale * 2);
        const result: Score = clamp(raw_static_eval + correction, -types.mate_threshold + 1, types.mate_threshold - 1);
        return result;
    }

    fn update_entry(entry: *SmallScore, scaled_err: Score, weight: Score) void {
        var score: Score = entry.*;
        score = @divFloor(score * (corr_scale - weight) + scaled_err * weight, corr_scale);
        score = clamp(score, corr_scale * -corr_max, corr_scale * corr_max);
        entry.* = @intCast(score);
    }
};
