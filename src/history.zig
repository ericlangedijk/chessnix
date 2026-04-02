// zig fmt: off

///! History Heuristics guiding search and move ordering.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");
const consts = @import("consts.zig");
const scoring = @import("scoring.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;

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

const tuned = consts.tuned;

const HistCalc = struct {
    /// depth & scale
    fn get_bonus(depth: i32) i16 {
        return @intCast(clamp(depth * tuned.history_scale, -tuned.history_max_bonus, tuned.history_max_bonus));
    }

    /// Add scaled bonus to the entry.
    fn apply_bonus(entry: *i16, bonus: i16) void {
        entry.* += scale_bonus(entry.*, bonus);
    }

    /// `bonus - score * abs(bonus) / max_score`.
    fn scale_bonus(score: i16, bonus: i16) i16 {
        const s: i32 = score;
        return @intCast(bonus - @divFloor(s * @abs(bonus), tuned.history_max_score));
    }
};

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
        if (comptime lib.verifications) {
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
    pub fn get_quiet_score(self: *const History, ex: ExtMove, ply: u16, nodes: []const Node) i32 {
        var v: i32 = self.quiet.get_score(ex);
        if (ply >= 1) v += ContinuationHistory.get_single_score(&nodes[ply - 1], ex);
        if (ply >= 2) v += ContinuationHistory.get_single_score(&nodes[ply - 2], ex);
        if (ply >= 4) v += ContinuationHistory.get_single_score(&nodes[ply - 4], ex);
        return v;
    }

    /// Returns the history score of a capture move. Used for move ordering and pruning decisions.
    pub fn get_capture_score(self: *const History, ex: ExtMove) i32 {
        return self.capture.get_score(ex);
    }
};

/// Heuristics for quiet moves.
pub const QuietHistory = struct {

    /// Quiet move scores. Indexing: [piece][from][to]
    table: [12][64][64]i16,

    fn update(self: *QuietHistory, depth: i32, ex: ExtMove, quiets: []const ExtMove) void {
        const bonus: i16 = HistCalc.get_bonus(depth);
        //const malus: i16 = hist_calc.get_malus(depth);

        // Increase score for this move.
        const v: *i16 = self.get_score_ptr(ex);
        HistCalc.apply_bonus(v, bonus);

        // Decrease score of previous moves. These did not cause a beta cutoff.
        for (quiets) |prev| {
            const p: *i16 = self.get_score_ptr(prev);
            HistCalc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *QuietHistory, ex: ExtMove) *i16 {
        return &self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }

    fn get_score(self: *const QuietHistory, ex: ExtMove) i32 {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }
};

/// Hearistics for capture moves.
pub const CaptureHistory = struct {

    /// Capture move scores. Indexing: [piece][to][captured-piecetype]
    table: [12][64][6]i16,

    pub fn update(self: *CaptureHistory, depth: i32, ex: ExtMove) void {
        const bonus: i16 = HistCalc.get_bonus(depth);

        // Increase score for this move.
        const v: *i16 = self.get_score_ptr(ex);
        HistCalc.apply_bonus(v, bonus);
    }

    pub fn punish(self: *CaptureHistory, depth: i32, captures: []const ExtMove) void {
        const bonus: i16 = HistCalc.get_bonus(depth);
        //const malus: i16 = hist_calc.get_malus(depth);
        // Decrease score of capture moves (that did not raise alpha).
        for (captures) |prev| {
            const p: *i16 = self.get_score_ptr(prev);
            HistCalc.apply_bonus(p, -bonus);
        }
    }

    fn get_score_ptr(self: *CaptureHistory, ex: ExtMove) *i16 {
        return &self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }

    fn get_score(self: *const CaptureHistory, ex: ExtMove) i16 {
        return self.table[ex.piece.u][ex.move.to.u][ex.captured.piecetype().u];
    }
};

pub const ContinuationEntry = *[12][64]i16;

/// Heuristics for quiet continuation moves.
pub const ContinuationHistory = struct {

    /// The previous plies of which we update the score.
    const depths_delta: [3]u16 = .{ 1, 2, 4 };

    /// Move pair scores. Indexing: [prevpiece][to][piece][to]
    table: [12][64][12][64]i16,

    /// The node's continuation_entry is used, so we do not need Self.
    pub fn update(depth: i32, ex: ExtMove, ply: u16, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ply == 0) {
            return;
        }

        const bonus: i16 = HistCalc.get_bonus(depth);
        //const malus: i16 = hist_calc.get_malus(depth);

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
    fn update_single_score(node: *const Node, ex: ExtMove, bonus: i16) void {
        if (node.continuation_entry) |entry| {
            const v: *i16 = &entry[ex.piece.u][ex.move.to.u];
            HistCalc.apply_bonus(v, bonus);
        }
    }

    /// The node's continuation_entry is used, so we do not need Self.
    fn get_single_score(node: *const Node, ex: ExtMove) i32 {
        if (node.continuation_entry) |entry| {
            return entry[ex.piece.u][ex.move.to.u];
        }
        return 0;
    }

    /// Chess programming is crazy. In the node we store a pointer to an entry in the table.
    pub fn get_continuation_entry_for_node(self: *ContinuationHistory, ex: ExtMove) ContinuationEntry {
        return &self.table[ex.piece.u][ex.move.to.u];
    }

    /// verifications function.
    fn verify_node(self: *const ContinuationHistory, node: *const Node) void {
        lib.not_in_release();
        if (node.current_move.move.is_empty()) {
            lib.verify(node.continuation_entry == null, "verify_node #1", .{});
        }
        if (!node.current_move.move.is_empty()) {
            lib.verify(node.continuation_entry != null, "verify_node #2", .{});
        }
        if (node.continuation_entry == null) {
            return;
        }
        lib.verify(node.continuation_entry == &self.table[node.current_move.piece.u][node.current_move.move.to.u], "verify_node #3", .{});
    }
};


/// Small entitities.
pub const CorrectionHistory = struct {
    const table_size: usize = 16384;

    /// Entries for pawns. Indexing: [color][position.pawnhash % tablesize]
    pawn_table: [2][table_size]i16,
    /// Entries for white pieces. Indexing: [color][position.non_pawns_white_key % tablesize]
    white_table: [2][table_size]i16,
    /// Entries for black pieces. Indexing: [color][position.non_pawns_black_key % tablesize]
    black_table: [2][table_size]i16,
    // Entries for minors. Indexing: [color][position.minorkey-index % tablesize]
    minor_table: [2][table_size]i16,
    // Entries for majors. Indexing: [color][position.majorkey-index % tablesize]
    major_table: [2][table_size]i16,

    /// Updates the error values: the difference between the search score and the static eval.
    /// The static_eval argument must be the value after correction to prevent explosion.
    pub fn update(self: *CorrectionHistory, comptime us: Color, pos: *const Position, depth: i32, search_score: i32, static_eval: i32) void {
        const err: i32 = search_score - static_eval;

        if (@abs(err) <= 8) {
            return;
        }

        const scaled_err: i32 = err * tuned.corr_hist_scale;
        const weight: i32 = @min(1 + depth, 16) * 2;

        const pawn_entry: *i16 = &self.pawn_table[us.u][pos.pawnkey % table_size];
        const white_entry: *i16 = &self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % table_size];
        const black_entry: *i16 = &self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % table_size];
        const minor_entry: *i16 = &self.minor_table[us.u][pos.minorkey % table_size];
        const major_entry: *i16 = &self.major_table[us.u][pos.majorkey % table_size];

        update_entry(pawn_entry, scaled_err, weight);
        update_entry(white_entry, scaled_err, weight);
        update_entry(black_entry, scaled_err, weight);
        update_entry(minor_entry, scaled_err, weight);
        update_entry(major_entry, scaled_err, weight);
    }

    /// Returns a corrected static eval.
    pub fn apply(self: *const CorrectionHistory, comptime us: Color, pos: *const Position, static_eval: i32) i32 {
        const p: i32 = self.pawn_table[us.u][pos.pawnkey % table_size];
        const w: i32 = self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % table_size];
        const b: i32 = self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % table_size];
        const m1: i32 = self.minor_table[us.u][pos.minorkey % table_size];
        const m2: i32 = self.major_table[us.u][pos.majorkey % table_size];

        var correction: i32 = @divFloor(p + w + b + m1 + m2, tuned.corr_hist_scale);

        correction = clamp(correction, -tuned.corr_hist_max_applied_correction, tuned.corr_hist_max_applied_correction);
        return clamp(static_eval + correction, -scoring.mate_threshold + 1, scoring.mate_threshold - 1);
    }

    /// TODO: can probably be removed. This does not work.
    pub fn is_complex(self: *const CorrectionHistory, comptime us: Color, pos: *const Position) bool {
        const entries: [5]i16 = .{

            self.pawn_table[us.u][pos.pawnkey % table_size],
            self.white_table[us.u][pos.nonpawnkeys[Color.WHITE.u] % table_size],
            self.black_table[us.u][pos.nonpawnkeys[Color.BLACK.u] % table_size],
            self.minor_table[us.u][pos.minorkey % table_size],
            self.major_table[us.u][pos.majorkey % table_size],
        };

        var hi: i32 = 0;
        var lo: i32 = 0;
        inline for (entries) |e| {
            if (e >= 128) hi += 1 else if (e <= -128) lo += 1;
        }
        return (hi == 3 and lo == 2) or (hi == 2 and lo == 3);
    }

    fn update_entry(entry: *i16, scaled_err: i32, weight: i32) void {
        const max: i32 = tuned.corr_hist_scale * tuned.corr_hist_max_entry_bonus;
        //const old: i32 = entry.*;
        var score: i32 = entry.*;
        score = @divFloor(score * (tuned.corr_hist_scale - weight) + (scaled_err * weight), tuned.corr_hist_scale);
        //score = clamp(score, old - 256 * 8, old + 256 * 8);
        score = clamp(score, -max, max);
        entry.* = @intCast(score);
    }
};
