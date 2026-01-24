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
            return @intCast(bonus - @divTrunc(s * @abs(bonus), max_score));
        }
    };
}

const hist_calc = HistoryBonus(hist_scale, hist_max_bonus, hist_max_score);

/// Container for all history heuristics.
pub const History = struct {
    quiet: QuietHistory,
    capture: CaptureHistory,
    continuation: ContinuationHistory,

    pub fn init() History {
        return std.mem.zeroes(History);
    }

    pub fn clear(self: *History) void {
        self.* = std.mem.zeroes(History);
    }

    /// Increase the score of the node's move. If the node move is quiet, punish the quiets.
    pub fn record_beta_cutoff(self: *History, depth: i32, ply: u16, ex: ExtMove, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ex.is_quiet) {
            self.quiet.update(depth, ex, bad_quiets);
            self.continuation.update(depth, ex, ply, nodes, bad_quiets);
        }
        else if (ex.is_capture) {
            self.capture.update(depth, ex);
        }
    }

    pub fn get_quiet_score(self: *const History, ex: ExtMove, ply: u16, nodes: []const Node) Value {
        var v: Value = 0;
        v += self.quiet.get_score(ex);
        v += self.continuation.get_score(ex, ply, nodes);
        return v;
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

    /// TODO: make i32.
    pub fn get_score(self: *const QuietHistory, ex: ExtMove) SmallValue {
        return self.table[ex.piece.u][ex.move.from.u][ex.move.to.u];
    }
};

/// Hearistics for captures moves.
pub const CaptureHistory = struct {

    /// Capture move scores. Indexing: [piece][to-square][captured-piecetype]
    table: [12][64][6]SmallValue,

    pub fn update(self: *CaptureHistory, depth: i32, ex: ExtMove) void {
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
    const depths_delta: [4]u16 = .{ 1, 2, 4, 6 };
    // const depths_delta: [3]u16 = .{ 1, 2, 4 };

    // Move pair scores. Indexing: [prevpiece][to-square][piece][to-square]
    table: [12][64][12][64]SmallValue,

    pub fn update(self: *ContinuationHistory, depth: i32, ex: ExtMove, ply: u16, nodes: []const Node, bad_quiets: []const ExtMove) void {
        if (ply == 0) {
            return;
        }

        const bonus: SmallValue = hist_calc.get_bonus(depth);

        // Increase score for this move.
        inline for (depths_delta) |d| {
            if (ply >= d) {
                self.update_single_score(ex, ply - d, nodes, bonus);
            }
        }

        // Decrease score of the bad quiet moves. These did not cause a beta cutoff.
        for (bad_quiets) |bad| {
            inline for (depths_delta) |d| {
                if (ply >= d) {
                    self.update_single_score(bad, ply - d, nodes, -bonus);
                }
            }
        }
    }

    pub fn get_score(self: *const ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node) Value {
        return if (ply >= 1) self.get_single_score(ex, ply - 1, nodes) else 0;
    }

    pub fn get_node_entry(self: *ContinuationHistory, ex: ExtMove) *[12][64]SmallValue {
        return &self.table[ex.piece.u][ex.move.to.u];
    }

    fn update_single_score(self: *ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node, bonus: SmallValue) void {

        _ = self;
        const node: *const Node = &nodes[ply];

        if (node.continuation_entry) |e| {
            const v: *SmallValue = &e[ex.piece.u][ex.move.to.u];
            hist_calc.apply_bonus(v, bonus);
        }

        // const node: *const Node = &nodes[ply];
        // if (node.current_move.move.is_empty()) {
        //     return;
        // }
        // const v: *SmallValue = &self.table[node.current_move.piece.u][node.current_move.move.to.u][ex.piece.u][ex.move.to.u];
        // hist_calc.apply_bonus(v, bonus);
    }

    fn get_single_score(self: *const ContinuationHistory, ex: ExtMove, ply: u16, nodes: []const Node) Value {

        _ = self;
        const node: *const Node = &nodes[ply];
        if (node.continuation_entry) |e| {
            return e[ex.piece.u][ex.move.to.u];
        }
        return 0;

        // const prev: *const Node = &nodes[ply];
        // if (prev.current_move.move.is_empty()) {
        //     return 0;
        // }
        // return self.table[prev.current_move.piece.u][prev.current_move.move.to.u][ex.piece.u][ex.move.to.u];
    }

};
