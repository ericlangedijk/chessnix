// zig fmt: off

///! Staged move picker.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const history = @import("history.zig");
const hce = @import("hce.zig");

const assert = std.debug.assert;

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Move = types.Move;
const Position = position.Position;
const ExtMove = search.ExtMove;
const Node = search.Node;
const Searcher = search.Searcher;
const ExtMoveList = search.ExtMoveList;
const History = history.History;

/// Depending on the search callsite we generate all or captures (or evasions).
pub const GenType = enum {
    Search,
    Quiescence,
};

// The current stage of the movepicker.
const Stage = enum {
    /// Stage 1: generate all moves, putting them in the correct list.
    Gen,
    /// Stage 2: the first move to consider is a tt-move.
    TTMove,
    /// Stage 3: score the noisy moves.
    ScoreNoisy,
    /// Stage 4: extract the noisy moves.
    Noisy,
    /// Stage 5: extract killer 1.
    FirstKiller,
    /// Stage 6: extract killer 2.
    SecondKiller,
    /// Stage 7: score the quiet moves.
    ScoreQuiet,
    /// Stage 8: Extract the quiet moves.
    Quiet,
    /// Stage 9: Extract the bad noisy moves.
    BadNoisy,
};

/// Depending on the stage we have to select a list.
const ListMode = enum {
    Noisies,
    Quiets,
    BadNoisies,
};

const Scores = struct {
    const tt           : Value = 8_000_000;
    const killer1      : Value = 7_000_000;
    const killer2      : Value = 6_000_000;
    const promotion    : Value = 4_000_000;
    const capture      : Value = 2_000_000;
    const bad_capture  : Value = -2_000_000;
};

/// There is no staged move generation (which is much slower).
/// Instead we put the moves in the correct list.
/// Bad noisy moves are added during the scoring of noisy moves. Extracting uses some tricky scheme.
/// There is no need to validate tt-move and killers: they pass by during move generation.
pub fn MovePicker(comptime gentype: GenType, comptime us: Color) type {

    return struct {
        const Self = @This();
        const them: Color = us.opp();

        stage: Stage,
        pos: *const Position,
        searcher: *const Searcher,
        node: *const Node,
        input_tt_move: Move,
        tt_move: ?ExtMove,
        first_killer: ?ExtMove,
        second_killer: ?ExtMove,
        move_idx: u8,
        noisies: ExtMoveList(80),
        quiets: ExtMoveList(224),
        bad_noisies: ExtMoveList(80),

        pub fn init(pos: *const Position, searcher: *const Searcher, node: *const Node, tt_move: Move) Self {
            return .{
                .stage = .Gen,
                .pos = pos,
                .searcher = searcher,
                .node = node,
                .input_tt_move = tt_move,
                .tt_move = null,
                .first_killer = null,
                .second_killer = null,
                .move_idx = 0,
                .noisies = .init(),
                .quiets = .init(),
                .bad_noisies = .init(),
            };
        }

        pub fn next(self: *Self) ?ExtMove {
            const st: Stage = self.stage;
            sw: switch (st) {
                .Gen => {
                    self.stage = .TTMove;
                    self.generate_moves();
                    continue :sw .TTMove;
                },
                .TTMove => {
                    self.stage = .ScoreNoisy;
                    if (self.tt_move != null) {
                        self.set_single_move_details(&self.tt_move.?);
                        return self.tt_move.?;
                    }
                    continue :sw .ScoreNoisy;
                },
                .ScoreNoisy => {
                    self.stage = .Noisy;
                    self.move_idx = 0;
                    self.score_moves(.Noisies);
                    continue :sw .Noisy;
                },
                .Noisy => {
                    if (self.extract_next(self.move_idx, .Noisies)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    self.stage = .FirstKiller;
                    continue :sw .FirstKiller;
                },
                .FirstKiller => {
                    self.stage = .SecondKiller;
                    if (self.first_killer != null) {
                        self.set_single_move_details(&self.first_killer.?);
                        return self.first_killer.?;
                    }
                    continue :sw .SecondKiller;
                },
                .SecondKiller => {
                    self.stage = .ScoreQuiet;
                    if (self.second_killer != null) {
                        self.set_single_move_details(&self.second_killer.?);
                        return self.second_killer.?;
                    }
                    continue :sw .ScoreQuiet;
                },
                .ScoreQuiet => {
                    self.stage = .Quiet;
                    self.move_idx = 0;
                    self.score_moves(.Quiets);
                    continue :sw .Quiet;
                },
                .Quiet => {
                    if (self.extract_next(self.move_idx, .Quiets)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    self.stage = .BadNoisy;
                    self.move_idx = 0;
                    continue :sw .BadNoisy;
                },
                .BadNoisy => {
                    if (self.extract_next(self.move_idx, .BadNoisies)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    return null;
                },
            }
        }

        /// Required function for movgen.
        pub fn reset(self: *Self) void {
            self.move_idx = 0;
            self.noisies.count = 0;
            self.quiets.count = 0;
            self.bad_noisies.count = 0;
        }

        /// Required function movgen.
        pub fn store(self: *Self, move: Move) ?void {
            if (move == self.input_tt_move) {
                self.tt_move = .init(move);
                self.tt_move.?.is_tt_move = true;
                self.tt_move.?.score = Scores.tt; // not strictly needed
            }
            else if (move == self.node.killers[0]) {
                self.first_killer = .init(move);
                self.first_killer.?.is_killer = true;
                self.first_killer.?.score = Scores.killer1; // not strictly needed
            }
            else if (move == self.node.killers[1]) {
                self.second_killer = .init(move);
                self.second_killer.?.is_killer = true;
                self.second_killer.?.score = Scores.killer2; // not strictly needed
            }
            else if (move.is_noisy()) {
                self.noisies.add(.init(move));
            }
            else {
                self.quiets.add(.init(move));
            }
        }

        /// Called during search.
        pub fn skip_quiets(self: *Self) void {
            if (self.stage == .Quiet) {
                self.stage = .BadNoisy;
                self.move_idx = 0;
            }
        }

        fn generate_moves(self: *Self) void {
            switch (gentype) {
                .Search => self.pos.generate_all_moves(us, self),
                .Quiescence => self.pos.generate_quiescence_moves(us, self),
            }
        }

        fn score_moves(self: *Self, comptime listmode: ListMode) void {
            if (comptime lib.is_paranoid) {
                assert(listmode != .BadNoisies);
            }
            const extmoves: []ExtMove = self.select_slice(listmode);
            for (extmoves) |*ex| {
                self.score_list_move(ex, listmode);
            }
        }

        /// Only used for tt_move and killers.
        fn set_single_move_details(self: *Self, ex: *ExtMove) void {
            ex.piece = self.pos.board[ex.move.from.u];
            switch (ex.move.flags) {
                Move.silent, Move.double_push, Move.castle_short, Move.castle_long => {
                    // Nothing to do.
                },
                Move.capture => {
                    ex.captured = self.pos.board[ex.move.to.u];
                },
                Move.ep => {
                    ex.captured = Piece.create(PieceType.PAWN, them);
                },
                Move.knight_promotion, Move.bishop_promotion, Move.rook_promotion, Move.queen_promotion => {
                    // Nothing to do.
                },
                Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                    ex.captured = self.pos.board[ex.move.to.u];
                },
                else => {
                    unreachable;
                }
            }
        }

        /// Besides scoring the move some details are set.
        fn score_list_move(self: *Self, ex: *ExtMove, comptime listmode: ListMode) void {

            // Some paranoid checks.
            if (comptime lib.is_paranoid) {
                assert(listmode != .BadNoisies);
                if (listmode == .Noisies) { assert(ex.move.is_noisy()); }
                else if (listmode == .Quiets) { assert(ex.move.is_quiet()); }
            }

            const pos: *const Position = self.pos;
            const hist: *const History = &self.searcher.hist;

            ex.piece = self.pos.board[ex.move.from.u];

            // Handle a noisy move
            if (listmode == .Noisies) {
                switch (ex.move.flags) {
                    Move.capture => {
                        ex.captured = pos.board[ex.move.to.u];
                        const see = hce.see_score(pos, ex.move);
                        // #testing
                        ex.is_bad_capture = see < 0; // -100;
                        if (see < 0) {
                            ex.score = Scores.bad_capture + see * 10; // 10
                        }
                        else {
                            ex.score = Scores.capture + see * 10; // 10
                        }
                        ex.score += hist.capture.get_score(ex.*);
                        // Right here, when scoring is complete, we add to the bad noisy moves.
                        if (see < 0) {
                            self.bad_noisies.add(ex.*);
                        }
                    },
                    Move.ep => {
                        ex.captured = Piece.create(PieceType.PAWN, them);
                        ex.score = Scores.capture + hist.capture.get_score(ex.*);
                    },
                    Move.knight_promotion, Move.bishop_promotion, Move.rook_promotion, Move.queen_promotion => {
                        ex.score = Scores.promotion + (ex.move.promoted_to().value() * 10);
                    },
                    Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                        ex.captured = pos.board[ex.move.to.u];
                        ex.score = Scores.promotion + (ex.move.promoted_to().value() * 10);
                        ex.score += ex.captured.value();
                        ex.score += hist.capture.get_score(ex.*);
                    },
                    else => {
                        unreachable;
                    }
                }
            }
            // Handle a quiet move.
            else if (listmode == .Quiets) {
                switch (ex.move.flags) {
                    Move.silent, Move.double_push, Move.castle_short, Move.castle_long => {
                        ex.score = hist.quiet.get_score(ex.*);
                        const parentnode: ?*const Node = if (pos.ply >= 1) &self.searcher.nodes[pos.ply - 1] else null;
                        if (parentnode) |parent| {
                            if (!parent.current_move.move.is_empty() and parent.current_move.move.is_quiet()) {
                                ex.score += hist.continuation.get_score(parent.current_move, ex.*);
                                // #testing
                                const grandparentnode: ?*const Node = if (pos.ply >= 2) &self.searcher.nodes[pos.ply - 2] else null;
                                if (grandparentnode) |grandparent| {
                                    if (!grandparent.current_move.move.is_empty() and grandparent.current_move.move.is_quiet()) {
                                        ex.score += hist.continuation.get_score(grandparent.current_move, parent.current_move);
                                    }
                                }
                            }
                        }
                    },
                    else => {
                        unreachable;
                    },
                }
            }
            // Bad noisy moves are already handled.
            else {
                unreachable;
            }
        }

        fn extract_next(self: *Self, idx: u8, comptime listmode: ListMode) ?ExtMove {
            const extmoves: []ExtMove = self.select_slice(listmode);

            if (idx >= extmoves.len) {
                return null;
            }

            var best_idx: u8 = idx;
            var max_score: Value = extmoves[idx].score;

            for (idx + 1..extmoves.len) |i| {
                const e: ExtMove = extmoves[i];
                if (e.score > max_score) {
                    max_score = e.score;
                    best_idx = @intCast(i);
                }
            }

            // If the best noisy score is a bad capture, we are done and skip to the next stage (quiets).
            if (listmode == .Noisies and extmoves[best_idx].is_bad_capture) {
                return null;
            }

            if (best_idx != idx) {
                std.mem.swap(ExtMove, &extmoves[best_idx], &extmoves[idx]);
            }

            return extmoves[idx];
        }

        fn select_slice(self: *Self, comptime listmode: ListMode) []ExtMove {
            return switch(listmode) {
                .Noisies => self.noisies.slice(),
                .Quiets => self.quiets.slice(),
                .BadNoisies => self.bad_noisies.slice(),
            };
        }
    };
}
