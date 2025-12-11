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

/// Depending on the search callsite we generate all moves or quiescence moves.
pub const GenType = enum {
    search,
    quiescence,
};

// The current stage of the movepicker.
pub const Stage = enum {
    /// Stage 1: generate all moves, putting them in the correct list.
    generate,
    /// Stage 2: the first move to consider is a tt-move.
    tt,
    /// Stage 3: score the noisy moves.
    score_noisy,
    /// Stage 4: extract the noisy moves.
    noisy,
    /// Stage 5: extract killer 1.
    first_killer,
    /// Stage 6: extract killer 2.
    second_killer,
    /// Stage 7: score the quiet moves.
    score_quiet,
    /// Stage 8: Extract the quiet moves.
    quiet,
    /// Stage 9: Extract the bad noisy moves.
    bad_noisy,
};

/// Depending on the stage we have to select a list.
const ListMode = enum {
    noisies,
    quiets,
    bad_noisies,
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
        first_killer_move: ?ExtMove,
        second_killer_move: ?ExtMove,
        move_count: u8,
        move_idx: u8,
        noisies: ExtMoveList(80),
        quiets: ExtMoveList(224),
        bad_noisies: ExtMoveList(80),

        pub fn init(pos: *const Position, searcher: *const Searcher, node: *const Node, tt_move: Move) Self {
            return .{
                .stage = .generate,
                .pos = pos,
                .searcher = searcher,
                .node = node,
                .input_tt_move = tt_move,
                .tt_move = null,
                .first_killer_move = null,
                .second_killer_move = null,
                .move_count = 0,
                .move_idx = 0,
                .noisies = .init(),
                .quiets = .init(),
                .bad_noisies = .init(),
            };
        }

        pub fn next(self: *Self) ?ExtMove {
            const st: Stage = self.stage;
            sw: switch (st) {
                .generate => {
                    self.stage = .tt;
                    self.generate_moves();
                    continue :sw .tt;
                },
                .tt => {
                    self.stage = .score_noisy;
                    if (self.tt_move != null) {
                        self.set_single_move_details(&self.tt_move.?);
                        return self.tt_move.?;
                    }
                    continue :sw .score_noisy;
                },
                .score_noisy => {
                    self.stage = .noisy;
                    self.move_idx = 0;
                    self.score_moves(.noisies);
                    continue :sw .noisy;
                },
                .noisy => {
                    if (self.extract_next(self.move_idx, .noisies)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    self.stage = .first_killer;
                    continue :sw .first_killer;
                },
                .first_killer => {
                    self.stage = .second_killer;
                    if (self.first_killer_move != null) {
                        self.set_single_move_details(&self.first_killer_move.?);
                        return self.first_killer_move.?;
                    }
                    continue :sw .second_killer;
                },
                .second_killer => {
                    self.stage = .score_quiet;
                    if (self.second_killer_move != null) {
                        self.set_single_move_details(&self.second_killer_move.?);
                        return self.second_killer_move.?;
                    }
                    continue :sw .score_quiet;
                },
                .score_quiet => {
                    self.stage = .quiet;
                    self.move_idx = 0;
                    self.score_moves(.quiets);
                    continue :sw .quiet;
                },
                .quiet => {
                    if (self.extract_next(self.move_idx, .quiets)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    self.stage = .bad_noisy;
                    self.move_idx = 0;
                    continue :sw .bad_noisy;
                },
                .bad_noisy => {
                    if (self.extract_next(self.move_idx, .bad_noisies)) |ex| {
                        self.move_idx += 1;
                        return ex;
                    }
                    return null;
                },
            }
        }

        /// Required function for movgen.
        pub fn reset(self: *Self) void {
            self.move_count = 0;
            self.move_idx = 0;
            self.noisies.count = 0;
            self.quiets.count = 0;
            self.bad_noisies.count = 0;
        }

        /// Required function movgen.
        pub fn store(self: *Self, move: Move) ?void {
            self.move_count += 1;
            if (move == self.input_tt_move) {
                self.tt_move = .init(move);
                self.tt_move.?.is_tt_move = true;
                self.tt_move.?.score = Scores.tt; // not strictly needed
            }
            else if (move == self.node.killers[0]) {
                self.first_killer_move = .init(move);
                self.first_killer_move.?.is_killer = true;
                self.first_killer_move.?.score = Scores.killer1; // not strictly needed
            }
            else if (move == self.node.killers[1]) {
                self.second_killer_move = .init(move);
                self.second_killer_move.?.is_killer = true;
                self.second_killer_move.?.score = Scores.killer2; // not strictly needed
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
            if (self.stage == .quiet) {
                self.stage = .bad_noisy;
                self.move_idx = 0;
            }
        }

        fn generate_moves(self: *Self) void {
            switch (gentype) {
                .search => self.pos.generate_all_moves(us, self),
                .quiescence => self.pos.generate_quiescence_moves(us, self),
            }
        }

        fn score_moves(self: *Self, comptime listmode: ListMode) void {
            if (comptime lib.is_paranoid) {
                assert(listmode != .bad_noisies);
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
                assert(listmode != .bad_noisies);
                if (listmode == .noisies) { assert(ex.move.is_noisy()); }
                else if (listmode == .quiets) { assert(ex.move.is_quiet()); }
            }

            const pos: *const Position = self.pos;
            const hist: *const History = &self.searcher.hist;

            ex.piece = self.pos.board[ex.move.from.u];

            // Handle a noisy move
            if (listmode == .noisies) {
                switch (ex.move.flags) {
                    Move.capture => {
                        ex.captured = pos.board[ex.move.to.u];

                        // #testing see score.
                        const see = hce.see_score(pos, ex.move);
                        ex.is_bad_capture = see < 0;
                        if (see < 0) {
                            ex.score = Scores.bad_capture + see * 10;
                        }
                        else {
                            ex.score = Scores.capture + see * 10;
                        }

                        // // #testing new see
                        // const is_bad: bool = !hce.see(pos, ex.move, 0);
                        // ex.is_bad_capture = is_bad;
                        // if (is_bad) {
                        //     ex.score = Scores.bad_capture - ex.piece.value() * 10; // testing minus
                        // }
                        // else {
                        //     ex.score = Scores.capture + ex.piece.value() * 10; // 10
                        // }

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
            else if (listmode == .quiets) {
                switch (ex.move.flags) {
                    Move.silent, Move.double_push, Move.castle_short, Move.castle_long => {
                        ex.score = hist.quiet.get_score(ex.*);

                        // #testing
                        // const see: bool = hce.see(pos, ex.move, 0);
                        // if (see) ex.score += 10 else ex.score -= 10;

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
            if (listmode == .noisies and extmoves[best_idx].is_bad_capture) {
                return null;
            }

            if (best_idx != idx) {
                std.mem.swap(ExtMove, &extmoves[best_idx], &extmoves[idx]);
            }

            return extmoves[idx];
        }

        fn select_slice(self: *Self, comptime listmode: ListMode) []ExtMove {
            return switch(listmode) {
                .noisies => self.noisies.slice(),
                .quiets => self.quiets.slice(),
                .bad_noisies => self.bad_noisies.slice(),
            };
        }
    };
}
