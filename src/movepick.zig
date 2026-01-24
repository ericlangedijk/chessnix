// zig fmt: off

///! Staged move picker.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const history = @import("history.zig");
const hce = @import("hce.zig");
const hcetables = @import("hcetables.zig");

const assert = std.debug.assert;

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Move = types.Move;
const ScorePair = types.ScorePair;
const Position = position.Position;
const ExtMove = search.ExtMove;
const Node = search.Node;
const Searcher = search.Searcher;
const ExtMoveList = search.ExtMoveList;
const History = history.History;

const use_countermoves: bool = false; // if no result we revert back to 28.

/// Depending on the search callsite we generate all moves or quiescence moves.
pub const GenType = enum {
    search,
    quiescence,
};

// The current stage of the movepicker.
pub const Stage = enum {
    /// Generate all moves, putting them in the correct list.
    generate,
    /// The first move to consider is a tt-move.
    tt,
    /// Score the noisy moves. Also updata the bad noisy list.
    score_noisy,
    /// Extract the noisy moves.
    noisy,
    /// Score the quiet moves.
    score_quiet,
    /// Extract the quiet moves.
    quiet,
    /// Extract the bad noisy moves.
    bad_noisy,
};

/// Depending on the stage we have to select a list.
const ListMode = enum {
    noisies,
    quiets,
    bad_noisies,
};

const Scores = struct {
    const promotion    : Value = 2_000_000;
    const capture      : Value = 1_000_000;
    const bad_capture  : Value = -1_000_000;
};

/// There is no staged move generation (which is much slower). Instead we put the moves in the correct list.
/// Bad noisy moves are added during the scoring of noisy moves. Extracting uses some tricky scheme.
pub fn MovePicker(comptime gentype: GenType, comptime us: Color) type {

    return struct {
        const Self = @This();
        const them: Color = us.opp();
        stage: Stage,
        /// Reference to the position (on the stack during search).
        pos: *const Position,
        /// A backlink to the searcher to access history and nodes.
        searcher: *const Searcher,
        /// The current node in the search.
        node: *const Node,
        /// The init parameter.
        input_tt_move: Move,
        /// Filled during move generation.
        tt_move: ?ExtMove,
        /// Only valid after the generate stage.
        move_count: u8,
        /// Readonly. Used for looping through the current atage-moves.
        move_idx: u8,
        /// Noisy move list.
        noisies: ExtMoveList(80),
        /// Quiet move list.
        quiets: ExtMoveList(224),
        /// Bad capture list. TODO: maybe directly fill this one during store if possible.
        bad_noisies: ExtMoveList(80),

        pub fn init(pos: *const Position, searcher: *const Searcher, node: *const Node, tt_move: Move) Self {
            return .{
                .stage = .generate,
                .pos = pos,
                .searcher = searcher,
                .node = node,
                .input_tt_move = tt_move,
                .tt_move = null,
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
                    self.stage = .score_quiet;
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

        /// Required function for movgen.
        pub fn store(self: *Self, move: Move) ?void {
            self.move_count += 1;
            var ex: ExtMove = init_move(move, self.pos);
            if (move == self.input_tt_move) {
                ex.is_tt_move = true;
                self.tt_move = ex;
            }
            else if (move.is_noisy()) {
                self.noisies.add(ex);
            }
            else {
                self.quiets.add(ex);
            }
        }

        fn init_move(move: Move, pos: *const Position) ExtMove {
            var ex: ExtMove = .init(move);
            ex.piece = pos.board[ex.move.from.u];
            switch (ex.move.flags) {
                Move.capture, Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                    ex.captured = pos.board[ex.move.to.u];
                },
                Move.ep => {
                    ex.captured = Piece.create(PieceType.PAWN, them);
                },
                else => {
                    // Nothing to do.
                }
            }
            return ex;
        }

        /// Called during search.
        /// - NOTE: Because we have no killer moves - and the stage can be set too early - we must have seen at least one quiet move. I still have to experiment with this. #testing.
        pub fn skip_quiets(self: *Self) void {
            if (self.stage == .quiet and self.move_idx > 0) {
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
                self.score_move(ex, listmode);
            }
        }

        /// Besides scoring the move set the details.
        fn score_move(self: *Self, ex: *ExtMove, comptime listmode: ListMode) void {

            // Some paranoid checks.
            if (comptime lib.is_paranoid) {
                assert(listmode != .bad_noisies);
                if (listmode == .noisies) { assert(ex.move.is_noisy()); }
                else if (listmode == .quiets) { assert(ex.move.is_quiet()); }
            }

            const pos: *const Position = self.pos;
            const hist: *const History = &self.searcher.hist;

            // Score a noisy move
            if (listmode == .noisies) {
                switch (ex.move.flags) {
                    Move.capture => {
                        ex.is_bad_capture = !hce.see(pos, ex.move, 0);
                        if (ex.is_bad_capture) {
                            ex.score = Scores.bad_capture + ex.captured.value() * 100 - ex.piece.value();
                        }
                        else {
                            ex.score = Scores.capture + ex.captured.value() * 100 - ex.piece.value();
                        }
                        ex.score += hist.capture.get_score(ex.*);

                        // Right here, when scoring is complete, we add the move to the bad noisy moves.
                        if (ex.is_bad_capture) {
                            self.bad_noisies.add(ex.*);
                        }
                    },
                    Move.ep => {
                        ex.score = Scores.capture + ex.captured.value() * 100 - ex.piece.value();
                        ex.score += hist.capture.get_score(ex.*);
                    },
                    Move.knight_promotion, Move.bishop_promotion, Move.rook_promotion, Move.queen_promotion => {
                        const prom: PieceType = ex.move.promoted_to();
                        switch (prom.e) {
                            .queen => ex.score = Scores.promotion + 2000,
                            .knight => ex.score = Scores.promotion + 1000,
                            .bishop, .rook => ex.score = Scores.promotion,
                            else => unreachable,
                        }
                    },
                    Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                        const prom: PieceType = ex.move.promoted_to();
                        switch (prom.e) {
                            .queen => ex.score = Scores.promotion + 2000,
                            .knight => ex.score = Scores.promotion + 1000,
                            .bishop, .rook => ex.score = Scores.promotion,
                            else => unreachable,
                        }
                        //ex.score += 10;
                        ex.score += hist.capture.get_score(ex.*); // #testing
                    },
                    else => {
                        unreachable;
                    }
                }
            }
            // Score a quiet move.
            else if (listmode == .quiets) {

                ex.score = hist.get_quiet_score(ex.*, self.node.ply, &self.searcher.nodes);
                //ex.score += self.from_to_score(ex.*); // #testing #experimental.
                // #testing: we do not need the switch.

                // ex.score = hist.quiet.get_score(ex.*);
                // ex.score += hist.continuation.get_score(ex.*, self.node.ply, &self.searcher.nodes);
                // switch (ex.move.flags) {
                //     Move.silent, Move.double_push, Move.castle_short, Move.castle_long => {
                //         ex.score = hist.quiet.get_score(ex.*);
                //         ex.score += hist.continuation.get_score(ex.*, self.node.ply, &self.searcher.nodes);
                //     },
                //     else => {
                //         unreachable;
                //     },
                // }
            }
            // Bad noisy moves are already handled.
            else {
                unreachable;
            }
        }

        fn from_to_score(self: *Self, ex: ExtMove) Value {
            const pt: u4 = ex.piece.piecetype().u;
            const from_sq: Square = ex.move.from.relative(comptime us);
            const to_sq: Square = ex.move.from.relative(comptime us);
            const from_score: ScorePair = hcetables.piece_square_table[pt][from_sq.u];
            const to_score  : ScorePair = hcetables.piece_square_table[pt][to_sq.u];
            const delta_score: ScorePair = to_score.sub(from_score);
            return hce.sliding_score(self.pos, delta_score);
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
