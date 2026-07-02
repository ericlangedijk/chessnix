// zig fmt: off

///! Staged move picker.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const history = @import("history.zig");
const hce = @import("hce.zig");
//const hcetables = @import("hcetables.zig");

const assert = std.debug.assert;

const Color = types.Color;
const Square = types.Square;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Move = types.Move;
const ExtMove = types.ExtMove;
const ExtMoveList = types.ExtMoveList;
const ScorePair = types.ScorePair;
const Position = position.Position;
const Node = search.Node;
const Searcher = search.Searcher;
const History = history.History;

/// Depending on the search callsite we generate all moves or quiescence moves.
pub const GenType = enum {
    search,
    quiescence,
};

// The current internal stage of the movepicker.
pub const InternalStage = enum {
    /// Generate all moves, putting them in the correct list. (fallthrough).
    generate,
    /// The first move to consider is a tt-move.
    tt,
    /// Score the noisy moves. Also updata the bad noisy list. (fallthrough).
    score_noisy,
    /// Extract the noisy moves.
    noisy,
    /// Score the quiet moves. (fallthrough).
    score_quiet,
    /// Extract the quiet moves.
    quiet,
    /// Extract the bad noisy moves.
    bad_noisy,
};

/// Only valid after extracting a move.
pub const Stage = enum {
    tt,
    noisy,
    quiet,
    bad_noisy,
};

/// Depending on the stage we have to select a list to pick from.
const ListMode = enum {
    noisies,
    quiets,
    bad_noisies,
};

pub const Scores = struct {
    pub const promotion       : i32 =  2_000_000;
    pub const capture         : i32 =  1_000_000;
    pub const bad_capture     : i32 = -1_000_000;
    pub const bad_capture_max : i32 = -900.000;
};

/// There is no real staged move generation. This is a hybrid solution.
/// We generate all moves in one go and put them in the correct list.
/// Bad noisy moves are added during the scoring of noisy moves. Extracting uses some tricky scheme.
pub fn MovePicker(comptime gentype: GenType, comptime us: Color) type {

    return struct {
        const Self = @This();
        const them: Color = us.opp();

        // This saves some stack space during quiescence search.
        const max_quiets: u8 = if (gentype == .search) 224 else 128;
        const max_noisies = types.max_noisy_count;

        /// The current private stage. **Not** usable for the caller.
        internal_stage: InternalStage,
        /// Only valid when extracting moves. It reflects the type of move we last extracted.
        stage: Stage,
        /// Reference to the position (on the stack during search).
        pos: *const Position,
        /// Reference to the searcher to access history and nodes.
        searcher: *const Searcher,
        /// Reference to the current node in the searcher.
        node: *const Node,
        /// The init parameter.
        input_tt_move: Move,
        /// Filled during move generation.
        tt_move: ExtMove,
        /// Only valid after move generation.
        move_count: u8,
        /// Used for looping through the currently active move list.
        internal_move_idx: u8,
        /// Noisy move list.
        noisies: ExtMoveList(max_noisies),
        /// Quiet move list.
        quiets: ExtMoveList(max_quiets),
        /// Bad capture list.
        bad_noisies: ExtMoveList(max_noisies),

        pub fn init(pos: *const Position, searcher: *const Searcher, node: *const Node, tt_move: Move) Self {
            return .{
                .internal_stage = .generate,
                .stage = .tt,
                .pos = pos,
                .searcher = searcher,
                .node = node,
                .input_tt_move = tt_move,
                .tt_move = .empty,
                .move_count = 0,
                .internal_move_idx = 0,
                .noisies = .init(),
                .quiets = .init(),
                .bad_noisies = .init(),
            };
        }

        /// Required function for movegen.
        pub fn reset(self: *Self) void {
            self.move_count = 0;
            self.internal_move_idx = 0;
            self.noisies.count = 0;
            self.quiets.count = 0;
            self.bad_noisies.count = 0;
        }

        /// Required function for movegen.
        pub fn store(self: *Self, extmove: ExtMove) void {
            self.move_count += 1;
            // Copy
            if (extmove.move == self.input_tt_move) {
                self.tt_move = extmove;
            }
            else if (extmove.move.is_noisy()) {
                self.noisies.add(extmove);
            }
            else {
                self.quiets.add(extmove);
            }
        }

        pub fn next(self: *Self) ?ExtMove {
            const st: InternalStage = self.internal_stage;
            sw: switch (st) {
                .generate => {
                    self.internal_stage = .tt;
                    self.generate_moves();
                    continue :sw .tt;
                },
                .tt => {
                    self.stage = .tt;
                    self.internal_stage = .score_noisy;
                    if (self.tt_move != ExtMove.empty) {
                        return self.tt_move;
                    }
                    continue :sw .score_noisy;
                },
                .score_noisy => {
                    self.internal_stage = .noisy;
                    self.internal_move_idx = 0;
                    self.score_moves(.noisies);
                    continue :sw .noisy;
                },
                .noisy => {
                    self.stage = .noisy;
                    if (self.extract_next(self.internal_move_idx, .noisies)) |ex| {
                        self.internal_move_idx += 1;
                        return ex;
                    }
                    self.internal_stage = .score_quiet;
                    continue :sw .score_quiet;
                },
                .score_quiet => {
                    self.internal_stage = .quiet;
                    self.internal_move_idx = 0;
                    self.score_moves(.quiets);
                    continue :sw .quiet;
                },
                .quiet => {
                    self.stage = .quiet;
                    if (self.extract_next(self.internal_move_idx, .quiets)) |ex| {
                        self.internal_move_idx += 1;
                        return ex;
                    }
                    self.internal_stage = .bad_noisy;
                    self.internal_move_idx = 0;
                    continue :sw .bad_noisy;
                },
                .bad_noisy => {
                    self.stage = .bad_noisy;
                    if (self.extract_next(self.internal_move_idx, .bad_noisies)) |ex| {
                        self.internal_move_idx += 1;
                        return ex;
                    }
                    return null;
                },
            }
        }

        /// Called during search.
        pub fn skip_quiets(self: *Self) void {
            if (self.internal_stage == .quiet) {
                self.internal_stage = .bad_noisy;
                self.internal_move_idx = 0;
            }
        }

        fn generate_moves(self: *Self) void {
            switch (gentype) {
                .search => movegen.generate_all_moves(self.pos, us, self),
                .quiescence => movegen.generate_quiescence_moves(self.pos, us, self),
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

        /// Set the score.
        fn score_move(self: *Self, ex: *ExtMove, comptime listmode: ListMode) void {
            if (comptime lib.is_paranoid) {
                assert(listmode != .bad_noisies);
                if (listmode == .noisies) { assert(ex.move.is_noisy()); }
                else if (listmode == .quiets) { assert(ex.move.is_quiet()); }
            }

            const pos: *const Position = self.pos;
            const hist: *const History = &self.searcher.hist;

            // TODO: we could have a 'gen_order' here.
            ex.score = 0;

            // Score a noisy move
            if (listmode == .noisies) {

                switch (ex.move.kind) {
                    Move.capture => {
                        const is_bad_capture: bool = !hce.see(pos, ex.move, 0);
                        if (is_bad_capture) {
                            ex.score += Scores.bad_capture + ex.captured.value() * 100 - ex.piece.value() >> 3;
                        }
                        else {
                            ex.score += Scores.capture + ex.captured.value() * 100 - ex.piece.value() >> 3;
                        }
                        ex.score += hist.get_capture_score(ex.*);

                        // Right here, when scoring is complete, we add a bad capture move to the bad noisy moves.
                        if (is_bad_capture) {
                            self.bad_noisies.add(ex.*);
                        }
                    },
                    Move.ep => {
                        ex.score += Scores.capture + ex.captured.value() * 101 - ex.piece.value() >> 3;
                        ex.score += hist.get_capture_score(ex.*);
                    },
                    Move.knight_promotion, Move.bishop_promotion, Move.rook_promotion, Move.queen_promotion => {
                        const prom: PieceType = ex.move.prom();
                        switch (prom.e) {
                            .queen => ex.score += Scores.promotion + 2000,
                            .knight => ex.score += Scores.promotion + 1000,
                            .bishop, .rook => ex.score += Scores.promotion,
                            else => unreachable,
                        }
                    },
                    Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                        const prom: PieceType = ex.move.prom();
                        switch (prom.e) {
                            .queen => ex.score += Scores.promotion + 2000,
                            .knight => ex.score += Scores.promotion + 1000,
                            .bishop, .rook => ex.score += Scores.promotion,
                            else => unreachable,
                        }
                        ex.score += hist.get_capture_score(ex.*);
                    },
                    else => {
                        unreachable;
                    }
                }
            }
            // Score a quiet move.
            else if (listmode == .quiets) {
                ex.score += hist.get_quiet_score(ex.*, self.node.ply, &self.searcher.nodes);
                if (ex.move.is_castle()) {
                    ex.score += 1;
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
            var max_score: i32 = extmoves[idx].score;

            for (idx + 1..extmoves.len) |i| {
                const score: i32 = extmoves[i].score;
                if (score > max_score) {
                    max_score = score;
                    best_idx = @intCast(i);
                }
            }

            // If the best noisy score is a bad capture, we are done and skip to the next stage (quiets).
            if (listmode == .noisies and extmoves[best_idx].score <= Scores.bad_capture_max) {
                return null;
            }

            // Selection sort.
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
