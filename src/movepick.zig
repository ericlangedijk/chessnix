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

const Score = types.Score;
const SmallScore = types.SmallScore;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const PieceType = types.PieceType;
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

/// Depending on the stage we have to select a list to pick from.
const ListMode = enum {
    noisies,
    quiets,
    bad_noisies,
};

const Scores = struct {
    const promotion    : Score = 2_000_000;
    const capture      : Score = 1_000_000;
    const bad_capture  : Score = -1_000_000;
};

/// A tiny shallow move ordering center bias.
const move_ordering_square_bias: [64]u3 = .{
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,3,3,1,0,0,
    0,0,1,3,3,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
};

/// There is no staged move generation (which is much slower).
/// Instead we generate all moves and put them in the correct list.
/// Bad noisy moves are added during the scoring of noisy moves. Extracting uses some tricky scheme.
pub fn MovePicker(comptime gentype: GenType, comptime us: Color) type {

    return struct {
        const Self = @This();
        const them: Color = us.opp();

        // This saves some stack space during quiescence search.
        const max_quiets: u8 = if (gentype == .search) 224 else 128;

        /// The current stage.
        /// TODO: make the stages cleaner, reflecting the actual current atage without these half baked score nonsense in between.
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
        noisies: ExtMoveList(128),
        /// Quiet move list.
        quiets: ExtMoveList(max_quiets),
        /// Bad capture list.
        bad_noisies: ExtMoveList(128),

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

        /// Required function for movegen.
        pub fn reset(self: *Self) void {
            self.move_count = 0;
            self.move_idx = 0;
            self.noisies.count = 0;
            self.quiets.count = 0;
            self.bad_noisies.count = 0;
        }

        /// Required function for movegen.
        pub fn store(self: *Self, extmove: ExtMove) ?void {
            self.move_count += 1;
            // Copy
            var ex: ExtMove = extmove;
            if (extmove.move == self.input_tt_move) {
                ex.is_tt_move = true;
                self.tt_move = ex;
            }
            else if (extmove.move.is_noisy()) {
                self.noisies.add(ex);
            }
            else {
                self.quiets.add(ex);
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
                self.score_move(ex, listmode);
            }
        }

        /// Set the score and is_bad_capture flag.
        fn score_move(self: *Self, ex: *ExtMove, comptime listmode: ListMode) void {
            // The idea for capture scores is MVV-LVA scoring. When history kicked in that will overwrite the LVA part.

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
                            ex.score = Scores.bad_capture + ex.captured.value() * 100 - ex.piece.value() >> 3;
                        }
                        else {
                            ex.score = Scores.capture + ex.captured.value() * 100 - ex.piece.value() >> 3;
                        }
                        ex.score += hist.get_capture_score(ex.*);

                        // Right here, when scoring is complete, we add the move to the bad noisy moves.
                        if (ex.is_bad_capture) {
                            self.bad_noisies.add(ex.*);
                        }
                    },
                    Move.ep => {
                        ex.score = Scores.capture + ex.captured.value() * 100 - ex.piece.value() >> 3;
                        ex.score += hist.get_capture_score(ex.*);
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
                        ex.score += hist.get_capture_score(ex.*);
                    },
                    else => {
                        unreachable;
                    }
                }
            }
            // Score a quiet move.
            else if (listmode == .quiets) {
                ex.score = hist.get_quiet_score(ex.*, self.node.ply, &self.searcher.nodes);
                ex.score += move_ordering_square_bias[ex.move.to.u];
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
            var max_score: Score = extmoves[idx].score;

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
