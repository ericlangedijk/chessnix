//! Hand Crafted Evaluation.

const std = @import("std");
const assert = std.debug.assert;

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const utils = @import("utils.zig");
const tt = @import("tt.zig");
const search = @import("search.zig");
const hcetables = @import("hcetables.zig");

const io = lib.io;

const wtf = lib.wtf;
const float = funcs.float;
const int = funcs.int;
const pop_square = funcs.pop_square;
const popcnt = funcs.popcnt;
const popcnt_v = funcs.popcnt_v;
const bit_loop = funcs.bit_loop;

const Value = types.Value;
const Float = types.Float;

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const GamePhase = types.GamePhase;
const Position = position.Position;
const Searcher = search.Searcher;
const ScorePair = hcetables.ScorePair;

const W = Color.WHITE;
const B = Color.BLACK;

const has_pawn_structure_cache: bool = false; // dummy.
const tracking: bool = true;

pub const Evaluator = struct {
    searcher: *const Searcher,
    pos: *const Position,
    has_pawns: bool,
    king_squares: [2]Square,
    king_areas: [2]u64,
    pawn_cover: [2]u64,
    knight_attacks: [2]u64,
    bishop_attacks: [2]u64,
    attack_power: [2]ScorePair,

    pub fn init(searcher: *const Searcher, pos: *const Position) Evaluator {
        return .{
            .searcher = searcher,
            .pos = pos,
            .has_pawns = false,
            .king_squares = .{ .A1, .A1 },
            .king_areas = .{ 0, 0 },
            .pawn_cover = .{ 0, 0 },
            .knight_attacks = .{ 0, 0 },
            .bishop_attacks = .{ 0, 0 },
            .attack_power = .{ .empty, .empty },
        };
    }


    // pub fn get_score(self: *const Evaluator) Value {
    // }

    pub fn evaluate(self: *Evaluator) Value {

        self.has_pawns = self.pos.all_pawns() != 0;

        self.king_squares = .{
            self.pos.king_square(W),
            self.pos.king_square(B)
        };

        self.king_areas[W.u] = bitboards.king_areas[self.king_squares[W.u].u];
        self.king_areas[B.u] = bitboards.king_areas[self.king_squares[B.u].u];

        self.pawn_cover = .{
            funcs.pawns_shift(self.pos.pawns(W), W, .northwest) | funcs.pawns_shift(self.pos.pawns(W), W, .northeast),
            funcs.pawns_shift(self.pos.pawns(B), B, .northwest) | funcs.pawns_shift(self.pos.pawns(B), B, .northeast),
        };

        var score: ScorePair = .empty;


        //score.mg = self.pos.values[W.u] - self.pos.values[B.u];
        //score.eg = self.pos.values[B.u] - self.pos.values[B.u];
        //io.debugprint("material {} {}\n", .{ score.mg, score.eg });


        score = score.add(self.eval_pawns(W)).sub(self.eval_pawns(B));
        score = score.add(self.eval_knights(W)).sub(self.eval_knights(B));
        score = score.add(self.eval_bishops(W)).sub(self.eval_bishops(B));
        score = score.add(self.eval_rooks(W)).sub(self.eval_rooks(B));


        // const int phase = std::min(state_.phase, kMaxPhase);
        // const Score mg_score = score_pair.MiddleGame();
        // const Score eg_score = score_pair.EndGame();
        // return (mg_score * phase + eg_score * (kMaxPhase - phase)) / kMaxPhase;

        const result: Value = sliding_score(self.pos.non_pawn_material(), score.mg, score.eg);


        io.debugprint("pairs {} {}\n", .{ score.mg, score.eg });
        io.debugprint("result {}\n", .{ result });
        return result;
        //return 0;
    }

    fn eval_pawns(self: *Evaluator, comptime us: Color) ScorePair {
        // https://lichess.org/editor/1k6/4p3/3pPp2/3P1p1p/2P5/2P5/P7/4K3_w_-_-_0_1?color=white

        if (has_pawn_structure_cache) return .empty; // dummy

        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        var score: ScorePair = .empty;
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);

        // Pawn phalanxes.
        var connected_pawns: u64 = funcs.pawns_shift(our_pawns, us, .northeast) & our_pawns;
        while (bit_loop(&connected_pawns)) |sq| {
            const sp: ScorePair = hcetables.pawn_phalanx_bonus[funcs.relative_rank(us, sq.coord.rank)];
            score.inc(sp);
            trace(sp, us, sq, "pawn phalanx", .{});
        }

        // TODO: KING BUCKET??

        var passed_pawns: u64 = 0;
        var pawns: u64 = our_pawns;
        while (bit_loop(&pawns)) |sq|{
            const sq_bb: u64 = sq.to_bitboard();
            const rank: u3 = funcs.relative_rank(us, sq.coord.rank);
            const file: u3 = sq.coord.file;

            // Defended pawn.
            if (self.pawn_cover[us.u] & sq_bb != 0) {
                const sp: ScorePair = hcetables.defended_pawn_bonus[funcs.relative_rank(us, rank)];
                score.inc(sp);
                trace(sp, us, sq, "defended pawn", .{});
            }

            // Doubled pawn.
            const pawns_ahead_on_file = funcs.forward_file(us, sq) & our_pawns;
            const is_doubled: bool = pawns_ahead_on_file != 0;
            if (pawns_ahead_on_file != 0) {
                const sp: ScorePair = hcetables.doubled_pawn_penalty[file];
                score.inc(sp);
                trace(sp, us, sq, "doubled pawn", .{});
            }

            // Passed pawn. (TODO: the original does not block our own pawns...)
            const their_pawns_ahead: u64 = bitboards.get_passed_pawn_mask(us, sq) & their_pawns;
            if (!is_doubled and their_pawns_ahead == 0) {
                passed_pawns |= sq_bb;
                const sp: ScorePair = hcetables.passed_pawn_bonus[funcs.relative_rank(us, rank)];
                score.inc(sp);
                trace(sp, us, sq, "passed pawn", .{});
            }


            // Isolated pawn.
            const pawns_on_adjacent_files = bitboards.adjacent_file_masks[sq.u] & our_pawns;
            if (pawns_on_adjacent_files == 0) {
                const sp: ScorePair = hcetables.isolated_pawn_penalty[file];
                score.inc(sp);
                trace(sp, us, sq, "isolated pawn", .{});
            }
        }

        // TODO: AddPawnPSQT

        // Pawn king race.
        const king_sq: Square = self.king_squares[us.u];
        const their_king_sq: Square = self.king_squares[them.u];
        while (bit_loop(&passed_pawns)) |sq| {
            const their_move: u1 = @intFromBool(pos.stm.e == them.e);
            const relative_rank: u3 = funcs.relative_rank(us, sq.coord.rank);
            const dist_to_king: u3 = funcs.square_distance(sq, king_sq);
            const sp1: ScorePair = hcetables.king_pp_distance_table[dist_to_king];
            score.inc(sp1);
            trace(sp1, us, sq, "king dist to pawn", .{});

            const dist_to_enemy_king: u3 = funcs.square_distance(sq, their_king_sq);
            const sp2: ScorePair = hcetables.enemy_king_pp_distance_table[dist_to_enemy_king];
            score.inc(sp2);
            trace(sp2, us, sq, "enemy king dist to pawn", .{});

            // Square rule for passed pawns.
            const enemy_non_pawn_king_pieces: u64 = pos.by_color(them) & ~pos.kings(them) & ~pos.pawns(them);
            const dist_to_promotion: u3 = (7 - relative_rank); // - if (rel_rank == 2) 1 else 0;
            //if (relative_rank == bitboards.rank_2) dist_to_promotion -= 1; // (TODO: the original does not take this into account...)
            //io.debugprint("{t} dist to promotion = {} dist to pawn = {} \n", .{ sq.e, dist_to_promotion, dist_to_enemy_king });

            if (enemy_non_pawn_king_pieces == 0 and dist_to_promotion < dist_to_enemy_king - their_move) {
                score.inc(hcetables.king_cannot_reach_pp_bonus);
                trace(hcetables.king_cannot_reach_pp_bonus, us, sq, "enemy king cannot reach pawn", .{});
            }
        }
        return score;
    }

    fn eval_knights(self: *Evaluator, comptime us: Color) ScorePair {
        // 2k5/8/2p3p1/1n6/5n2/4N3/2PP4/2K5 b - - 0 1
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        var score: ScorePair = .empty;
        var our_knights: u64 = pos.knights(us);

        while (bit_loop(&our_knights)) |sq| {
            const relative_square: Square = sq.relative(us);

            // Piece square score.
            if (!self.has_pawns) {
                const sp: ScorePair = hcetables.normal_piece_square_table[PieceType.KNIGHT.u][relative_square.u];
                score.inc(sp);
            }

            // Mobility.
            const legal_moves: u64 = self.legalize_moves(PieceType.KNIGHT, us, sq, attacks.get_knight_attacks(sq) & ~pos.by_color(us));
            const mobility: u64 = legal_moves & ~self.pawn_cover[them.u];
            const cnt: u7 = popcnt(mobility);
            score.inc(hcetables.knight_mobility_table[cnt]);
            trace(hcetables.knight_mobility_table[cnt], us, sq, "knight mobility, cnt = {}", .{ cnt });

            // Update.
            self.knight_attacks[us.u] |= legal_moves;

            // King attack.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u]; // TODO: checkout if we also could use legalmoves instead of mobility here.
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                const sp: ScorePair = hcetables.attack_power[PieceType.KNIGHT.u][king_attack_count];
                self.attack_power[us.u].inc(sp);
                trace(sp, us, sq, "attackpower kingarea knight, king_attack_count = {}", .{ king_attack_count });
            }

            // Outpost
            if (self.is_outpost(sq, us)) {
                score.inc(hcetables.knight_outpost_table[relative_square.u]);
                trace(hcetables.knight_outpost_table[relative_square.u], us, sq, "knight outpost", .{});
            }
        }
        return score;
    }

    fn eval_bishops(self: *Evaluator, comptime us: Color) ScorePair {
        // 2k5/8/2p2bp1/3b4/8/2PBB3/1P1P4/2K5 w - - 0 1
        var score: ScorePair = .empty;
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        var our_bishops: u64 = pos.bishops(us);
        const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.bishops(us); // TODO: why ???
        var sp: ScorePair = undefined;

        // Bishop pair.
        if ((our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
            sp = hcetables.bishop_pair_bonus;
            score.inc(sp);
            trace(sp, us, null, "bishoppair", .{});
        }

        // TODO: kingbucket

        while (bit_loop(&our_bishops)) |sq| {
            const relative_square: Square = sq.relative(us);

            // Piece square score.
            if (!self.has_pawns) {
                sp = hcetables.normal_piece_square_table[PieceType.BISHOP.u][relative_square.u];
                score.inc(sp);
            }

            // Mobility
            const moves: u64 = self.legalize_moves(PieceType.BISHOP, us, sq, attacks.get_bishop_attacks(sq, occ) & ~pos.by_color(us));
            const mobility: u64 = moves & ~self.pawn_cover[them.u];
            const cnt: u7 = popcnt(mobility);
            sp = hcetables.bishop_mobility_table[cnt];
            score.inc(sp);
            trace(sp, us, sq, "bishop mobility, cnt = {}", .{ cnt });

            // Update
            self.bishop_attacks[us.u] |= moves;

            // Outpost
            if (self.is_outpost(sq, us)) {
                sp = hcetables.bishop_outpost_table[relative_square.u];
                score.inc(sp);
                trace(sp, us, sq, "bishop outpost", .{});
            }
        }
        return score;
    }

    fn eval_rooks(self: *Evaluator, comptime us: Color) ScorePair {
        var score: ScorePair = .empty;
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.rooks(us);// TDOO: why ???
        var sp: ScorePair = undefined;
        var our_rooks: u64 = pos.rooks(us);
        while (bit_loop(&our_rooks)) |sq| {
            const moves: u64 = self.legalize_moves(PieceType.ROOK, us, sq, attacks.get_rook_attacks(sq, occ));
            const mobility: u64 = moves & ~self.pawn_cover[them.u];
            const cnt: u7 = popcnt(mobility);
            sp = hcetables.rook_mobility_table[cnt];
            score.inc(sp);
            trace(sp, us, sq, "rook mobility, cnt = {}", .{ cnt });
        }
        return score;
    }

    fn eval_queens(self: *Evaluator, comptime us: Color) ScorePair {
        _ = self;
        _ = us;
        return .empty;
    }

    fn eval_king(self: *Evaluator, comptime us: Color) ScorePair {
        _ = self;
        _ = us;
        return .empty;
    }

    fn is_outpost(self: *const Evaluator, sq: Square, comptime us: Color) bool {
        const them: Color = comptime us.opp();
        const sq_bb: u64 = sq.to_bitboard();
        const safe: bool = self.pawn_cover[us.u] & sq_bb != 0 and self.pawn_cover[them.u] & sq_bb == 0; // TODO: maybe smarter. for example enemy pawn in front?
        return safe and funcs.outpost(us) & sq_bb != 0;
    }

    fn legalize_moves(self: *const Evaluator, comptime pt: PieceType, comptime us: Color, from: Square, bb_moves: u64) u64 {
        const from_bb: u64 = from.to_bitboard();

        // TODO: checks??

        if (from_bb & self.pos.pins == 0) {
            return bb_moves;
        }

        switch (pt.e) {
            .pawn => {
                return bb_moves;
            },
            .knight => {
                return 0;
            },
            .bishop, .rook, .queen => {
                return bitboards.get_squarepair(self.king_squares[us.u], from).ray & bb_moves; // king_sq first!
            },
            // .rook => {
            //     return bb_moves;  // TODO: write code
            // },
            // .queen => {
            //     return bb_moves;  // TODO: write code
            // },
            .king => {
                return bb_moves;
            },
        }
    }

    pub fn sliding_score(non_pawn_material: Value, opening: Value, endgame: Value) Value {
        const max: i32 = comptime types.max_material_without_pawns;
        const phase: i32 = @min(non_pawn_material, max);
        return @truncate(@divTrunc(opening * phase + endgame * (max - phase), max));
    }

    fn trace(sp: ScorePair, comptime us: Color, sq: ?Square, comptime msg: []const u8, args: anytype) void {
        if (sq) |q| {
            io.debugprint("{t} {t} mg {} eg {} ", .{ us.e, q.e, sp.mg, sp.eg });
        }
        else {
            io.debugprint("{t} mg {} eg {} ", .{ us.e, sp.mg, sp.eg });
        }

        io.debugprint(msg, args);
        io.debugprint("\n", .{});
    }
};

