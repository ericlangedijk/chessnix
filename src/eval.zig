// zig fmt: off

//! Static evaluation.

const std = @import("std");

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const tt = @import("tt.zig");

const assert = std.debug.assert;
const io = lib.io;
const wtf = lib.wtf;
const float = funcs.float;
const int = funcs.int;
const pop_square = funcs.pop_square;
const popcnt = funcs.popcnt;
const popcnt_v = funcs.popcnt_v;

const Value = types.Value;
const Float = types.Float;

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const GamePhase = types.GamePhase;
const Position = position.Position;

const P: PieceType = types.P;
const N: PieceType = types.N;
const B: PieceType = types.B;
const R: PieceType = types.R;
const Q: PieceType = types.Q;
const K: PieceType = types.K;

/// Returns the evaluation from white perspective, with detailed debug info.
pub fn evaluate_with_tracking_absolute(pos: *const Position, tt_eval: *tt.EvalTranspositionTable, tt_pawns: *tt.PawnTranspositionTable) Value {
    const non_pawn_material = pos.non_pawn_material();
    const phase: GamePhase = Position.phase_of(non_pawn_material);
    const v: Value = switch (phase) {
        .Opening => eval(pos, non_pawn_material, tt_eval, tt_pawns, .Opening, true),
        .Midgame => eval(pos, non_pawn_material, tt_eval, tt_pawns, .Midgame, true),
        .Endgame => eval(pos, non_pawn_material, tt_eval, tt_pawns, .Endgame, true),
    };
    return if (pos.stm.e == .white) v else -v;
}

pub fn evaluate(pos: *const Position, tt_eval: *tt.EvalTranspositionTable, tt_pawns: *tt.PawnTranspositionTable) Value {
    const non_pawn_material = pos.non_pawn_material();
    const phase: GamePhase = Position.phase_of(non_pawn_material);
    switch (phase) {
        .Opening => return eval(pos, non_pawn_material, tt_eval, tt_pawns, .Opening, false),
        .Midgame => return eval(pos, non_pawn_material, tt_eval, tt_pawns, .Midgame, false),
        .Endgame => return eval(pos, non_pawn_material, tt_eval, tt_pawns, .Endgame, false),
    }
}

pub fn simple_eval(pos: *const Position, comptime us: Color) Value {
    return switch (us.e) {
        .white => pos.values[Color.WHITE.u] - pos.values[Color.BLACK.u],
        .black => pos.values[Color.BLACK.u] - pos.values[Color.WHITE.u],
    };
}

fn pesto_add(mg: *Value, eg: *Value, delta: Value) void {
    //_ = mg; _ = eg; _ = delta;
    mg.* += delta;
    eg.* += delta;
}

fn add(v: *Value, delta: Value, comptime track: bool, comptime msg: []const u8, sq: ?Square) void {
    v.* += delta;
    if (lib.is_debug and track and msg.len > 0) {
        if (sq) |q|
            io.debugprint("{t} {s} {}\n", .{ q.e, msg, delta })
        else
            io.debugprint("    {s} {}\n", .{ msg, delta });
    }
}

const PASSED_PAWN_BY_RANK: [8]Value = .{ 0, 2, 4, 6, 8, 16, 20, 0 }; // .{ 0, 15, 20, 32, 56, 92, 140, 0 }; // most is handled by pesto

/// Evaluation from the perspective of the side to move.
fn eval_fucked(pos: *const Position, non_pawn_material: Value, tt_eval: *tt.EvalTranspositionTable, tt_pawns: *tt.PawnTranspositionTable, comptime phase: GamePhase, comptime tracking: bool) Value {

    if (tt_eval.probe(pos.key)) |e| {
        if (lib.is_release) return e;
    }

    //const pawn_score = tt_pawns.probe(pos.pawn)
    //_ = tt_eval;
    _ = tt_pawns;

    const bb_all = pos.all();
    const negate: bool = pos.stm.e == .black;
    const is_queenless: bool = pos.all_queens() == 0;
    const has_pawns: bool = pos.all_pawns() != 0;

    // EXPERIMENTAL: pawn up in pawn endgame is often desastrous.
    // const pawns_add: Value = if (non_pawn_material == 0) 30 else 0;
    const material_scores: [2]Value = .{ pos.values[Color.WHITE.u], pos.values[Color.BLACK.u]};

    // Sliding values.
    var pesto_scores_mg: [2]Value = .{ 0, 0 };
    var pesto_scores_eg: [2]Value = .{ 0, 0 };
    var pawn_scores_mg: [2]Value = .{ 0, 0 };
    var pawn_scores_eg: [2]Value = .{ 0, 0 };
    var mobility_scores_mg: [2]Value = .{ 0, 0 };
    var mobility_scores_eg: [2]Value = .{ 0, 0 };
    var strategic_scores_mg: [2]Value = .{ 0, 0 };
    var strategic_scores_eg: [2]Value = .{ 0, 0 };
    // Scores
    var bonus_scores: [2]Value = .{ 0, 0 };
    var king_attacker_count: [2]Value = .{ 0, 0 };



    // We need pawn hits multiple times.
    const pawn_cover: [2]u64 = .{
        if (!has_pawns) 0 else funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northwest) | funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northeast),
        if (!has_pawns) 0 else funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northwest) | funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northeast),
    };

    ////////////////////////////////////////////////////////////////
    // Color loop.
    ////////////////////////////////////////////////////////////////
    inline for (Color.all) |us| {

        const them = comptime us.opp();
        const bb_us = pos.by_color(us);
        const our_king_sq = pos.king_square(us);
        const their_king_sq = pos.king_square(them);
        const our_king_area = bitboards.king_areas[our_king_sq.u];
        const their_king_area = bitboards.king_areas[their_king_sq.u];

        // WHAT IS THE BUG.
        // if (phase == .Opening) {
        //         //funcs.print_bitboard(pos.non_pawns(us));
        //         //funcs.print_bitboard(funcs.relative_rank_bb(us, bitboards.rank_2));
        //     const backrow: u64 = funcs.relative_rank_bb(us, bitboards.rank_1) & pos.non_pawns(us);
        //     const pawnrow: u64 = funcs.relative_rank_bb(us, bitboards.rank_2) & pos.pawns(us);
        //     //const cnt: Value = popcnt_v(pawnrow);
        //     if (popcnt(pawnrow) == 8 and popcnt(backrow) < 8)
        //         bonus.* -= 20;
        //         io.debugprint("WTF {t} {} {} \n", .{us.e, popcnt(pawnrow), popcnt(backrow)});
        //         funcs.print_bitboard(pawnrow);
        //         funcs.print_bitboard(backrow);
        // }

        ////////////////////////////////////////////////////////////////
        // Piece loop.
        ////////////////////////////////////////////////////////////////
        inline for (PieceType.all) |piecetype| {
            var is_first_piece: bool = true;
            const bb_pieces = pos.pieces(piecetype, us);
            var bb_running = bb_pieces;

            ////////////////////////////////////////////////////////////////
            // Square loop.
            ////////////////////////////////////////////////////////////////
            while (bb_running != 0) : (is_first_piece = false) {
                const sq: Square = funcs.pop_square(&bb_running);
                const bb_this_piece = bb_pieces & sq.to_bitboard();
                const file: u3 = sq.file();
                const rank: u3 = sq.rank();
                const relative_rank: u3 = funcs.relative_rank(us, rank);

                // pesto.
                const pair = Tables.get_scorepair(us, piecetype, sq);
                add(&pesto_scores_mg[us.u], pair.mg, tracking, "pesto", sq);
                add(&pesto_scores_eg[us.u], pair.eg, tracking, "pesto", sq);

                ////////////////////////////////////////////////////////////////
                // Piece loop.
                ////////////////////////////////////////////////////////////////
                switch (piecetype.e) {
                    .pawn => {
                        const our_pawns: u64 = bb_pieces;
                        const pawn_attacks: u64 = attacks.get_pawn_attacks(sq, us);
                        const pawn_push: u64 = funcs.pawns_shift(bb_this_piece, us, funcs.PawnShift.up) & ~bb_all;

                        // King area threat.
                        const att_king_area: u64 = (pawn_attacks | pawn_push) & their_king_area;
                        if (att_king_area != 0) {
                            // NOTE: Do not add to king attackers. Not good
                            add(&pawn_scores_mg[us.u], 10, tracking, "pawn king threat mg", sq);
                            add(&pawn_scores_eg[us.u],  4, tracking, "pawn king threat eg", sq);
                        }

                        // Passed pawn. TODO: majority + candidate passed pawn
                        const is_passed: bool = is_passed_pawn(pos, us, sq);
                        if (is_passed) {
                            //ps += funcs.relative_rank(us, sq.rank());
                            add(&pawn_scores_mg[us.u], PASSED_PAWN_BY_RANK[relative_rank], tracking, "passed pawn mg", sq);
                            add(&pawn_scores_eg[us.u], PASSED_PAWN_BY_RANK[relative_rank], tracking, "passed pawn eg", sq);
                        }

                        // Doubled pawn.
                        const bb_doubled = bitboards.file_bitboards[file] & our_pawns;
                        const is_doubled: bool = popcnt(bb_doubled) > 1;
                        if (is_doubled) {
                            add(&pawn_scores_mg[us.u], -5, tracking, "doubled pawn mg", sq);
                            add(&pawn_scores_eg[us.u], -15, tracking, "doubled pawn eg", sq);
                        }

                        // Isolated pawn.
                        const bb_isolated = bitboards.adjacent_file_masks[sq.u] & our_pawns;
                        const is_isolated: bool = bb_isolated == 0;
                        if (is_isolated) {
                            add(&pawn_scores_mg[us.u], -10, tracking, "isolated pawn mg", sq);
                            add(&pawn_scores_eg[us.u], -20, tracking, "isolated pawn eg", sq);
                        }

                        // Protected pawn.
                        const is_protected: bool = !is_isolated and (bb_this_piece & pawn_cover[us.u] != 0);
                        if (is_protected) {
                            add(&pawn_scores_mg[us.u], 10, tracking, "protected pawn mg", sq);
                            add(&pawn_scores_eg[us.u], 20, tracking, "protected pawn eg", sq);
                        }

                        // Protected and passed pawn
                        if (is_passed and !is_isolated) {
                            add(&pawn_scores_mg[us.u], 10, tracking, "protected passed pawn mg", sq);
                            add(&pawn_scores_eg[us.u], 20, tracking, "protected passed pawn eg", sq);
                        }

                        // Backward pawn.
                        if (is_backward_pawn(pos, us, sq)) {
                            add(&pawn_scores_mg[us.u], -10, tracking, "backward pawn mg", sq);
                            add(&pawn_scores_eg[us.u], -30, tracking, "backward pawn eg", sq);
                        }

                        const control: u64 = pawn_attacks & funcs.relative_side_bitboard(them);
                        if (control != 0) {
                            add(&pawn_scores_mg[us.u], 4, tracking, "pawn controlling opponent side mg", sq);
                            add(&pawn_scores_eg[us.u], 2, tracking, "pawn controlling opponent side eg", sq);
                        }

                        // const mob: u64 = (pawn_push);
                        // if (popcnt(mob) > 0) {
                        //     ps += 1;
                        // }
                    },
                    .knight => {
                        const knight_attacks: u64 = attacks.get_knight_attacks(sq);

                        // King area threats.
                        const att_king_area: u64 = knight_attacks & their_king_area;
                        if (att_king_area != 0) {
                            add(&king_attacker_count[us.u], 1, tracking, "king threat knight", sq);
                        }

                        // Knight outpost.
                        if (funcs.is_relative_rank_456(us, rank)) {
                            const up: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
                            if (pos.board[up.u].is_pawn()) {
                                add(&strategic_scores_mg[us.u], 20, tracking, "knight outpost mg", sq);
                                add(&strategic_scores_eg[us.u], 10, tracking, "knight outpost eg", sq);
                            }
                        }

                        // Reward knight is protected by a pawn.
                        if (bb_this_piece & pawn_cover[us.u] != 0) {
                            add(&strategic_scores_mg[us.u], 10, tracking, "knight protected mg", sq);
                            add(&strategic_scores_eg[us.u], 30, tracking, "knight protected eg", sq);
                        }

                        // Knight mobility.
                        const mob: u64 = knight_attacks & ~bb_us & ~pawn_cover[them.u];
                        if (popcnt(mob) > 4) {
                            add(&mobility_scores_mg[us.u], 10, tracking, "knight mobility mg", sq);
                            add(&mobility_scores_eg[us.u],  2, tracking, "knight mobility eg", sq);
                        }
                    },
                    .bishop => {
                        const our_bishops: u64 = bb_pieces;
                        const bishop_moves: u64 = attacks.get_bishop_attacks(sq, bb_all);

                        // King area threats.
                        const att_king_area: u64 = bishop_moves & their_king_area;
                        if (att_king_area != 0) {
                            add(&king_attacker_count[us.u], 1, tracking, "king threat bishop", sq);
                        }
                        // Bishop pair.
                        if (is_first_piece and (our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
                            bonus_scores[us.u] += 30;
                            if (lib.is_debug and tracking) io.debugprint("bishoppair 30\n", .{});
                        }

                        // Reward bishop is protected by a pawn.
                        if (bb_this_piece & pawn_cover[us.u] != 0) {
                            add(&strategic_scores_mg[us.u], 10, tracking, "bishop protected mg", sq);
                            add(&strategic_scores_eg[us.u], 30, tracking, "bishop protected eg", sq);
                        }

                        // Bad bishop. TODO enhance
                        if (relative_rank > bitboards.rank_1 and relative_rank < bitboards.rank_5) {
                            const pawns_in_front: u64 = attacks.get_pawn_attacks(sq, us) & pos.pawns(us);
                            const cnt: Value = popcnt_v(pawns_in_front);
                            if (cnt > 0) {
                                add(&strategic_scores_mg[us.u], -(cnt * 10), tracking, "bad bishop mg", sq);
                                add(&strategic_scores_eg[us.u], -(cnt * 20), tracking, "bad bishop eg", sq);
                            }
                        }

                        // Bishop mobility.
                        const mob: u64 = bishop_moves & ~bb_us & ~pawn_cover[them.u];
                        const m: u7 = @min(popcnt(mob), 5);
                        //io.debugprint("bishop moves = {}\n", .{m});
                        if (m > 0) {
                            add(&mobility_scores_mg[us.u], m * 4, tracking, "mobility bishop mg", sq);
                            add(&mobility_scores_eg[us.u], m * 2, tracking, "mobility bishop eg", sq);
                        }
                    },
                    .rook => {
                        const our_rooks: u64 = bb_pieces;
                        const rook_moves: u64 = attacks.get_rook_attacks(sq, bb_all);

                        // King area threat.
                        const att_king_area: u64 = rook_moves & their_king_area;
                        if (att_king_area != 0) {
                            add(&king_attacker_count[us.u], 1, tracking, "king threat rook", sq);
                        }

                        // Pigs on 7th rank if there are also enemy pawns on the 8th rank.
                        if (rank == funcs.relative_rank_7(us)) {
                            const their_pawns_on_7th = pos.pawns(them) & funcs.relative_rank_7_bitboard(them);
                            if (their_pawns_on_7th != 0) {
                                add(&strategic_scores_mg[us.u], 20, tracking, "rook on 7th mg", sq);
                                add(&strategic_scores_eg[us.u], 10, tracking, "rook on 7th eg", sq);
                            }
                        }

                        // Rook on open file
                        if (bitboards.file_bitboards[file] & pos.all_pawns() == 0) {
                            add(&strategic_scores_mg[us.u], 30, tracking, "rook open file mg", sq);
                            add(&strategic_scores_eg[us.u], 10, tracking, "rook open file eg", sq);
                        }
                        // Rook on half open file
                        else if (bitboards.file_bitboards[file] & pos.pawns(us) == 0) {
                            add(&strategic_scores_mg[us.u], 20, tracking, "rook halfopen file mg", sq);
                            add(&strategic_scores_eg[us.u],  5, tracking, "rook halfopen file eg", sq);
                        }

                        // Rooks connected.
                        if (popcnt(rook_moves & our_rooks) > 0) {
                            add(&strategic_scores_mg[us.u], 20, tracking, "rooks connected mg", sq);
                            add(&strategic_scores_eg[us.u], 10, tracking, "rooks connected eg", sq);
                        }

                        // Mobility
                        const mob: u64 = rook_moves & ~bb_us & ~pawn_cover[them.u];
                        const m: u7 = @min(popcnt(mob), 5);
                        if (m > 0) {
                            add(&mobility_scores_mg[us.u], m * 4, tracking, "mobility rook mg", sq);
                            add(&mobility_scores_eg[us.u], m * 2, tracking, "mobility rook eg", sq);
                        }
                    },
                    .queen => {
                        const queen_moves: u64 = attacks.get_queen_attacks(sq, bb_all);

                        // King area threats.
                        const att_king_area: u64 = queen_moves & their_king_area;
                        if (att_king_area != 0) {
                            add(&king_attacker_count[us.u], 1, tracking, "king threat queen", sq);
                        }

                        // Queen on open or halfopen file
                        if (bitboards.file_bitboards[file] & pos.all_pawns() == 0) {
                            add(&strategic_scores_mg[us.u], 10, tracking, "queen on open file mg", sq);
                            add(&strategic_scores_eg[us.u],  5, tracking, "queen on open file eg", sq);
                        }
                        else if (bitboards.file_bitboards[file] & pos.pawns(us) == 0) {
                            add(&strategic_scores_mg[us.u], 8, tracking, "queen on halfopen file mg", sq);
                            add(&strategic_scores_eg[us.u], 2, tracking, "queen on halfopen file eg", sq);
                        }

                        // Queen mobility
                        const mob: u64 = queen_moves & ~bb_us & ~pawn_cover[them.u];
                        const m: u7 = @min(popcnt(mob), 5);
                        if (m > 0) {
                            add(&mobility_scores_mg[us.u], m * 2, tracking, "mobility queen mg", sq);
                            add(&mobility_scores_eg[us.u], m * 1, tracking, "mobility queen eg", sq);
                        }
                    },
                    .king => {
                        const king_moves: u64 = attacks.get_king_attacks(our_king_sq) & ~bb_us;

                        // Check.
                        // if (check and us.e == pos.stm.e) {
                        //     add(k_score, Tables.in_check, us, piecetype, .in_check, tracking, sq);
                        // }

                        // // King area threat.
                        // if (phase == .Endgame) {
                        //     const att_king_area: u64 = attacks.get_king_attacks(our_king_sq) & their_king_area;
                        //     // EXPERIMENTAL
                        //     if (att_king_area != 0 and pos.stm.e == us.e) {
                        //         king_attackers.* += 1;
                        //         ps += 1;
                        //     }
                        // }

                        // Open ranks to our king
                        if (phase != .Endgame) {
                            const open_files: u64 = attacks.get_file_attacks(our_king_sq, bb_all) & ~bb_us; // changed was rank???
                            if (open_files != 0) {
                                add(&strategic_scores_mg[us.u], -30, tracking, "king open files mg", sq);
                                add(&strategic_scores_eg[us.u],  -5, tracking, "king open files eg", sq);
                            }
                        }

                        // Open diagonals to our king
                        if (phase != .Endgame) {
                            const open_diags: u64 = attacks.get_bishop_attacks(our_king_sq, bb_all) & ~bb_us;
                            if (open_diags != 0) {
                                add(&strategic_scores_mg[us.u], -20, tracking, "king open diags mg", sq);
                                add(&strategic_scores_eg[us.u],  -5, tracking, "king open diags eg", sq);
                            }
                        }

                        // King pawn protection.
                        if (!is_queenless) {
                            const pawn_protection = our_king_area & pos.pawns(us);
                            const cnt: u7 = popcnt(pawn_protection);
                            if (cnt > 0) {
                                add(&strategic_scores_mg[us.u], cnt * 8, tracking, "king pawn protection mg", sq);
                                add(&strategic_scores_eg[us.u], cnt * 5, tracking, "king pawn protection eg", sq);
                            }
                        }

                        // King piece protection.
                        if (!is_queenless) {
                            const piece_protection = our_king_area & pos.non_pawns(us);
                            const cnt: u7 = popcnt(piece_protection);
                            if (cnt > 0) {
                                add(&strategic_scores_mg[us.u], cnt * 4, tracking, "king piece protection mg", sq);
                                add(&strategic_scores_eg[us.u], cnt * 2, tracking, "king piece protection eg", sq);
                            }
                        }

                        // King mobility.
                        if (king_moves == 0) {
                            add(&mobility_scores_mg[us.u], -20, tracking, "mobility king mg", sq);
                            add(&mobility_scores_eg[us.u], -40, tracking, "mobility king eg", sq);
                        }

                    },
                }
            }
        }

        if (king_attacker_count[us.u] > 3)
            bonus_scores[us.u] += 50
        else if (king_attacker_count[us.u] > 2)
            bonus_scores[us.u] += 25
        else if (king_attacker_count[us.u] > 1)
            bonus_scores[us.u] += 4;
    }

    // Taper sliding values.
    const w_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[0], pesto_scores_eg[0]);
    const b_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[1], pesto_scores_eg[1]);

    const w_pawns: Value = Tables.sliding_score(non_pawn_material, pawn_scores_mg[0], pawn_scores_eg[0]);
    const b_pawns: Value = Tables.sliding_score(non_pawn_material, pawn_scores_mg[1], pawn_scores_eg[1]);

    const w_strategic: Value = Tables.sliding_score(non_pawn_material, strategic_scores_mg[0], strategic_scores_eg[0]);
    const b_strategic: Value = Tables.sliding_score(non_pawn_material, strategic_scores_mg[1], strategic_scores_eg[1]);

    const w_mobility: Value = Tables.sliding_score(non_pawn_material, mobility_scores_mg[0], mobility_scores_eg[0]);
    const b_mobility: Value = Tables.sliding_score(non_pawn_material, mobility_scores_mg[1], mobility_scores_eg[1]);

    // const w_score: Value = material_scores[0] + w_pesto + w_strategic + w_mobility + bonus_scores[0];
    // const b_score: Value = material_scores[1] + b_pesto + b_strategic + b_mobility + bonus_scores[1];

    const w_score: Value = material_scores[0] + w_pesto + w_pawns + w_strategic + w_mobility + bonus_scores[0];
    const b_score: Value = material_scores[1] + b_pesto + b_pawns + b_strategic + b_mobility + bonus_scores[1];

    // const w_score: Value =  w_pesto + bonus_scores[0];
    // const b_score: Value = b_pesto + bonus_scores[1];

    var result = (w_score - b_score) + 12; // stm
    if (negate) {
         result = -result;
    }

    // TODO: not sure.
    //result = scale_towards_draw(result, pos.state.rule50);

    if (tracking) {
        //const tm: Value = if (pos.stm.e == .white) 20 else -20;
        lib.io.print(
            \\perspective    : {t}
            \\material       : {} {}
            \\pestos         : {} {}
            \\pawns          : {} {}
            \\strategic      : {} {}
            \\mobility       : {} {}
            \\bonus          : {} {}
            \\king attackers : {} {}
            \\scores         : {} {}
            \\result         : {}
            \\
            ,
            .{
                pos.stm.e,
                material_scores[0], material_scores[1],
                w_pesto, b_pesto,
                w_pawns, b_pawns,
                w_strategic, b_strategic,
                w_mobility, b_mobility,
                bonus_scores[0], bonus_scores[1],
                king_attacker_count[0], king_attacker_count[1],
                w_score, b_score,
                result,
            }
        );
    }
    tt_eval.store(pos.key, result);
    return result;
}


fn eval(pos: *const Position, non_pawn_material: Value, hash: *tt.EvalTranspositionTable, tt_pawns: *tt.PawnTranspositionTable, comptime phase: GamePhase, comptime tracking: bool) Value {

    if (hash.probe(pos.key)) |e| {
        return e;
    }

    _ = tt_pawns;
    const bb_all = pos.all();
    const negate: bool = pos.stm.e == .black;
    const is_queenless: bool = pos.all_queens() == 0;
    const has_pawns: bool = pos.all_bishops() != 0;

    // EXPERIMENTAL: pawn up in pawn endgame is often desastrous.
    // const pawns_add: Value = if (non_pawn_material == 0) 30 else 0;
    const material_scores: [2]Value = .{ pos.values[Color.WHITE.u], pos.values[Color.BLACK.u]};

    // Sliding values of the Pesto tables, which we taper at the end.
    // var pesto_phase: Value = 0;
    var pesto_scores_mg: [2]Value = .{ 0, 0 };
    var pesto_scores_eg: [2]Value = .{ 0, 0 };

    // Scores
    var bonus_scores: [2]Value = .{ 0, 0 };
    var king_attacker_count: [2]Value = .{ 0, 0 };

    // We need pawn hits multiple times.
    const pawn_cover: [2]u64 = .{
        if (!has_pawns) 0 else funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northwest) | funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northeast),
        if (!has_pawns) 0 else funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northwest) | funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northeast),
    };


    ////////////////////////////////////////////////////////////////
    // Color loop.
    ////////////////////////////////////////////////////////////////
    inline for (Color.all) |us| {

        const them = comptime us.opp();
        const bb_us = pos.by_color(us);
        const our_king_sq = pos.king_square(us);
        const their_king_sq = pos.king_square(them);
        const our_king_area = bitboards.king_areas[our_king_sq.u];
        const their_king_area = bitboards.king_areas[their_king_sq.u];

        const mg: *Value = &pesto_scores_mg[us.u];
        const eg: *Value = &pesto_scores_eg[us.u];
        const bonus: *Value = &bonus_scores[us.u];
        const king_attackers: *Value = &king_attacker_count[us.u];

        // if (phase == .Opening) {
        //     const pawnrow: u64 = funcs.relative_rank(us, bitboards.rank_2) & pos.pawns(us);
        //     const cnt: Value = popcnt_v(pawnrow);
        //     if (cnt > 5)
        //         bonus.* -= 10;
        // }

        ////////////////////////////////////////////////////////////////
        // Piece loop.
        ////////////////////////////////////////////////////////////////
        inline for (PieceType.all) |piecetype| {
            var is_first_piece: bool = true;
            const bb_current = pos.pieces(piecetype, us);
            var bb_running = bb_current;

            ////////////////////////////////////////////////////////////////
            // Square loop.
            ////////////////////////////////////////////////////////////////
            while (bb_running != 0) : (is_first_piece = false) {
                const sq: Square = funcs.pop_square(&bb_running);
                const bb_this_piece = bb_current & sq.to_bitboard();
                const file: u3 = sq.file();
                const rank: u3 = sq.rank();

                // Bonus per square.
                var ps: Value = 0;

                ////////////////////////////////////////////////////////////////
                // Piece loop.
                ////////////////////////////////////////////////////////////////
                switch (piecetype.e) {
                    .pawn => {
                        const our_pawns: u64 = bb_current;
                        const pawn_attacks: u64 = attacks.get_pawn_attacks(sq, us);
                        const pawn_moves: u64 = funcs.pawns_shift(bb_this_piece, us, funcs.PawnShift.up) & ~bb_all;

                        // King area threat.
                        const att_king_area: u64 = (pawn_attacks | pawn_moves) & their_king_area;
                        if (att_king_area != 0) {
                            ps += 2; // NOTE: Do not add to king attackers. Bad results.
                        }

                        // Passed pawn.
                        const is_passed: bool = is_passed_pawn(pos, us, sq);
                        if (is_passed) {
                            ps += funcs.relative_rank(us, sq.rank());
                        }

                        // Doubled pawn.
                        const bb_doubled = bitboards.file_bitboards[file] & our_pawns;
                        const is_doubled: bool = popcnt(bb_doubled) > 1;
                        if (is_doubled) {
                            ps -= 4;
                        }

                        // Isolated pawn.
                        const bb_isolated = bitboards.adjacent_file_masks[sq.u] & our_pawns;
                        const is_isolated: bool = bb_isolated == 0;
                        if (is_isolated) {
                            ps -= 2;
                        }

                        // Protected pawn.
                        const is_protected: bool = !is_isolated and (bb_this_piece & pawn_cover[us.u] != 0);
                        if (is_protected) {
                            ps += 2;
                        }

                        if (is_passed and !is_isolated) {
                            ps += 4;
                        }

                        // Backward pawn.
                        if (is_backward_pawn(pos, us, sq)) {
                            ps -= 2;
                        }

                        // const mob: u64 = (pawn_moves);
                        // if (popcnt(mob) > 0) {
                        //     ps += 1;
                        // }
                    },
                    .knight => {
                        const knight_attacks: u64 = attacks.get_knight_attacks(sq);

                        // King area threats.
                        const att_king_area: u64 = knight_attacks & their_king_area;
                        if (att_king_area != 0) {
                            king_attackers.* += 1;
                            ps += 4;
                        }

                        // Reward knight is protected by a pawn.
                        if (bb_this_piece & pawn_cover[us.u] != 0) {
                            ps += 2;
                        }

                        // Knight mobility.
                        const mob: u64 = knight_attacks & ~bb_us & ~pawn_cover[them.u];
                        if (popcnt(mob) > 4) {
                            ps += 1;
                        }
                    },
                    .bishop => {
                        const our_bishops: u64 = bb_current;
                        const bishop_moves: u64 = attacks.get_bishop_attacks(sq, bb_all);

                        // King area threats.
                        const att_king_area: u64 = bishop_moves & their_king_area;
                        if (att_king_area != 0) {
                            king_attackers.* += 1;
                            ps += 4;
                        }

                        // Bishop pair.
                        if (is_first_piece and (our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
                            bonus.* += 15;
                        }

                        // Bishop protected.
                        if (bb_this_piece & pawn_cover[us.u] != 0) {
                            ps += 2;
                        }

                        // Bishop mobility.
                        const mob: u64 = bishop_moves & ~bb_us & ~pawn_cover[them.u];
                        if (popcnt(mob) > 4) {
                            ps += 2;
                        }
                    },
                    .rook => {
                        const our_rooks: u64 = bb_current;
                        const rook_moves: u64 = attacks.get_rook_attacks(sq, bb_all);

                        // King area threat.
                        const att_king_area: u64 = rook_moves & their_king_area;
                        if (att_king_area != 0) {
                            king_attackers.* += 1;
                            ps += 2;
                        }

                        // Pigs on 7th rank if there are also enemy pawns on the 8th rank.
                        if (rank == funcs.relative_rank_7(us)) {
                            const their_pawns_on_7th = pos.pawns(them) & funcs.relative_rank_7_bitboard(them);
                            if (their_pawns_on_7th != 0) {
                                ps += 10;
                            }
                        }

                        // Rook on open or halfopen file
                        if (bitboards.file_bitboards[file] & pos.all_pawns() == 0) {
                            ps += 4;
                        }
                        else if (bitboards.file_bitboards[file] & pos.pawns(us) == 0) {
                            ps += 6;
                        }

                        // Rook protected.
                        if (bb_this_piece & pawn_cover[us.u] != 0) {
                            ps += 2;
                        }

                        // Rooks connected.
                        if (popcnt(rook_moves & our_rooks) > 0) {
                            ps += 4;
                        }

                        // Mobility
                        const mob: u64 = rook_moves & ~bb_us & ~pawn_cover[them.u];
                        if (popcnt(mob) > 5) {
                            ps += 4;
                        }
                    },
                    .queen => {
                        const queen_moves: u64 = attacks.get_queen_attacks(sq, bb_all);

                        // King area threats.
                        const att_king_area: u64 = queen_moves & their_king_area;
                        if (att_king_area != 0) {
                            king_attackers.* += 1;
                            ps += 4;
                        }

                        // Queen on open or halfopen file
                        if (bitboards.file_bitboards[file] & pos.all_pawns() == 0) {
                            ps += 4;
                        }
                        else if (bitboards.file_bitboards[file] & pos.pawns(us) == 0) {
                            ps += 4;
                        }

                        // Queen mobility
                        const mob: u64 = queen_moves & ~bb_us & ~pawn_cover[them.u];
                        if (popcnt(mob) > 8) {
                            ps += 4;
                        }
                    },
                    .king => {
                        const king_moves: u64 = attacks.get_king_attacks(our_king_sq) & ~bb_us;

                        // Open ranks to our king
                        if (phase != .Endgame) {
                            const open_files: u64 = attacks.get_file_attacks(our_king_sq, bb_all) & ~bb_us;
                            if (open_files != 0)
                                ps -= 6;
                        }

                        // Open diagonals to our king
                        if (phase != .Endgame) {
                            const open_diags: u64 = attacks.get_bishop_attacks(our_king_sq, bb_all) & ~bb_us;
                            if (open_diags != 0)
                                ps -= 4;
                        }

                        // King pawn protection.
                        if (!is_queenless) {
                            const pawn_protection = our_king_area & pos.pawns(us);
                            if (popcnt(pawn_protection) > 2) {
                                ps += 2;
                            }
                        }

                        // King piece protection.
                        if (!is_queenless) {
                            const piece_protection = our_king_area & pos.non_pawns(us);
                            if (popcnt(piece_protection) > 2) {
                                ps += 1;
                            }
                        }

                        // King mobility.
                        if (king_moves == 0) {
                            ps -= 2;
                        }

                    },
                }

                // pesto + ps bonus.
                const pair = Tables.get_scorepair(us, piecetype, sq);
                mg.* += (pair.mg + ps);
                eg.* += (pair.eg + ps);

            }
        }

        if (king_attackers.* > 3)
            bonus.* += 50
        else if (king_attackers.* > 2)
            bonus.* += 25
        else if (king_attackers.* > 1)
            bonus.* += 4;
    }

    // Taper sliding values.
    const w_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[0], pesto_scores_eg[0]);
    const b_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[1], pesto_scores_eg[1]);

    const w_score: Value =  material_scores[0] + w_pesto + bonus_scores[0];
    const b_score: Value = material_scores[1] + b_pesto + bonus_scores[1];

    // const w_score: Value =  w_pesto + bonus_scores[0];
    // const b_score: Value = b_pesto + bonus_scores[1];

    var result = (w_score - b_score) + 12; // stm
    if (negate) {
         result = -result;
    }

    // TODO: not sure.
    //result = scale_towards_draw(result, pos.state.rule50);

    if (tracking) {
        //const tm: Value = if (pos.stm.e == .white) 20 else -20;
        lib.io.print(
            \\perspective : {t}
            \\material    : {} {}
            \\pestos      : {} {}
            \\bonus       : {} {}
            \\attackers   : {} {}
            \\result      : {}
            \\
            ,
            .{
                pos.stm.e,
                material_scores[0], material_scores[1],
                w_pesto, b_pesto,
                bonus_scores[0], bonus_scores[1],
                king_attacker_count[0], king_attacker_count[1],
                result,
            }
        );
    }
    hash.store(pos.key, result);
    return result;
}


pub fn scale_towards_draw(score: Value, drawcounter: u16) Value {
    if (drawcounter <= 2) return score;
    // drawcounter ranges 0...100
    const s: f32 = @floatFromInt(score);
    const d: f32 = @floatFromInt(drawcounter);
    const factor: f32 = (100.0 - d) / 100.0;
    return @intFromFloat(s * factor);
}

pub fn is_trivial_win(pos: *const Position) bool {
    _ = pos;
    return false;
}

pub fn is_draw_by_insufficient_material(pos: *const Position) bool {
    if (pos.all_pawns() != 0) return false;
    // No pieces.
    if (pos.non_pawn_material() == 0) return true;
    const w: Value = pos.materials[0];
    const b: Value = pos.materials[1];
    if (w + b <= types.material_bishop) return true;
    if (w + b == types.material_knight * 2) return true;
    return false;
}

/// Static exchange evaluation. Get score of capture fest on square.
/// TODO: perfect pins?
/// TODO: threshold.
pub fn see_score(pos: *const Position, m: Move) Value {
    //const st: *const StateInfo = pos.state;gi
    if (comptime lib.is_paranoid) {
        assert(m.is_capture());
    }

    // This will hit a no_piece square.
    if (m.is_ep()) {
        return 0;
    }

    const from: Square = m.from;
    const to: Square = m.to;
    const value_them = pos.get(to).value();
    const value_us = pos.get(from).value();
    // good capture: if (value_them - value_us > P.value()) return true;
    var gain: [24]Value = @splat(0);
    gain[0] = value_them;
    gain[1] = value_us - value_them;

    var depth: u8 = 1;
    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();
    var occupation = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var attackers: u64 = pos.get_combined_attacks_to_for_occupation(occupation, to);
    var bb: u64 = 0;
    var side: Color = pos.stm;

    attackloop: while (true) {
        attackers &= occupation;
        if (attackers == 0) break;
        side = side.opp();

        // Pawn.
        bb = attackers & pos.pawns(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = P.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Knight.
        bb = attackers & pos.knights(side); // & ~st.pins;
        if (bb != 0) {
            depth += 1;
            gain[depth] = N.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight (cannot reveal more sliding attackers).
            continue :attackloop;
        }

        // Bishop.
        bb = attackers & pos.bishops(side);// & ~st.pins_orthogonal;
        if (bb != 0) {
            depth += 1;
            gain[depth] = B.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 bishop.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Rook.
        bb = attackers & pos.rooks(side);// & ~st.pins_diagonal[side.u];
        if (bb != 0) {
            depth += 1;
            gain[depth] = R.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        // Queen.
        bb = attackers & pos.queens(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = Q.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 queen.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        bb = attackers & pos.kings(side);
        if (bb != 0) {
            side = side.opp();
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 king.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            attackers &= occupation & pos.by_color(side);
            // King can take the next.
            if (attackers == 0) {
                depth += 1;
                gain[depth] = K.value() - gain[depth - 1];
                break :attackloop;
            }
            break :attackloop;
        }
        break :attackloop;
    }

    // Bubble up the score
    depth -= 1;
    while (depth > 0) : (depth -= 1) {
        if (gain[depth] > -gain[depth - 1]) {
            gain[depth - 1] = -gain[depth];
        }
    }
    return gain[0];
}

pub fn is_protected_by_pawn(pos: *const Position, comptime us: Color, sq: Square) bool {
    const them = comptime us.opp();
    return attacks.get_pawn_attacks(sq, them) & pos.pawns(us) != 0; // using inversion trick
}

pub fn is_passed_pawn(pos: *const Position, comptime us: Color, sq: Square) bool {
    return switch (us.e) {
        .white => bitboards.get_passed_pawn_mask(.WHITE, sq) & pos.pawns(us.opp()) == 0,
        .black => bitboards.get_passed_pawn_mask(.BLACK, sq) & pos.pawns(us.opp()) == 0,
    };
}

pub fn is_backward_pawn(pos: *const Position, comptime us: Color, sq: Square) bool {
    if (true) return false;
    const p: Piece = comptime Piece.create(P, us);
    if (sq.next(.west)) |adj| { if (pos.board[adj.u].e == p.e) return false; }
    if (sq.next(.east)) |adj| { if (pos.board[adj.u].e == p.e) return false; }
    const bb_same_file: u64 = ~(bitboards.file_bitboards[sq.file()] & pos.pawns(us));
    const square_in_front: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
    const covered: u64 = attacks.get_pawn_attacks(square_in_front, us) & pos.pawns(us.opp()); // inversion trick
    return switch (us.e) {
        .white => (bitboards.get_passed_pawn_mask(.WHITE, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0,
        .black => (bitboards.get_passed_pawn_mask(.BLACK, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0,
    };
}

// pub fn is_pinned(pos: *const Position, us: Color, sq: Square) bool {
//     return pos.state.pins[us.u] & sq.to_bitboard() != 0;
// }

// pub fn is_pinned_diagonally(pos: *const Position, us: Color, sq: Square) bool {
//     return pos.state.pins_diagonal[us.u] & sq.to_bitboard() != 0;
// }

// pub fn is_pinned_orthogonally(pos: *const Position, us: Color, sq: Square) bool {
//     return pos.state.pins_orthogonal[us.u] & sq.to_bitboard() != 0;
// }

pub const Tables = struct {
    fn get_index(us: Color, sq: Square) u6 {
        return switch(us.e) {
            .white => sq.u ^ 56,
            .black => sq.u,
        };
    }

    /// Returns the table values for opening and endgame, between which we slide the final score.
    pub fn get_scorepair(comptime us: Color, comptime pc: PieceType, sq: Square) struct { mg: Value, eg: Value} {

        const mg: [*]const Value = switch (pc.e) {
            .pawn   => &mg_pawn_table,
            .knight => &mg_knight_table,
            .bishop => &mg_bishop_table,
            .rook   => &mg_rook_table,
            .queen  => &mg_queen_table,
            .king   => &mg_king_table,
        };

        const eg: [*]const Value = switch (pc.e) {
            .pawn   => &eg_pawn_table,
            .knight => &eg_knight_table,
            .bishop => &eg_bishop_table,
            .rook   => &eg_rook_table,
            .queen  => &eg_queen_table,
            .king   => &eg_king_table,
        };

        const idx: u6 = get_index(us, sq);
        return .{ .mg = mg[idx], .eg = eg[idx] };
    }

    pub fn sliding_score(non_pawn_material: Value, opening: Value, endgame: Value) Value {
        const max: i32 = comptime types.max_material_without_pawns;
        const phase: i32 = @min(non_pawn_material, max);
        return @truncate(@divTrunc(opening * phase + endgame * (max - phase), max));
    }

    //const int gamephaseInc[12] = {0,0,1,1,1,1,2,2,4,4,0,0}; // p p n n b b r r q q k k
    const pesto_game_phase: [8]Value = .{
        0, // no_piece
        0, // pawn
        1, // knight
        1, // bishop
        2, // rook
        4, // queen
        0, // king
        0, // nothing
    };

    // Pesto tables.
    pub const mg_pawn_table: [64]Value = .{
        0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    pub const eg_pawn_table: [64]Value = .{
        0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    pub const mg_knight_table: [64]Value = .{
        -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
    };

    pub const eg_knight_table: [64]Value = .{
        -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64,
    };

    pub const mg_bishop_table: [64]Value = .{
        -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21,
    };

    pub const eg_bishop_table: [64]Value = .{
        -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15, -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17,
    };

    pub const mg_rook_table: [64]Value = .{
         32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26,
    };

    pub const eg_rook_table: [64]Value = .{
        13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
    };

    pub const mg_queen_table: [64]Value = .{
        -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50,
    };

    pub const eg_queen_table: [64]Value = .{
        -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41,
    };

    pub const mg_king_table: [64]Value = .{
        -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14,
    };

    pub const eg_king_table: [64]Value = .{
        -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43
    };

    // // Experimental symmetrical Pesto's.
    // const mg_pawn_table: [64]Value = .{ 0, 0, 0, 0, 0, 0, 0, 0, 43, 84, 93, 81, 81, 93, 84, 43, -13, 16, 41, 48, 48, 41, 16, -13, -18, 15, 9, 22, 22, 9, 15, -18, -26, 4, 0, 14, 14, 0, 4, -26, -19, 14, 0, -3, -3, 0, 14, -19, -28, 18, 2, -19, -19, 2, 18, -28, 0, 0, 0, 0, 0, 0, 0, 0 };
    // const eg_pawn_table: [64]Value = .{ 0, 0, 0, 0, 0, 0, 0, 0, 182, 169, 145, 140, 140, 145, 169, 182, 89, 91, 69, 61, 61, 69, 91, 89, 24, 20, 8, 1, 1, 8, 20, 24, 6, 6, -5, -7, -7, -5, 6, 6, -2, 3, -5, 0, 0, -5, 3, -2, 3, 5, 4, 11, 11, 4, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0 };
    // const mg_knight_table: [64]Value = .{ -137, -52, -65, 6, 6, -65, -52, -137, -45, -17, 67, 29, 29, 67, -17, -45, -1, 66, 83, 74, 74, 83, 66, -1, 6, 17, 44, 45, 45, 44, 17, 6, -10, 12, 17, 20, 20, 17, 12, -10, -19, 8, 14, 14, 14, 14, 8, -19, -24, -33, 3, -2, -2, 3, -33, -24, -64, -20, -43, -25, -25, -43, -20, -64 };
    // const eg_knight_table: [64]Value = .{ -78, -50, -20, -29, -29, -20, -50, -78, -38, -16, -25, -5, -5, -25, -16, -38, -32, -19, 0, 4, 4, 0, -19, -32, -17, 5, 16, 22, 22, 16, 5, -17, -18, -1, 16, 20, 20, 16, -1, -18, -22, -11, -2, 12, 12, -2, -11, -22, -43, -21, -15, -3, -3, -15, -21, -43, -46, -50, -20, -18, -18, -20, -50, -46 };
    // const mg_bishop_table: [64]Value = .{ -18, 5, -62, -31, -31, -62, 5, -18, -36, 17, 20, 8, 8, 20, 17, -36, -9, 37, 46, 37, 37, 46, 37, -9, -3, 6, 28, 43, 43, 28, 6, -3, -1, 11, 12, 30, 30, 12, 11, -1, 5, 16, 21, 14, 14, 21, 16, 5, 2, 24, 18, 3, 3, 18, 24, 2, -27, -21, -13, -17, -17, -13, -21, -27 };
    // const eg_bishop_table: [64]Value = .{ -19, -19, -10, -7, -7, -10, -19, -19, -11, -4, -3, -7, -7, -3, -4, -11, 3, -4, 3, -1, -1, 3, -4, 3, 0, 6, 11, 11, 11, 11, 6, 0, -7, 0, 11, 13, 13, 11, 0, -7, -13, -5, 5, 11, 11, 5, -5, -13, -20, -16, -8, 1, 1, -8, -16, -20, -20, -7, -19, -7, -7, -19, -7, -20 };
    // const mg_rook_table: [64]Value = .{ 37, 36, 20, 57, 57, 20, 36, 37, 35, 29, 62, 71, 71, 62, 29, 35, 5, 40, 35, 26, 26, 35, 40, 5, -22, -9, 21, 25, 25, 21, -9, -22, -29, -10, -9, 4, 4, -9, -10, -29, -39, -15, -8, -7, -7, -8, -15, -39, -57, -11, -4, -5, -5, -4, -11, -57, -22, -25, 4, 16, 16, 4, -25, -22 };
    // const eg_rook_table: [64]Value = .{ 9, 9, 15, 13, 13, 15, 9, 9, 7, 10, 8, 4, 4, 8, 10, 7, 2, 1, 2, 4, 4, 2, 1, 2, 3, 1, 7, 1, 1, 7, 1, 3, -4, -1, 1, 0, 0, 1, -1, -4, -10, -4, -8, -4, -4, -8, -4, -10, -4, -8, -4, -3, -3, -4, -8, -4, -14, 3, -5, -3, -3, -5, 3, -14 };
    // const mg_queen_table: [64]Value = .{ 8, 21, 36, 35, 35, 36, 21, 8, 15, -5, 26, -7, -7, 26, -5, 15, 22, 15, 31, 18, 18, 31, 15, 22, -13, -14, 0, -8, -8, 0, -14, -13, -6, -11, -6, -6, -6, -6, -11, -6, -4, 8, -4, -3, -3, -4, 8, -4, -17, -5, 13, 5, 5, 13, -5, -17, -25, -24, -17, -2, -2, -17, -24, -25 };
    // const eg_queen_table: [64]Value = .{ 5, 16, 20, 27, 27, 20, 16, 5, -8, 25, 28, 49, 49, 28, 25, -8, -5, 12, 22, 48, 48, 22, 12, -5, 19, 39, 32, 51, 51, 32, 39, 19, 2, 33, 26, 39, 39, 26, 33, 2, -5, -8, 16, 7, 7, 16, -8, -5, -27, -29, -26, -16, -16, -26, -29, -27, -37, -24, -27, -24, -24, -27, -24, -37 };
    // const mg_king_table: [64]Value = .{ -26, 12, -9, -35, -35, -9, 12, -26, 0, -19, -12, -7, -7, -12, -19, 0, -15, 23, 4, -18, -18, 4, 23, -15, -26, -17, -18, -28, -28, -18, -17, -26, -50, -17, -35, -42, -42, -35, -17, -50, -20, -14, -26, -45, -45, -26, -14, -20, 4, 8, -12, -53, -53, -12, 8, 4, 0, 30, -8, -23, -23, -8, 30, 0 };
    // const eg_king_table: [64]Value = .{ -45, -15, -1, -14, -14, -1, -15, -45, 0, 20, 26, 17, 17, 26, 20, 0, 11, 30, 34, 17, 17, 34, 30, 11, -2, 24, 28, 26, 26, 28, 24, -2, -14, 2, 22, 25, 25, 22, 2, -14, -14, 2, 13, 22, 22, 13, 2, -14, -22, -8, 4, 13, 13, 4, -8, -22, -48, -29, -17, -19, -19, -17, -29, -48 };
};

// A little 1 second time eval bench.
pub fn bench(pos: *const Position, hash: *tt.EvalTranspositionTable) void {
    var v: i64 = 0;
    var cnt: u64 = 0;
    var timer = utils.Timer.start();
    while (timer.elapsed_ms() < 1000) {
        v += evaluate(pos, hash);
        cnt += 1;
    }
    const t = timer.read();
    lib.io.print("v {} cnt {} {}\n", .{v, cnt, funcs.nps(cnt, t)});
}


/// Not used. Incomplete copy of see_value.
pub fn see_unfinished_bool(pos: *const Position, m: Move) bool {
    if (m.type == .promotion) return true;

    const from: Square = m.from;
    const to: Square = m.to;

    const victim: Value = pos.get(to).value();
    const attacker: Value = pos.get(from).value();

    // Fast delta: clearly winning trades (e.g., PxN)
    if (victim - attacker > P.value()) return true;

    var occ = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var atks: u64 = pos.get_all_attacks_to_for_occupation(occ, to);
    const qb = pos.all_queens_bishops();
    const qr = pos.all_queens_rooks();

    var side: Color = pos.stm;
    var bb: u64 = 0;

    // balance = margin we must preserve to stay >= 0 (threshold = 0)
    var balance: Value = victim - attacker;

    while (true) {
        atks &= occ;
        if (atks == 0) break;
        side = side.opp(); // next capturer

        // Pawn
        bb = atks & pos.pawns(side);
        if (bb != 0) {
            balance = -balance - P.value();
            if (balance < 0) return (side.e != pos.stm.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Knight
        bb = atks & pos.knights(side);
        if (bb != 0) {
            balance = -balance - N.value();
            if (balance < 0) return (side.e != pos.stm.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            continue;
        }

        // Bishop
        bb = atks & pos.bishops(side);
        if (bb != 0) {
            balance = -balance - B.value();
            if (balance < 0) return (side.e != pos.stm.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Rook
        bb = atks & pos.rooks(side);
        if (bb != 0) {
            balance = -balance - R.value();
            if (balance < 0) return (side.e != pos.stm.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_rook_attacks(to, occ) & qr;
            continue;
        }

        // Queen
        bb = atks & pos.queens(side);
        if (bb != 0) {
            balance = -balance - Q.value();
            if (balance < 0) return (side.e != pos.stm.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= (attacks.get_bishop_attacks(to, occ) & qb);
            atks |= (attacks.get_rook_attacks(to, occ) & qr);
            continue;
        }

        // King ends the sequence
        if ((atks & pos.kings(side)) != 0) break; // TODO: BUGGY.
        break;
    }

    // No early fail: if the last side to *have the move* is the opponent, our capture stands.
    return (side.e != pos.stm.e);
}

// EXPERIMENTAL symmetry code.

    // symm(eval.Tables.mg_pawn_table, "mg_pawn_table");
    // symm(eval.Tables.eg_pawn_table, "eg_pawn_table");

    // symm(eval.Tables.mg_knight_table, "mg_knight_table");
    // symm(eval.Tables.eg_knight_table, "eg_knight_table");

    // symm(eval.Tables.mg_bishop_table, "mg_bishop_table");
    // symm(eval.Tables.eg_bishop_table, "eg_bishop_table");

    // symm(eval.Tables.mg_rook_table, "mg_rook_table");
    // symm(eval.Tables.eg_rook_table, "eg_rook_table");

    // symm(eval.Tables.mg_queen_table, "mg_queen_table");
    // symm(eval.Tables.eg_queen_table, "eg_queen_table");

    // symm(eval.Tables.mg_king_table, "mg_king_table");
    // symm(eval.Tables.eg_king_table, "eg_king_table");

    // const bb: bitboards.BitBoard = .init(bitboards.bb_border);
    // funcs.print_bitboard(bb.u);
    // lib.io.debugprint("{}\n", .{bb.ranks.b});





