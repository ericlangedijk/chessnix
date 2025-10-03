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
const StateInfo = position.StateInfo;
const Position = position.Position;

const P: PieceType = types.P;
const N: PieceType = types.N;
const B: PieceType = types.B;
const R: PieceType = types.R;
const Q: PieceType = types.Q;
const K: PieceType = types.K;

/// Returns the evaluation from white perspective, with detailed debug info.
pub fn evaluate_with_tracking_absolute(pos: *const Position) Value {
    const phase: GamePhase = pos.phase();
    const v: Value = switch (phase) {
        .Opening => eval(pos, .Opening, true),
        .Midgame => eval(pos, .Midgame, true),
        .Endgame => eval(pos, .Endgame, true),
    };
    return if (pos.to_move.e == .white) v else -v;
}

pub fn evaluate(pos: *const Position) Value {
    const phase: GamePhase = pos.phase();
    switch (phase) {
        .Opening => return eval(pos, .Opening, false),
        .Midgame => return eval(pos, .Midgame, false),
        .Endgame => return eval(pos, .Endgame, false),
    }
}

pub fn simple_eval(pos: *const Position, comptime us: Color) Value {
    return switch (us.e) {
        .white => pos.values[Color.WHITE.u] - pos.values[Color.BLACK.u],
        .black => pos.values[Color.BLACK.u] - pos.values[Color.WHITE.u],
    };
}

fn add(mg: *Value, eg: *Value, delta: Value) void
{
    //_ = mg; _ = eg; _ = delta;
    mg.* += delta;
    eg.* += delta;
}

/// Evaluation from the perspective of the side to move. Very bad it is.
fn eval(pos: *const Position, comptime phase: GamePhase, comptime tracking: bool) Value {

    // ? pawn_majority_q_side
    // ? pawn_majority_k_side
    // ? bishop_pair
    // ? bad_bishop
    // ? strong_square
    // ? outpost
    // ? can_castle

    //const check: bool = pos.state.checkers != 0;
    //const st: *const StateInfo = pos.state;
    const non_pawn_material: Value = pos.non_pawn_material();
    const bb_all = pos.all();
    const negate: bool = pos.to_move.e == .black;
    const is_queenless: bool = pos.all_queens() == 0;

    // EXPERIMENTAL: pawn up in pawn endgame is often desastrous.
    // const pawns_add: Value = if (non_pawn_material == 0) 30 else 0;
    const material_scores: [2]Value = .{ pos.values[Color.WHITE.u], pos.values[Color.BLACK.u]};

    // var covered: [2]u64 = undefined; // us - them
    // switch (pos.to_move.e) {
    //     .white => covered = .{ pos.attacks_by(Color.WHITE), pos.attacks_by(Color.BLACK) },
    //     .black => covered = .{ pos.attacks_by(Color.BLACK), pos.attacks_by(Color.WHITE) },
    // }

    // Sliding values of the Pesto tables, which we taper at the end.
    // var pesto_phase: Value = 0;
    var pesto_scores_mg: [2]Value = .{ 0, 0 };
    var pesto_scores_eg: [2]Value = .{ 0, 0 };

    // Scores
    var bonus_scores: [2]Value = .{ 0, 0 };
    var king_attacker_count: [2]Value = .{ 0, 0 };

    ////////////////////////////////////////////////////////////////
    // Color loop.
    ////////////////////////////////////////////////////////////////
    inline for (Color.all) |us| {

        const them = comptime us.opp();
        const bb_us = pos.by_color(us);
        const our_king_sq = pos.king_square(us);
        const their_king_sq = pos.king_square(them);
        const our_king_area = bitboards.get_king_area(our_king_sq);
        const their_king_area = bitboards.get_king_area(their_king_sq);

        const mg: *Value = &pesto_scores_mg[us.u];
        const eg: *Value = &pesto_scores_eg[us.u];
        const bonus: *Value = &bonus_scores[us.u];
        const king_attackers: *Value = &king_attacker_count[us.u];
        //const covered_by_them: u64 = covered[us.u];
        //const covered_by_us: u64 = covered[us.u];

        ////////////////////////////////////////////////////////////////
        // Piece loop.
        ////////////////////////////////////////////////////////////////
        inline for (PieceType.all) |piecetype| {
            //const piece: Piece = comptime Piece.make(piecetype, us);
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
                //const relative_rank: u3 = funcs.relative_rank(us, rank);

                // Keep track of Pesto table scores.
                const pair = Tables.get_scorepair(us, piecetype, sq);
                var ps: Value = 0;

                // NOTE: Keep this int-based pesto around for comparison
                //pesto_phase += Tables.pesto_game_phase[piecetype.u];

                switch (piecetype.e) {
                    .pawn => {
                        const our_pawns: u64 = bb_current;
                        const their_pawns: u64 = pos.pawns(them); _ = their_pawns;
                        const pawn_attacks: u64 = attacks.get_pawn_attacks(sq, us);
                        const pawn_moves: u64 = funcs.pawns_shift(bb_this_piece, us, funcs.PawnShift.up);

                        // King area threat.
                        const att_king_area: u64 = (pawn_attacks | pawn_moves) & their_king_area;
                        if (att_king_area != 0) {
                            king_attackers.* += 1;
                            ps += 2;
                        }

                        // Passed pawn.
                        const is_passed: bool = is_passed_pawn(pos, us, sq);
                        if (is_passed) {
                            ps += 4;
                        }

                        // Doubled pawn.
                        const bb_doubled = bitboards.file_bitboards[file] & our_pawns;
                        const is_doubled: bool = popcnt(bb_doubled) > 1;
                        if (is_doubled) {
                            ps -= 4;
                        }

                        // Isolated pawn.
                        const bb_isolated = bitboards.get_isolated_pawn_mask(sq) & our_pawns;
                        const is_isolated: bool = bb_isolated == 0;
                        if (is_isolated) {
                            ps -= 2;
                        }

                        // Protected pawn.
                        const is_protected: bool = !is_isolated and is_protected_by_pawn(pos, us, sq);
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

                        const mob: u64 = (pawn_moves & ~bb_us);
                        if (popcnt(mob) > 0) {
                            ps += 1;
                        }
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
                        if (is_protected_by_pawn(pos, us, sq)) {
                            ps += 2;
                        }

                        // Knight mobility.
                        const mob: u64 = knight_attacks & ~bb_us;
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

                        // Bishop mobility.
                        const mob: u64 = bishop_moves & ~bb_us;
                        if (popcnt(mob) > 6) {
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

                        // Rooks connected.
                        if (popcnt(rook_moves & our_rooks) > 0) {
                            ps += 4;
                        }

                        // Mobility
                        const mob: u64 = rook_moves & ~bb_us;
                        if (popcnt(mob) > 7) {
                            ps += 4;
                        }
                    },
                    .queen => {
                        //const our_queens: u64 = bb_current;
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
                        const mob: u64 = queen_moves & ~bb_us;
                        if (popcnt(mob) > 14) {
                            ps += 4;
                        }
                    },
                    .king => {
                        const king_moves: u64 = attacks.get_king_attacks(our_king_sq) & ~bb_us;

                        // Check.
                        // if (check and us.e == pos.to_move.e) {
                        //     add(k_score, Tables.in_check, us, piecetype, .in_check, tracking, sq);
                        // }

                        // // King area threat.
                        // if (phase == .Endgame) {
                        //     const att_king_area: u64 = attacks.get_king_attacks(our_king_sq) & their_king_area;
                        //     // EXPERIMENTAL
                        //     if (att_king_area != 0 and pos.to_move.e == us.e) {
                        //         king_attackers.* += 1;
                        //         ps += 1;
                        //     }
                        // }

                        // Open files to our king
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

                        // King mobility
                        if (king_moves == 0) {
                            ps -= 2;
                        }

                    },
                    else => {
                        unreachable;
                    },
                }

                // pesto + ps bonus.
                mg.* += (pair.mg + ps);
                eg.* += (pair.eg + ps);

            }
        }

        if (king_attackers.* > 2) {
            bonus.* += 20;
        }
    }

    // Taper sliding values.
    const w_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[0], pesto_scores_eg[0]);
    const b_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[1], pesto_scores_eg[1]);

    const w_score: Value =  material_scores[0] + w_pesto + bonus_scores[0];
    const b_score: Value = material_scores[1] + b_pesto + bonus_scores[1];

    var result = (w_score - b_score) + 20; // stm
    if (negate) {
         result = -result;
    }

    // TODO: not sure.
    //result = scale_towards_draw(result, pos.state.rule50);

    if (tracking) {
        //const tm: Value = if (pos.to_move.e == .white) 20 else -20;
        lib.io.debugprint(
            \\perspective : {t}
            \\material    : {} {}
            \\pestos      : {} {}
            \\bonus       : {} {}
            \\attackers   : {} {}
            \\result      : {}
            \\
            ,
            .{
                pos.to_move.e,
                material_scores[0], material_scores[1],
                w_pesto, b_pesto,
                bonus_scores[0], bonus_scores[1],
                king_attacker_count[0], king_attacker_count[1],
                result,
            }
        );
    }
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

pub fn is_draw_by_insufficient_material(pos: *const Position) bool
{
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
pub fn see_score(pos: *const Position, m: Move) Value
{
    const st: *const StateInfo = pos.state;
    const from: Square = m.from;
    const to: Square = m.to;
    const value_them = pos.get(to).value();
    const value_us = pos.get(from).value();
    // --> good capture. if (value_them - value_us > P.value()) return true;
    var gain: [24]Value = @splat(0);
    gain[0] = value_them;
    gain[1] = value_us - value_them;

    var depth: u8 = 1;
    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();
    var occupation = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var attackers: u64 = pos.get_all_attacks_to_for_occupation(occupation, to);
    var bb: u64 = 0;
    var side: Color = pos.to_move;

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
        bb = attackers & pos.knights(side) & ~st.pins[side.u];
        if (bb != 0) {
            depth += 1;
            gain[depth] = N.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight (cannot reveal more sliding attackers).
            continue :attackloop;
        }

        // Bishop.
        bb = attackers & pos.bishops(side) & ~st.pins_orthogonal[side.u];
        if (bb != 0) {
            depth += 1;
            gain[depth] = B.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 bishop.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Rook.
        bb = attackers & pos.rooks(side) & ~st.pins_diagonal[side.u];
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

/// Not used. Incomplete copy of see_value.
pub fn see(pos: *const Position, m: Move) bool {
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

    var side: Color = pos.to_move;
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
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Knight
        bb = atks & pos.knights(side);
        if (bb != 0) {
            balance = -balance - N.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            continue;
        }

        // Bishop
        bb = atks & pos.bishops(side);
        if (bb != 0) {
            balance = -balance - B.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Rook
        bb = atks & pos.rooks(side);
        if (bb != 0) {
            balance = -balance - R.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= attacks.get_rook_attacks(to, occ) & qr;
            continue;
        }

        // Queen
        bb = atks & pos.queens(side);
        if (bb != 0) {
            balance = -balance - Q.value();
            if (balance < 0) return (side.e != pos.to_move.e);
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
    return (side.e != pos.to_move.e);
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
    const bb_same_file: u64 = ~(bitboards.file_bitboards[sq.file()] & pos.pawns(us));
    const square_in_front: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
    const covered: u64 = attacks.get_pawn_attacks(square_in_front, us) & pos.pawns(us.opp()); // inversion trick
    return switch (us.e) {
        // .white => (masks.get_passed_pawn_mask(.WHITE, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0 and pos.get(square_in_front).is_empty(),
        // .black => (masks.get_passed_pawn_mask(.BLACK, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0 and pos.get(square_in_front).is_empty(),
        .white => (bitboards.get_passed_pawn_mask(.WHITE, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0,
        .black => (bitboards.get_passed_pawn_mask(.BLACK, sq) & bb_same_file & pos.pawns(us) != 0) and covered != 0,
    };
}

pub fn is_pinned(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins[us.u] & sq.to_bitboard() != 0;
}

pub fn is_pinned_diagonally(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins_diagonal[us.u] & sq.to_bitboard() != 0;
}

pub fn is_pinned_orthogonally(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins_orthogonal[us.u] & sq.to_bitboard() != 0;
}

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
            else => unreachable,
        };

        const eg: [*]const Value = switch (pc.e) {
            .pawn   => &eg_pawn_table,
            .knight => &eg_knight_table,
            .bishop => &eg_bishop_table,
            .rook   => &eg_rook_table,
            .queen  => &eg_queen_table,
            .king   => &eg_king_table,
            else => unreachable,
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
    const mg_pawn_table: [64]Value = .{
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    };

    const eg_pawn_table: [64]Value = .{
        0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100,  85,  67,  56,  53,  82,  84,
        32,  24,  13,   5,  -2,   4,  17,  17,
        13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
        13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0,
    };

    const mg_knight_table: [64]Value = .{
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73,  -41,  72,  36,  23,  62,   7,  -17,
        -47,   60,  37,  65,  84, 129,  73,   44,
        -9,    17,  19,  53,  37,  69,  18,   22,
        -13,    4,  16,  13,  28,  19,  21,   -8,
        -23,   -9,  12,  10,  19,  17,  25,  -16,
        -29,  -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    };

    const eg_knight_table: [64]Value = .{
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    };

    const mg_bishop_table: [64]Value = .{
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
         0,  15,  15,  15,  14,  27,  18,  10,
         4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    };

    const eg_bishop_table: [64]Value = .{
        -14, -21, -11,  -8, -7,  -9, -17, -24,
        -8,  -4,   7, -12, -3, -13,  -4, -14,
        2,  -8,   0,  -1, -2,   6,   0,   4,
        -3,   9,  12,   9, 14,  10,   3,   2,
        -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    };

    const mg_rook_table: [64]Value = .{
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    };

    const eg_rook_table: [64]Value = .{
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
        7,  7,  7,  5,  4,  -3,  -5,  -3,
        4,  3, 13,  1,  2,   1,  -1,   2,
        3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    };

    const mg_queen_table: [64]Value = .{
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,
    };

    const eg_queen_table: [64]Value = .{
        -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    };

    const mg_king_table: [64]Value = .{
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,  8, -28,  24,  14,
    };

    const eg_king_table: [64]Value = .{
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    };
};

/// A little 1 second time eval bench.
pub fn bench(pos: *const Position, pawn_tt: *tt.PawnTranspositionTable) void {
    var v: i64 = 0;
    var cnt: u64 = 0;
    var timer = utils.Timer.start();
    while (timer.elapsed_ms() < 1000) {
        v += evaluate(pos, pawn_tt, false);
        cnt += 1;
    }
    const t = timer.read();
    lib.io.debugprint("v {} cnt {} {}\n", .{v, cnt, funcs.nps(cnt, t)});
}

test "see"
{
    try lib.initialize();
    var pos: Position = .empty;

    const GOOD = true;
    const BAD = false;

    // Pawn exchange.
    try execute_see_test(&pos, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "d5e6", GOOD);
    // Queen takes covered knight.
    try execute_see_test(&pos, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "f3f6", BAD);
    // Pawn takes knight.
    try execute_see_test(&pos, "8/8/5b1k/4n3/K2P4/8/8/8 w - - 0 1", "d4e5", GOOD);
    // Losing queen for pawn.
    try execute_see_test(&pos, "8/kb6/2p5/3p4/4Q3/5B2/6QK/8 w - - 0 1", "e4d5", BAD);
    // Losing rook for knight.
    try execute_see_test(&pos, "3q3k/8/4p3/3n4/3R4/8/8/K2R4 w - - 0 1", "d4d5", BAD);
    // Bishop captures pawn. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4B3/5K2/8/8 w - - 0 1", "e4d5", BAD);
    // Pawn captures bishop. King involved.
    try execute_see_test(&pos, "8/8/2k5/3b4/4P3/5K2/8/8 w - - 0 1", "e4d5", GOOD);
    // Pawn exchange. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4P3/5K2/8/8 w - - 0 1", "e4d5", GOOD);
    // Winning a pawn. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4P3/6K1/6B1/8 w - - 0 1", "e4d5", GOOD);

    // Pinned rook capturing pawn.
    try execute_see_test(&pos, "3r4/8/1K1p4/3R4/3R4/8/1k6/6b1 w - - 0 1", "d5d6", BAD);
    // Pinned rook capturing knight.
    try execute_see_test(&pos, "3r4/8/1K1n4/3R4/3R4/8/1k6/6b1 w - - 0 1", "d5d6", BAD);

    // TODO: This should - in a perfect situation - reveal more pins after the first move.
    try execute_see_test(&pos, "7b/1k6/8/4p3/1r1P1PK1/5R2/5B2/8 w - - 0 1", "f4e5", GOOD);

    // Mutally pinned rooks.
    try execute_see_test(&pos, "4B3/3r4/2k5/K7/3p4/3R4/3R4/4b3 w - - 0 1", "d3d4", GOOD);
}

/// TODO: complete with see->bool and expected value.
fn execute_see_test(pos: *Position, fen: []const u8, move: []const u8, expected_good: bool) !void {
    var st: StateInfo = undefined;
    try pos.set(&st, fen);
    const m: Move = try pos.parse_move(move);
    const val: Value = see_score(pos, m);
    io.debugprint("{}\n", .{ val });
    if (expected_good)
        try std.testing.expect(val >= 0)
    else
        try std.testing.expect(val < 0);
}
