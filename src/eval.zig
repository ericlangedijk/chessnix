// zig fmt: off

//! --- NOT USED YET ---

// TODO: get rid of the comptime color. just negate the result if needed.
// TODO: make eval quite simple! depth will beat eval. Don't do too many "attack" stuff.
// TODO: remember to prevent doing useless moves in the endgame -> adjust score with drawcounter!
// TODO: if overwhelmingly winning (900??) skip positional stuff.
// TODO: draw by insufficient material.
// TODO: if (color_perspective.e == pos.to_move.e) score += 20;
// TODO: filter / taper / cap the feature scores by game phase.
// TODO: make pesto less biased and more symmetrical?
// TODO: rooks like open files
// TODO: backward pawn
// TODO: knights reward if never attack possible by pawn (outpost)
// TODO: knights reward if never attack possible by bishop (outpost + other color bishop)
// TODO: if endgame reward king close to enemyking

const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const squarepairs = @import("squarepairs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const data = @import("data.zig");
const masks = @import("masks.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const io = lib.io;
const wtf = lib.wtf;
const float = funcs.float;
const int = funcs.int;
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

const P = types.P;
const N = types.N;
const B = types.B;
const R = types.R;
const Q = types.Q;
const K = types.K;

/// Returns the evaluation from white perspective.
pub fn evaluate_abs(pos: *const Position, comptime tracking: bool) Value
{
    const v = evaluate(pos, tracking);
    return if (pos.to_move.e == .white) v else -v;
}

pub fn evaluate(pos: *const Position, comptime tracking: bool) Value
{
    const phase: GamePhase = pos.phase();
    switch (phase)
    {
        .Opening => return eval(pos, .Opening, tracking),
        .Midgame => return eval(pos, .Midgame, tracking),
        .Endgame => return eval(pos, .Endgame, tracking),
    }
    return evaluate(pos, tracking);
}

/// Evaluation from the perspective of the side to move.
fn eval(pos: *const Position, comptime phase: GamePhase, comptime tracking: bool) Value
{
    const Addition = fn (score: *Value, delta: Value, comptime piece: Piece, comptime feature: Feature, comptime track: bool, sq: ?Square) void;

    const Operation = struct
    {
        fn add(score: *Value, delta: Value, comptime piece: Piece, comptime feature: Feature, comptime track: bool, sq: ?Square) void
        {
            score.* += delta;
            if (track) print(score, delta, piece, feature, sq);
        }

        fn sub(score: *Value, delta: Value, comptime piece: Piece, comptime feature: Feature, comptime track: bool, sq: ?Square) void
        {
            score.* -= delta;
            if (track) print(score, delta, piece, feature, sq);
        }

        fn print(score: *Value, delta: Value, comptime piece: Piece, comptime feature: Feature, sq: ?Square) void
        {
            const s: []const u8 = if (sq) |q| q.to_string() else "  ";
            lib.io.debugprint("{s} {t:<8} {t:<32} {d:<6} -> {d:<6}\n", .{ s, piece.e, feature, delta, score.* });
        }
    };

    const non_pawn_material: Value = pos.non_pawn_material();
    const bb_all = pos.all();
    const negate: bool = pos.to_move.e == .black;

    const simple_score: Value = pos.values[Color.WHITE.u] - pos.values[Color.BLACK.u];

    // if (@abs(simple_score) > 500)
    // {
    //     return if (negate) -simple_score else simple_score;
    // }

    // Sliding values of the Pesto tables, which we taper at the end.
    var piece_square_score_mg: Value = 0;
    var piece_square_score_eg: Value = 0;

    // Positional scores per piece.
    var pawn_score: Value = 0;
    var knight_score: Value = 0;
    var bishop_score: Value = 0;
    var rook_score: Value = 0;
    var queen_score: Value = 0;
    var king_score: Value = 0;

    // Generic.
    var space: Value = 0;
    var mobility: Value = 0;
    var king_safety: Value = 0;

    inline for (Color.all) |us|
    {
        const add: Addition = comptime if (us.e == .white) Operation.add else Operation.sub;

        // Colored consts.
        const them = comptime us.opp();
        const bb_us = pos.by_color(us);
        const our_king_sq = pos.king_square(us);
        const check: bool = pos.state.checkers != 0;

        // Naive general space heuristic.
        if (phase != .Endgame)
        {
            space += 0;
            var bits: Value = 0;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_3));
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_4));
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_5)) * 2;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_6)) * 3;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_7)) * 3;
            add(&space, bits, Piece.NO_PIECE, .space, tracking, null);
        }

        inline for (PieceType.all) |piecetype|
        {
            // Piece consts.
            const piece: Piece = comptime Piece.make(piecetype, us);
            var is_first_piece: bool = true;
            const bb_current = pos.pieces(piecetype, us);
            var bb_running = bb_current;

            while (bb_running != 0) : (is_first_piece = false)
            {
                // Square consts.
                const sq: Square = funcs.pop_square(&bb_running);
                const file: u3 = sq.file();
                const rank: u3 = sq.rank();
                const relative_rank: u3 = funcs.relative_rank(us, rank);

                // Keep track of Pesto table scores.
                const pair = Tables.get_scorepair(us, piecetype, sq);
                add(&piece_square_score_mg, pair.mg, piece, .pesto_mg, false, sq);
                add(&piece_square_score_eg, pair.eg, piece, .pesto_eg, false, sq);

                // Individual piece on square.
                switch (piecetype.e)
                {
                    .pawn =>
                    {
                        const our_pawns: u64 = bb_current;

                        // EXPERIMENTAL
                        if (phase == .Opening and is_first_piece) {
                            const e: Value = popcnt_v(our_pawns & bitboards.bb_center) * 10;
                            add(&pawn_score, e, piece, .center_pawn, tracking, sq);
                        }

                        // Reward passed pawn.
                        if (is_passed_pawn(pos, us, sq)) {
                            const e: Value = Tables.passed_pawn_by_rank_scores[relative_rank];
                            add(&pawn_score, e, piece, .passed_pawn, tracking, sq);
                        }

                        // Punish doubled pawn.
                        const bb_doubled = bitboards.file_bitboards[file] & our_pawns;
                        if (popcnt(bb_doubled) > 1) {
                            const e: Value = Tables.doubled_pawn;
                            add(&pawn_score, e, piece, .doubled_pawn, tracking, sq);
                        }

                        // Punish isolated pawn.
                        const bb_isolated = masks.get_isolated_pawn_mask(sq) & our_pawns;
                        if (bb_isolated == 0) {
                            const e: Value = Tables.isolated_pawn;
                            add(&pawn_score, e, piece, .isolated_pawn, tracking, sq);
                        }
                        // Reward connected pawn.
                        else
                        {
                            //add(&pawn_score, 10);
                        }
                    },
                    .knight => {
                        // Reward knight is supported by a pawn.
                        if (is_supported_by_pawn(pos, us, sq))
                        {
                            const e: Value = Tables.supported_by_pawn;
                            add(&knight_score, e, piece, .supported_by_pawn, tracking, sq);
                        }

                        // Knight mobility
                        const knight_attacks: u64 = data.get_knight_attacks(sq) & ~bb_us;
                        add(&mobility, popcnt(knight_attacks), piece, .mobility, tracking, sq);
                        //add(&mobility, Tables.mobility(piecetype, phase, knight_attacks), piece, .mobility, tracking, sq);
                    },
                    .bishop => {
                        const our_bishops: u64 = bb_current;

                        // Reward bishop pair.
                        if (is_first_piece and our_bishops & bitboards.bb_black_squares & bitboards.bb_white_squares != 0)
                        {
                            const e = Tables.bishop_pair;
                            add(&bishop_score, e, piece, .bishop_pair, tracking, sq);
                        }

                        // Bishop mobility.
                        const bishop_attacks: u64 = data.get_bishop_attacks(sq, pos.all()) & ~bb_us;
                        add(&mobility, popcnt(bishop_attacks), piece, .mobility, tracking, sq);
                    },
                    .rook => {
                        const our_rooks: u64 = bb_current;
                        const rook_attacks: u64 = data.get_rook_attacks(sq, bb_all);

                        // Reward pigs on 7th rank if there are also enemy pawns or the king is cutoff on the 8th rank.
                        if (rank == funcs.relative_rank_7(us))
                        {
                            const their_pawns_on_7th = pos.pawns(them) & funcs.relative_rank_7_bitboard(them);
                            //const their_king_on_8th: u64 = pos.kings(them) & funcs.relative_rank_8_bitboard(us);
                            if (their_pawns_on_7th != 0)
                            {
                                const e = Tables.rook_on_seventh;
                                add(&rook_score, e, piece, .rook_on_seventh, tracking, sq);
                            }
                        }

                        // Reward connected rooks.
                        if (is_first_piece and popcnt(rook_attacks & our_rooks) > 0)
                        {
                            const e = Tables.rooks_connected;
                            add(&rook_score, e, piece, .rooks_connected, tracking, sq);
                        }

                        // Rook mobility
                        add(&mobility, popcnt(rook_attacks & ~bb_us), piece, .mobility, tracking, sq);
                    },
                    .queen => {
                        //const our_queens: u64 = bb_current;
                        const queen_attacks: u64 = data.get_queen_attacks(sq, bb_all);

                        //add(&queen_score, 0);
                        queen_score += 0;

                        // Queen mobility
                        add(&mobility, popcnt(queen_attacks & ~bb_us), piece, .mobility, tracking, sq);
                    },
                    .king => {
                        king_safety += 0; // not yet used.

                        const our_king_area: u64 = data.get_king_attacks(our_king_sq) & bb_us;

                        if (check)
                        {
                            const e: Value = Tables.in_check;
                            add(&king_score, e, piece, .in_check, tracking, sq);
                        }

                        // Punish when enemy pieces are indirectly pointing at our king.
                        if (!check)
                        {
                            var rooks_queens: u64 = data.get_rook_attacks(our_king_sq, 0) & pos.queens_rooks(them);
                            while (rooks_queens != 0)
                            {
                                const q: Square = funcs.pop_square(&rooks_queens);
                                const in_between = squarepairs.in_between_bitboard(q, sq) & bb_all;
                                const popcount = popcnt(in_between);
                                if (popcount <= 2)
                                {
                                    const e: Value = Tables.rooks_or_queens_staring_at_king;
                                    add(&king_score, e, piece, .rooks_or_queens_staring_at_king, tracking, sq);
                                }
                            }
                            var bishop_queens: u64 = data.get_bishop_attacks(our_king_sq, 0) & pos.queens_bishops(them);
                            while (bishop_queens != 0)
                            {
                                const q: Square = funcs.pop_square(&bishop_queens);
                                const in_between = squarepairs.in_between_bitboard(q, sq) & bb_all;
                                const popcount = popcnt(in_between);
                                if (popcount <= 2)
                                {
                                    const e: Value = Tables.bishops_or_queens_staring_at_king;
                                    add(&king_score, e, piece, .bishops_or_queens_staring_at_king, tracking, sq);
                                }
                            }
                        }

                        // Rather naive danger heuristic.
                        const bb = pos.attacks_by_for_occupation(them, bb_all) & our_king_area;
                        const danger: Value = popcnt(bb);
                        add(&king_score, -danger, piece, .attacks_close_to_king, tracking, sq);

                        // King protection
                        const pawn_protection = our_king_area & pos.pawns(us);
                        add(&king_score, popcnt(pawn_protection) * 4, piece, .king_protection_by_pawns, tracking, sq);

                        if (phase != .Endgame)
                        {
                            const piece_protection = our_king_area & pos.non_pawns(us);
                            add(&king_score, popcnt(piece_protection) * 2, piece, .king_protection_by_pieces, tracking, sq);
                        }
                    },
                    else =>
                    {
                        unreachable;
                    },
                }
            }
        }
    }

    // Taper sliding values.
    const pesto: Value = Tables.sliding_score(non_pawn_material, piece_square_score_mg, piece_square_score_eg);
    //const to_move_score: Value = 12;

    if (tracking)
    {
        lib.io.debugprint(
            \\perspective : {t}
            // \\to_move     : {}
            \\material    : {}
            \\pestotables : {}
            \\mobility    : {}
            \\space       : {}
            \\pawn        : {}
            \\knight      : {}
            \\bishop      : {}
            \\rook        : {}
            \\queen       : {}
            \\king        : {}
            \\
            ,
            .{ pos.to_move.e, simple_score, pesto, mobility, space, pawn_score, knight_score, bishop_score, rook_score, queen_score, king_score });
    }

    //const tm: Value = if (pos.to_move.e == .white) 12 else 0;

    const score: Value =
        simple_score +
        pesto +
        pawn_score + knight_score + bishop_score + rook_score + queen_score + king_score + king_safety +
        mobility +
        space;

    return if (negate) -score else score;
}

fn clamp(v: Value, limit: Value) Value
{
    //std.math.clamp(val: anytype, lower: anytype, upper: anytype)
    return @max(-limit, @min(limit, v));
}

pub fn is_draw_by_insufficient_material(pos: *const Position) bool
{
    // Only kings left.
    if (pos.non_pawn_material() == 0) return true;
    const w: Value = pos.materials[0];
    const b: Value = pos.materials[1];
    if (w + b <= types.material_bishop) return true;
    return false;
}

/// Static exchange evaluation. Quickly decide if a (capture) move is good or bad.
pub fn see_score(pos: *const Position, m: Move) Value
{
    const from: Square = m.from;
    const to: Square = m.to;
    const value_them = pos.get(to).value();
    const value_us = pos.get(from).value();
    // This is a good capture. For example pawn takes knight.
    //if (value_them - value_us > P.value()) return true;
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
    while (true)
    {
        attackers &= occupation;
        if (attackers == 0) break;
        side = side.opp();

        // Pawn.
        bb = attackers & pos.pawns(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = P.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue;
        }

        // Knight.
        bb = attackers & pos.knights(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = N.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight (knight move cannot reveal more sliding attackers to the same square).
            continue;
        }

        // Bishop.
        bb = attackers & pos.bishops(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = B.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 bishop
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue;
        }

        // Rook.
        bb = attackers & pos.rooks(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = R.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook
            attackers |= (data.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue;
        }

        // Queen.
        bb = attackers & pos.queens(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = Q.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 queen
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (data.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue;
        }

        bb = attackers & pos.kings(side);
        if (bb != 0) {
            break;
            // UNDECIDED
            // // When the king captures and there are still opponent attackers, we return a flipped result.
            // // Return true if there are zero attacks to our king left.
            // funcs.clear_square(&occupation, funcs.first_square(bb));
            // bb = pos.get_attackers_to_for_occupation(to_sq, occupation) & occupation & pos.by_side(side.opp());
            // return switch (us.e == side.e)
            // {
            //     false => bb == 0,
            //     true => bb != 0,
            // };
        }
        break;
    }

    // Bubble up the score
    depth -= 1;
    while (depth > 0) : (depth -= 1)
    {
        if (gain[depth] > -gain[depth - 1])
        {
            gain[depth - 1] = -gain[depth];
        }
    }
    return gain[0];
}

/// TODO: add threshold.
pub fn see(pos: *const Position, m: Move) bool
{
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

    while (true)
    {
        atks &= occ;
        if (atks == 0) break;
        side = side.opp(); // next capturer

        // Pawn
        bb = atks & pos.pawns(side);
        if (bb != 0)
        {
            balance = -balance - P.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_bishop_attacks(to, occ) & qb;
            continue;
        }
        // Knight
        bb = atks & pos.knights(side);
        if (bb != 0)
        {
            balance = -balance - N.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            continue;
        }
        // Bishop
        bb = atks & pos.bishops(side);
        if (bb != 0)
        {
            balance = -balance - B.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_bishop_attacks(to, occ) & qb;
            continue;
        }
        // Rook
        bb = atks & pos.rooks(side);
        if (bb != 0)
        {
            balance = -balance - R.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_rook_attacks(to, occ) & qr;
            continue;
        }
        // Queen
        bb = atks & pos.queens(side);
        if (bb != 0)
        {
            balance = -balance - Q.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= (data.get_bishop_attacks(to, occ) & qb);
            atks |= (data.get_rook_attacks(to, occ) & qr);
            continue;
        }

        // King ends the sequence
        if ((atks & pos.kings(side)) != 0) break;
        break;
    }

    // No early fail: if the last side to *have the move* is the opponent, our capture stands.
    return (side.e != pos.to_move.e);
}

pub fn is_supported_by_pawn(pos: *const Position, comptime us: Color, sq: Square) bool
{
    const them = comptime us.opp();
    return data.get_pawn_attacks(sq, them) & pos.pawns(us) != 0; // using inversion trick
}

pub fn is_passed_pawn(pos: *const Position, comptime us: Color, sq: Square) bool
{
    return switch (us.e)
    {
        .white => masks.get_passed_pawn_mask(.WHITE, sq) & pos.all_pawns() == 0,
        .black => masks.get_passed_pawn_mask(.BLACK, sq) & pos.all_pawns() == 0,
    };
}

pub fn is_pinned(pos: *const Position, sq: Square) bool
{
    return pos.pins & sq.to_bitboard() != 0;
}

const Feature = enum
{
    // Pawn.
    passed_pawn,
    doubled_pawn,
    isolated_pawn,
    center_pawn,

    // Bishop.
    bishop_pair,

    // Rook.
    rook_on_seventh,
    rooks_connected,

    // King.
    rooks_or_queens_staring_at_king,
    bishops_or_queens_staring_at_king,
    attacks_close_to_king,
    king_protection_by_pawns,
    king_protection_by_pieces,

    // Generic.
    pesto_mg,
    pesto_eg,

    supported_by_pawn,
    space,
    mobility,
    in_check,
};

pub const Tables = struct
{
    //fn get_index(comptime us: Color, sq: Square) u6
    fn get_index(us: Color, sq: Square) u6
    {
        return switch(us.e)
        {
            .white => sq.u ^ 56,
            .black => sq.u,
        };
    }

    pub fn get_move_ordering_score(non_pawn_material: Value, us: Color, m: Move, pt: PieceType) Value
    {
        const pair_from = get_scorepair(us, pt, m.from);
        const pair_to = get_scorepair(us, pt, m.to);
        const score_from = sliding_score(non_pawn_material, pair_from.mg, pair_from.eg);
        const score_to = sliding_score(non_pawn_material, pair_to.mg, pair_to.eg);
        return score_to - score_from;
    }

    /// Returns the table values for opening and endgame, between which we slide the final score.
    //fn get_scorepair(comptime us: Color, comptime pc: PieceType, sq: Square) struct { mg: Value, eg: Value}
    pub fn get_scorepair(us: Color, pc: PieceType, sq: Square) struct { mg: Value, eg: Value}
    {
        const mg: [*]const Value = switch (pc.e)
        {
            .pawn   => &mg_pawn_table,
            .knight => &mg_knight_table,
            .bishop => &mg_bishop_table,
            .rook   => &mg_rook_table,
            .queen  => &mg_queen_table,
            .king   => &mg_king_table,
            else => unreachable,
        };

        const eg: [*]const Value = switch (pc.e)
        {
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

    pub fn sliding_score(non_pawn_material: Value, opening: Value, endgame: Value) Value
    {
        const max: i32 = comptime types.max_material_without_pawns;
        const phase: i32 = @min(non_pawn_material, max);
        return @truncate(@divTrunc(opening * phase + endgame * (max - phase), max));
    }

    // Pesto tables.
    const mg_pawn_table: [64]Value =
    .{
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    };

    const eg_pawn_table: [64]Value =
    .{
        0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100,  85,  67,  56,  53,  82,  84,
        32,  24,  13,   5,  -2,   4,  17,  17,
        13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
        13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0,
    };

    const mg_knight_table: [64]Value =
    .{
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73,  -41,  72,  36,  23,  62,   7,  -17,
        -47,   60,  37,  65,  84, 129,  73,   44,
        -9,    17,  19,  53,  37,  69,  18,   22,
        -13,    4,  16,  13,  28,  19,  21,   -8,
        -23,   -9,  12,  10,  19,  17,  25,  -16,
        -29,  -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    };

    const eg_knight_table: [64]Value =
    .{
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    };

    const mg_bishop_table: [64]Value =
    .{
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
         0,  15,  15,  15,  14,  27,  18,  10,
         4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    };

    const eg_bishop_table: [64]Value =
    .{
        -14, -21, -11,  -8, -7,  -9, -17, -24,
        -8,  -4,   7, -12, -3, -13,  -4, -14,
        2,  -8,   0,  -1, -2,   6,   0,   4,
        -3,   9,  12,   9, 14,  10,   3,   2,
        -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    };

    const mg_rook_table: [64]Value =
    .{
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    };

    const eg_rook_table: [64]Value =
    .{
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
        7,  7,  7,  5,  4,  -3,  -5,  -3,
        4,  3, 13,  1,  2,   1,  -1,   2,
        3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    };

    const mg_queen_table: [64]Value =
    .{
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,
    };

    const eg_queen_table: [64]Value =
    .{
        -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    };

    const mg_king_table: [64]Value =
    .{
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    };

    const eg_king_table: [64]Value =
    .{
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    };

    // Pawn.
    const passed_pawn_by_rank_scores: [8]Value = .{ 0, 15, 15, 30, 50, 80, 120, 0 }; // .{ 0, 2, 4, 6, 8, 20, 40, 0 };
    const doubled_pawn: Value = -5;
    const isolated_pawn: Value = -10;

    // Bishop.
    const bishop_pair: Value = 20;

    // Generic.
    const supported_by_pawn: Value = 10;

    // Rook.
    const rook_on_seventh: Value = 20;
    const rooks_connected: Value = 20;

    // King.
    const in_check: Value = -20;
    const rooks_or_queens_staring_at_king: Value = -5;
    const bishops_or_queens_staring_at_king: Value = -5;

    pub fn mobility(comptime pt: PieceType, comptime phase: GamePhase, mobility_bitboard: u64) Value
    {
        const m: f32 = switch (phase)
        {
            .Opening => 0.8,
            .Midgame => 1.2,
            .Endgame => 0.2,
        };

        const bitcount: Value = popcnt(mobility_bitboard);
        switch (pt.e)
        {
            .pawn =>
            {
                return funcs.mul(bitcount, m);
            },
            .knight =>
            {
                return funcs.mul(bitcount, m);
            },
            .bishop =>
            {
                return bitcount;
            },
            .rook =>
            {
                return bitcount;
            },
            .queen =>
            {
                return bitcount;
            },
            .king =>
            {
                return bitcount;
            },
            else =>
            {
                unreachable;
            }
        }
        return 0;
    }

};

/// A little 1 second time eval. around 14 million per second.
pub fn bench(pos: *const Position) void
{
    if (pos.to_move.e == .black)
    {
        var v: i64 = 0;
        var cnt: u64 = 0;
        var timer = utils.Timer.start();
        while (timer.elapsed_ms() < 1000)
        {
            v += evaluate(pos, Color.WHITE, false);
            cnt += 1;
        }
        const t = timer.read();
        lib.io.debugprint("v {} cnt {} {}\n", .{v, cnt, funcs.nps(cnt, t)});
    }
    else
    {
        var v: i64 = 0;
        var cnt: u64 = 0;
        var timer = utils.Timer.start();
        while (timer.elapsed_ms() < 1000)
        {
            v += evaluate(pos, Color.BLACK, false);
            cnt += 1;
        }
        const t = timer.read();
        lib.io.debugprint("v {} cnt {} {}\n", .{v, cnt, funcs.nps(cnt, t)});
    }
}

test "see"
{
    try lib.initialize();

    var st: StateInfo = undefined;
    var pos: Position = .empty;

    // Good.
    {
        try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        const m: Move = .create(.D5, .E6);
        const good: bool = see(&pos, m);
        try std.testing.expectEqual(true, good);
        const val: Value = see_score(&pos, m);
        try std.testing.expectEqual(0, val);
    }

    // Good.
    {
        try pos.set(&st, "8/8/5b1k/4n3/K2P4/8/8/8 w - - 0 1");
        const m: Move = .create(.D4, .E5);
        const good: bool = see(&pos, m);
        try std.testing.expectEqual(true, good);
        const val: Value = see_score(&pos, m);
        try std.testing.expectEqual(PieceType.KNIGHT.value() - PieceType.PAWN.value(), val);
    }

    // Bad
    {
        try pos.set(&st, "8/kb6/2p5/3p4/4Q3/5B2/6QK/8 w - - 0 1");
        const m: Move = .create(.E4, .D5);
        const good: bool = see(&pos, m);
        try std.testing.expectEqual(false, good);
        const val: Value = see_score(&pos, m);
        try std.testing.expectEqual(-750, val);
    }

    // Bad
    {
        try pos.set(&st, "3q3k/8/4p3/3n4/3R4/8/8/K2R4 w - - 0 1");
        const m: Move = .create(.D4, .D5);
        const good: bool = see(&pos, m);
        try std.testing.expectEqual(false, good);
        const val: Value = see_score(&pos, m);
        try std.testing.expectEqual(PieceType.KNIGHT.value() - PieceType.ROOK.value(), val);
    }
}