// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const data = @import("data.zig");
const funcs = @import("funcs.zig");

const wtf = lib.wtf;

const Value = types.Value;
const Float = types.Float;

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const Position = position.Position;

const float = funcs.float;
const int = funcs.int;

/// Passed to all eval methods.
const Params = struct
{
    pos: *const Position,
    /// We cache the material value here.
    non_pawn_material: Value,
};

/// Debug only.
pub fn lazy_evaluate(pos: *const Position) Value
{
    return switch (pos.to_move.e)
    {
        .white => evaluate(pos, Color.WHITE),
        .black => evaluate(pos, Color.BLACK),
    };
}

/// `us` must be the color to move.
pub fn evaluate(pos: *const Position, comptime us: Color) Value
{
    const them = comptime us.opp();

    const ep: Params =
    .{
        .pos = pos,
        .non_pawn_material = pos.non_pawn_material(),
    };

    var score: Value = 0;

    // if (us.e == pos.to_move.e) score += 20;

    score += eval_material(&ep, us);
    score -= eval_material(&ep, them);
    score += eval_pawns(&ep, us);
    score -= eval_pawns(&ep, them);
    score += eval_knights(&ep, us);
    score -= eval_knights(&ep, them);
    score += eval_bishops(&ep, us);
    score -= eval_bishops(&ep, them);
    score += eval_rooks(&ep, us);
    score -= eval_rooks(&ep, them);
    score += eval_queens(&ep, us);
    score -= eval_queens(&ep, them);
    score += eval_king(&ep, us);
    score -= eval_king(&ep, them);

    return score;
}

fn eval_material(e: *const Params, comptime us: Color) Value
{
    const v: Value = e.pos.values[us.u];
    return v;
}

fn eval_pawns(e: *const Params, comptime us: Color) Value
{
    const bb_pawns = e.pos.pawns(us);
    var score: Value = 0;
    var bb = bb_pawns;

    while (bb != 0)
    {
        const sq: Square = funcs.pop_square(&bb);

        // Piece on square value.
        score += pc_sq(us, PieceType.PAWN, sq, e.non_pawn_material);

        // Reward passed pawns.
        if (funcs.is_passed_pawn(e.pos, us, sq))
        {
            score += 20;
        }

        // Punish doubled pawns.
        {
            const bb_file = bitboards.file_bitboards[sq.file()] & e.pos.pawns(us);
            if (@popCount(bb_file) > 1)
            {
                score -= 10;
            }
        }
    }
    return score;
}

fn eval_knights(e: *const Params, comptime us: Color) Value
{
    var s: Value = 0;
    var bb = e.pos.knights(us);

    while (bb != 0)
    {
        const sq: Square = funcs.pop_square(&bb);

        // Piece on square value.
        s += pc_sq(us, PieceType.KNIGHT, sq, e.non_pawn_material);

        // Reward knight is supported by a pawn.
        if (funcs.is_supported_by_pawn(e.pos, us, sq))
        {
            s += 20;
        }

        // Reward if never attack possible by pawn
        // Reward if never attack possible by bishop

        // Mobility
        const attacks: u64 = data.get_knight_attacks(sq) & ~e.pos.by_color(us);
        s += @popCount(attacks);
    }

    return s;
}

fn eval_bishops(e: *const Params, comptime us: Color) Value
{
    var s: Value = 0;
    const bb_bishops = e.pos.bishops(us);
    var bb = bb_bishops;
    while (bb != 0)
    {
        const sq: Square = funcs.pop_square(&bb);

        // Piece on square value.
        s += pc_sq(us, PieceType.BISHOP, sq, e.non_pawn_material);

        // Mobility.
        {
            const attacks: u64 = data.get_bishop_attacks(sq, e.pos.all()) & ~e.pos.by_color(us);
            const v = @popCount(attacks);
            s += v;
        }
    }

    // Reward bishop pair.
    const bishop_pair = @popCount(bb_bishops & bitboards.bb_black_squares) + @popCount(bb_bishops & bitboards.bb_white_squares);
    if (bishop_pair >= 2)
    {
        s += 20;
    }
    return s;
}

fn eval_rooks(e: *const Params, comptime us: Color) Value
{
    const them: Color = comptime us.opp();
    var s: Value = 0;
    const bb_rooks = e.pos.rooks(us);
    var bb = bb_rooks;

    while (bb != 0)
    {
        const sq: Square = funcs.pop_square(&bb);

        // Piece on square value.
        s += pc_sq(us, PieceType.ROOK, sq, e.non_pawn_material);

        const bb_not_us: u64 = ~e.pos.by_color(us);
        const attacks: u64 = data.get_rook_attacks(sq, e.pos.all());

        // Mobility.
        s += @popCount(attacks & bb_not_us);

        // Reward connected rooks.
        if (@popCount(attacks & bb_rooks) > 0)
        {
            s += 20;
        }

        // Reward pigs on 7th rank if there are also enemy pawns or the king is cutoff on the 8th rank.
        if (sq.rank() == funcs.relative_rank_7(us))
        {
            const their_pawns_on_7th = e.pos.pawns(them) & funcs.relative_rank_7_bitboard(them);
            const their_king_on_8th: u64 = e.pos.kings(them) & funcs.relative_rank_8_bitboard(us);
            if (their_pawns_on_7th | their_king_on_8th != 0)
            {
                s += 20;
            }
        }
    }
    return s;
}

fn eval_queens(e: *const Params, comptime us: Color) Value
{
    var s: Value = 0;
    const bb_queens = e.pos.queens(us);
    var bb = bb_queens;

    while (bb != 0)
    {
        const sq: Square = funcs.pop_square(&bb);

        // Piece on square value.
        s += pc_sq(us, PieceType.QUEEN, sq, e.non_pawn_material);

        // Mobility.
        const bb_not_us: u64 = ~e.pos.by_color(us);
        const attacks: u64 = data.get_rook_attacks(sq, e.pos.all());
        const v = @popCount(attacks & bb_not_us);
        s += v;
    }
    return s;
}

fn eval_king(e: *const Params, comptime us: Color) Value
{
    const them: Color = comptime us.opp();
    var s: Value = 0;
    const king_sq = e.pos.king_square(us);
    var bb: u64 = 0;

    // Piece on square value.
    s += pc_sq(us, PieceType.KING, king_sq,e.non_pawn_material);

    // punish pieces pointing at us, disregarding what is in between.
    bb = data.get_rook_attacks(king_sq, 0) & e.pos.queens_rooks(them);
    const v = @popCount(bb); // this is the number of queens or rooks indirectly pointing at the king.
    s -= v;

    // todo enemy knights close by? enemy pieces pointing closeby (3x3)

    // let bb: u64 = calc_rook_attacks(king_sq.idx(), 0) & pos.queens_rooks_c::<THEM>();
    // score -= ((bb.popcount() as i16) * 2);

    // todo castling + pawn / piece protection.
    // const piece_protection: u64 = data.get_king_attacks(king_sq) & e.pos.by_color(us);


    return s;
}

/// Static exchange evaluation. Quickly decide if a (capture) move is good or bad.
pub fn see(pos: *const Position, m: Move) bool
{
    if (m.movetype == .promotion) return true;

    const us: Color = pos.to_move;
    var side: Color = us;
    const from_sq = m.from;
    const to_sq = m.to;
    const value_them = pos.get(to_sq).value();
    const value_us = pos.get(from_sq).value();
    // This is a good capture. For example pawn takes knight.
    if (value_them - value_us > 100) return true;
    var gain: [24]Value = @splat(0);
    gain[0] = value_them;
    gain[1] = value_us - value_them;
    var depth: u8 = 1;
    const queens_or_bishops = pos.all_queens_bishops();
    const queens_or_rooks = pos.all_queens_rooks();
    var occupation = pos.all() ^ to_sq.to_bitboard() ^ from_sq.to_bitboard();
    var attackers: u64 = pos.get_attackers_to_for_occupation(to_sq, occupation);
    var bb: u64 = 0;

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
            gain[depth] = PieceType.PAWN.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn
            attackers |= data.get_bishop_attacks(to_sq, occupation) & queens_or_bishops; // reveal next diagonal attacker.
            continue;
        }

        // Knight.
        bb = attackers & pos.bishops(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = PieceType.KNIGHT.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight
            //attackers |= data.get_knight_attacks(to_sq, occupation) & queens_or_rooks; // reveal next straight attacker.
            // Note: a knight move cannot reveal more sliding attackers to the same square.
            continue;
        }

        // Bishop.
        bb = attackers & pos.bishops(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = PieceType.BISHOP.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook
            attackers |= data.get_bishop_attacks(to_sq, occupation) & queens_or_rooks; // reveal next diagonal attacker.
            continue;
        }

        // Rook.
        bb = attackers & pos.rooks(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = PieceType.ROOK.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook
            attackers |= data.get_rook_attacks(to_sq, occupation) & queens_or_rooks; // reveal next straight attacker.
            continue;
        }

        // Queen.
        bb = attackers & pos.queens(side);
        if (bb != 0)
        {
            depth += 1;
            gain[depth] = PieceType.QUEEN.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 queen
            attackers |= data.get_bishop_attacks(to_sq, occupation) & queens_or_bishops; // reveal next diagonal attacker.
            attackers |= data.get_rook_attacks(to_sq, occupation) & queens_or_rooks; // reveal next straight attacker.
            continue;
        }

        bb = attackers & pos.kings(side);
        if (bb != 0)
        {
            break;

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
    return gain[0] >= 0;
}

/// Piece on square value for a certain phase of the game, using the materials value.
pub fn pc_sq(comptime us: Color, comptime pt: PieceType, sq: Square, pos_material: Value) Value
{
    const pair = PestoTables.get_scorepair(us, pt, sq);
    return sliding_score(pos_material, pair.mg, pair.eg);
}

/// TODO: check check check.
fn sliding_score(non_pawn_material: Value, opening: Value, endgame: Value) Value
{
    // const maximum: Value = comptime types.max_material_without_pawns;
    // const range: Value = endgame - opening;
    // const min: Value = @min(non_pawn_material, maximum);
    // const step: Float = (float(min) / float(maximum)) * float(range);
    // return int(float(endgame) - step);

    const maximum: i32 = comptime types.max_material_without_pawns;
    const phase: i32 = @min(non_pawn_material, maximum);
    return @truncate(@divTrunc(opening * phase + endgame * (maximum - phase), maximum));
}

const PestoTables = struct
{
    fn get_index(comptime us: Color, sq: Square) u6
    {
        return switch(us.e)
        {
            .white => sq.u ^ 56,
            .black => sq.u,
        };
    }

    fn get_scorepair(comptime us: Color, comptime pc: PieceType, sq: Square) struct { mg: Value, eg: Value}
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

    // TODO: make less biased and symmetrical

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
};
