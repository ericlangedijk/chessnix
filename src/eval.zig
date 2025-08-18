// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const data = @import("data.zig");
const funcs = @import("funcs.zig");

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

/// Piece Square score for opening and endgame.\
/// During evaluation we slide between the values depending on the game phase, which is deduced from the material.
const ScorePair = struct { opening: i16, endgame: i16 };
/// A compressed piece square table.
const HalfTable = [32]ScorePair;

pub const MatingGame = enum(u16)
{
    //B
    R =  PieceType.ROOK.material(),
    RR = PieceType.ROOK.material() * 2,
};

/// Returns "quiet" evaluation. This should be a position without captures possible.
pub fn evaluate(pos: *const Position, comptime us: Color) Value
{
    return eval(pos, us, false);
}

pub fn eval(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    // TODO: if material is high and high depth exit early?
    const them = comptime us.opp();
    var s: Value = 0;
    s += eval_value(pos, us, output);
    s -= eval_value(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_pawns(pos, us, output);
    s -= eval_pawns(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_knights(pos, us, output);
    s -= eval_knights(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_bishops(pos, us, output);
    s -= eval_bishops(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_rooks(pos, us, output);
    s -= eval_rooks(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_queens(pos, us, output);
    s -= eval_queens(pos, them, output); if (output) std.debug.print("VAL {}\n", .{s});
    s += eval_king(pos, us, output);
    s -= eval_king(pos, them, output);
    if (output) std.debug.print("eval = {}\n", .{s});
    return s;
}

/// The raw sum of pieces values.
fn eval_value(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    const v: Value = pos.values[us.u];
    if (output) std.debug.print("eval_material {s} value = {}, \n", .{@tagName(us.e), v });
    return v;
}

fn eval_pawns(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    const bb_pawns = pos.pawns(us);
    var s: Value = 0;
    var bb = bb_pawns;
    var sq: Square = .zero;
    var v: Value = 0;

    while (bb != 0)
    {
        sq = funcs.pop_square(&bb);

        // Piece on square value.
        v = pc_sq(us, PieceType.PAWN, sq, pos.material());
        s += v;
        if (output) std.debug.print("eval_pawns {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // Reward passed pawns.
        if (funcs.is_passed_pawn(pos, us, sq))
        {
            v = 20;
            if (output) std.debug.print("eval_pawns {s} (pass) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
            s += v;
        }

        // Punish doubled pawns.
        var bb_file = bitboards.file_bitboards[sq.file()] & pos.pawns(us);
        bb_file &= ~sq.to_bitboard();
        v = @popCount(bb_file);
        if (v > 0)
        {
            s -= 10;
            if (output) std.debug.print("eval_pawns {s} (doubled) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
        }
        //std.debug.print("doubled pawn {s}", .{sq.to_string()});

    }

    return s;
}

fn eval_knights(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    var s: Value = 0;
    var bb = pos.knights(us);
    var sq: Square = .zero;
    var v: Value = 0;

    while (bb != 0)
    {
        sq = funcs.pop_square(&bb);

        // Piece on square value.
        v = pc_sq(us, PieceType.KNIGHT, sq, pos.material());
        s += v;
        if (output) std.debug.print("eval_knights {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // Reward support by a pawn.
        v = if (funcs.is_supported_by_pawn(pos, us, sq)) 20 else 0;
        s += v;
        if (output) std.debug.print("eval_knights {s} (supp) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // Mobility
        const attacks: u64 = data.get_knight_attacks(sq) & ~pos.by_side(us);
        v = @popCount(attacks);
        s += v;
        if (output) std.debug.print("eval_knights {s}  (mob) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
    }
    return s;
}

fn eval_bishops(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    var s: Value = 0;
    const bb_bishops = pos.bishops(us);
    var bb = bb_bishops;
    var sq: Square = .zero;
    var v: Value = 0;

    while (bb != 0)
    {
        sq = funcs.pop_square(&bb);
        // Piece on square value.
        v = pc_sq(us, PieceType.BISHOP, sq, pos.material());
        s += v;
        if (output) std.debug.print("eval_bishops {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
        // Mobility.
        const attacks: u64 = data.get_bishop_attacks(sq, pos.all()) & ~pos.by_side(us);
        v = @popCount(attacks);
        s += v;
        if (output) std.debug.print("eval_bishops {s}  (mob) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
    }

    // Reward bishop pair.
    v = @popCount(bb_bishops & bitboards.bb_black_squares) + @popCount(bb_bishops & bitboards.bb_white_squares);
    if (v >= 2)
    {
        s += 20;
        if (output) std.debug.print("eval_bishops {s} bishoppair, result = {}\n", .{@tagName(us.e), s });
    }
    return s;
}

fn eval_rooks(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    var s: Value = 0;
    const bb_rooks = pos.rooks(us);
    var bb = bb_rooks;
    var sq: Square = .zero;
    var v: Value = 0;

    while (bb != 0)
    {
        sq = funcs.pop_square(&bb);
        // Piece on square value.
        v = pc_sq(us, PieceType.ROOK, sq, pos.material());
        s += v;
        if (output) std.debug.print("eval_rooks {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // Mobility.
        const bb_not_us: u64 = ~pos.by_side(us);
        const attacks: u64 = data.get_rook_attacks(sq, pos.all());
        v = @popCount(attacks & bb_not_us);
        s += v;
        if (output) std.debug.print("eval_rooks {s}  (mob) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // if (us.e == .white)
        // {
        // funcs.print_bitboard(attacks);
        // funcs.print_bitboard(bb_rooks);
        // }

        // Reward connected rooks.
        if (@popCount(attacks & bb_rooks) > 0)
        {
            v = 20;
            s += v;
            if (output) std.debug.print("connected rooks {}\n", .{v});
        }
    }
    return s;
}

fn eval_queens(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    var s: Value = 0;
    const bb_queens = pos.queens(us);
    var bb = bb_queens;
    var sq: Square = .zero;
    var v: Value = 0;

    while (bb != 0)
    {
        sq = funcs.pop_square(&bb);
        // Piece on square value.
        v = pc_sq(us, PieceType.QUEEN, sq, pos.material());
        s += v;
        if (output) std.debug.print("eval_queens {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });

        // Mobility.
        const bb_not_us: u64 = ~pos.by_side(us);
        const attacks: u64 = data.get_rook_attacks(sq, pos.all());
        v = @popCount(attacks & bb_not_us);
        s += v;
        if (output) std.debug.print("eval_queens {s}  (mob) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), sq.to_string(), v, s });
    }
    return s;
}

fn eval_king(pos: *const Position, comptime us: Color, comptime output: bool) Value
{
    const them: Color = comptime us.opp();
    var s: Value = 0;
    const king_sq = pos.king_square(us);
    var v: Value = 0;
    var bb: u64 = 0;

    // Piece on square value.
    v = pc_sq(us, PieceType.KING, king_sq, pos.material());
    s += v;
    if (output) std.debug.print("eval_king {s} (pc_sq) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), king_sq.to_string(), v, s });


    // punish pieces pointing at us, disregarding what is in between (remember this is a quiet position: there are no direct attacks)
    bb = data.get_rook_attacks(king_sq, 0) & pos.queens_rooks(them);
    v = @popCount(bb); // this is the number of queens or rooks indirectly pointing at the king.
    if (output) std.debug.print("eval_king {s} (indirect attacks) sq = {s}, value = {}, result = {}, \n", .{@tagName(us.e), king_sq.to_string(), v, s });
    s -= v;

    // todo enemy knights close by? enemy pieces pointing closeby (3x3)

    // let bb: u64 = calc_rook_attacks(king_sq.idx(), 0) & pos.queens_rooks_c::<THEM>();
    // score -= ((bb.popcount() as i16) * 2);

    // todo castling + protection.

    return s;
}

/// Static exchange evaluation. Quickly decide if a (capture) move is ok.
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
            gain[depth] = PieceType.BISHOP.value() - gain[depth - 1];
            if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight
            attackers |= data.get_bishop_attacks(to_sq, occupation) & queens_or_rooks; // reveal next straight attacker.
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
            // When the king captures and there are still opponent attackers, we return a flipped result.
            // Return true if there are zero attacks to our king left.
            funcs.clear_square(&occupation, funcs.first_square(bb));
            bb = pos.get_attackers_to_for_occupation(to_sq, occupation) & occupation & pos.by_side(side.opp());
            return switch (us.e == side.e)
            {
                false => bb == 0,
                true => bb != 0,
            };
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

pub fn pc_sq(comptime us: Color, comptime pt: PieceType, sq: Square, pos_material: Value) Value
{
    const pair: ScorePair = get_scorepair(us, pt, sq);
    //return sliding_score(@floatFromInt(pos_material), @floatFromInt(pair.opening), @floatFromInt(pair.endgame));
    return sliding_score2(pos_material, pair.opening, pair.endgame);
}

/// Calculates a sliding number between opening and endgame.
fn sliding_score(pos_material: f32, opening: f32, endgame: f32) Value
{
    //std.debug.print("material {d:.1}, opening {d:.1} endgame {d:.1}, ", .{pos_material, opening, endgame});
    const maximum: f32 = comptime @floatFromInt(types.max_material_value_threshold);
    //std.debug.print("max {d:.1}, ", .{maximum});
    const range: f32 = endgame - opening;
    //std.debug.print("range {d:.1}, ", .{range});
    const min: f32 = @min(pos_material, maximum);
    //std.debug.print("mt {d:.1}, ", .{mt});
    const step: f32 = (min / maximum) * range;
    //std.debug.print("step {d:.1}, ", .{step});
    //std.debug.print("result {d:.1} -> ", .{endgame - step});

// const t: f32 = mt / maximum;
// const value: f32 = opening + t * (endgame - opening); // linear interpolation
// return @intFromFloat(value);

    return @intFromFloat(endgame - step);
}

fn sliding_score2(pos_material: Value, opening: Value, endgame: Value) Value
{
    //std.debug.print("material {d:.1}, opening {d:.1} endgame {d:.1}, ", .{pos_material, opening, endgame});
    const maximum: Value = comptime types.max_material_value_threshold;

    //std.debug.print("max {d:.1}, ", .{maximum});
    const range: Value = endgame - opening;

    //std.debug.print("range {d:.1}, ", .{range});
    const min: Value = @min(pos_material, maximum);
    //std.debug.print("mt {d:.1}, ", .{mt});

    const step: Float = (float(min) / float(maximum)) * float(range);
    //std.debug.print("step {d:.1}, ", .{step});
    //std.debug.print("result {d:.1} -> ", .{endgame - step});

// const t: f32 = mt / maximum;
// const value: f32 = opening + t * (endgame - opening); // linear interpolation
// return @intFromFloat(value);

    return int(float(endgame) - step);
}


pub fn get_scorepair(comptime us: Color, comptime pt: PieceType, sq: Square) ScorePair
{
    const idx = get_half_table_idx(us, sq);

    return switch (pt.e)
    {
        .pawn => pawn_table[idx],
        .knight => knight_table[idx],
        .bishop => bishop_table[idx],
        .rook => rook_table[idx],
        .queen => queen_table[idx],
        .king => king_table[idx],
        else => unreachable,
    };
}

fn get_half_table_idx(comptime us: Color, sq: Square) u8
{
    return switch(us.e)
    {
        .white => white_indexer[sq.u],
        .black => black_indexer[sq.u],
    };
}


fn sp(comptime opening: Value, comptime endgame: Value) ScorePair
{
    return .{ .opening = opening, .endgame = endgame };
}

const pawn_table: HalfTable =
.{
    sp(   0,  90 ), sp(   0,  90 ), sp(   0,  90 ), sp(   0,  90 ),
    sp(  10,  80 ), sp(  12,  80 ), sp(  20,  80 ), sp(  22,  80 ),
    sp(   8,  60 ), sp(  10,  60 ), sp(  16,  60 ), sp(  18,  60 ),
    sp(   6,  50 ), sp(   8,  50 ), sp(  14,  50 ), sp(  16,  50 ),
    sp(   4,  40 ), sp(   6,  40 ), sp(  12,  40 ), sp(  14,  40 ),
    sp(   2,  30 ), sp(   8,  30 ), sp(  10,  30 ), sp(  12,  30 ),
    sp(   0,  20 ), sp(   0,  20 ), sp( -10,  20 ), sp( -10,  20 ),
    sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
};

const knight_table: HalfTable =
.{
    sp( -15,  -15 ), sp(  -3,  -3 ), sp(  -3, -3 ), sp(  -3, -3 ),
    sp( -10,  -10 ), sp(   3,   3 ), sp(  -2, -2 ), sp(   0,  0 ),
    sp(  -5,   -5 ), sp(  10,  10 ), sp(  40, 40 ), sp(  55, 55 ),
    sp(  -5,   -5 ), sp(  15,  15 ), sp(  40, 40 ), sp(  45, 45 ),
    sp(  -5,   -5 ), sp(  10,  10 ), sp(  30, 30 ), sp(  40, 40 ),
    sp(  -3,   -3 ), sp(  10,  10 ), sp(  40, 40 ), sp(  30, 30 ),
    sp(  -1,   -1 ), sp(  -3,  -3 ), sp(  10, 10 ), sp(  10, 10 ),
    sp( -10,  -10 ), sp(  -5,  -5 ), sp(  -2, -2 ), sp(  -2, -2 ),
};

const bishop_table: HalfTable =
.{
    sp( -20, -20 ), sp( -10,  -5 ), sp( -10,  -5 ), sp( -10,  -5 ),
    sp( -10,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
    sp( -10,   0 ), sp(   0,  10 ), sp(   5,  15 ), sp(  10,  20 ),
    sp( -10,   0 ), sp(   5,  10 ), sp(   5,  15 ), sp(  10,  20 ),
    sp( -10,   0 ), sp(   0,  10 ), sp(  10,  15 ), sp(  10,  20 ),
    sp( -10,   0 ), sp(  10,  10 ), sp(  10,  15 ), sp(  10,  20 ),
    sp( -10,   0 ), sp(   5,   0 ), sp(   0,   0 ), sp(   0,  20 ),
    sp( -20, -10 ), sp( -10,  -5 ), sp( -10,  -5 ), sp( -10,  -5 ),
};

const rook_table: HalfTable =
.{
    sp(   1,   0 ), sp(   2,   2 ), sp(   2,   2 ), sp(   2,   2 ),
    sp(   5,  10 ), sp(  10,  10 ), sp(  10,  10 ), sp(  10,  10 ),
    sp(  -5,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
    sp(  -5,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
    sp(  -5,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
    sp(  -5,   0 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,   0 ),
    sp(  -5,   0 ), sp(   0,   5 ), sp(   0,   5 ), sp(   0,   0 ),
    sp(   0,   0 ), sp(   0,   0 ), sp(   3,   0 ), sp(   5,   0 ),
};

const queen_table: HalfTable =
.{
    sp( -20, -20 ), sp( -10, -10 ), sp( -10, -10 ), sp(  -5, -5 ),
    sp( -10, -10 ), sp(   0,   0 ), sp(   0,   0 ), sp(   0,  0 ),
    sp( -10, -10 ), sp(   0,   0 ), sp(   5,   5 ), sp(   5,  5 ),
    sp(  -5,  -5 ), sp(   0,   0 ), sp(   5,   8 ), sp(   5, 12 ),
    sp(   0,   0 ), sp(   0,   0 ), sp(   5,   8 ), sp(   5, 12 ),
    sp( -10, -10 ), sp(   5,   5 ), sp(   5,   5 ), sp(   5,  5 ),
    sp( -10, -10 ), sp(   0,   0 ), sp(   5,   5 ), sp(   0,  0 ),
    sp( -20, -20 ), sp( -10, -10 ), sp( -10, -10 ), sp(  -5, -5 ),
};

const king_table: HalfTable =
.{
    sp( -80, -20 ), sp( -70, -10 ), sp( -70, -10 ), sp( -70, -10 ),
    sp( -60,  -5 ), sp( -60,   0 ), sp( -60,   5 ), sp( -70,  30 ),
    sp( -40, -10 ), sp( -50,  -5 ), sp( -50,  20 ), sp( -60,  45 ),
    sp( -30, -15 ), sp( -40, -10 ), sp( -40,  35 ), sp( -50,  45 ),
    sp( -20, -20 ), sp( -30, -15 ), sp( -30,  30 ), sp( -40,  40 ),
    sp( -10, -25 ), sp( -20, -20 ), sp( -20,  20 ), sp( -20,  25 ),
    sp(  15, -30 ), sp(  20, -25 ), sp(  -5,   0 ), sp(  -5,   0 ),
    sp(  20, -50 ), sp(  30, -30 ), sp(  10, -30 ), sp( -10, -30 ),
};

/// Helper table (white) to convert squares to HalfTable indexes.
const white_indexer: [64]u8 = blk:
{
    const us = Color.WHITE;
    var result: [64]u8 = undefined;
    for (Square.all) |sq|
    {
        const r = sq.rank();
        const f = sq.file();
        const rank = if (us.e == .white) 7 - r else r;
        const file = if (f > 3) 7 - f else f;
        const idx: usize = @as(usize, rank) * 4 + file;
        result[sq.u] = idx;
    }
    break :blk result;
};

/// Helper table (black) to convert squares to HalfTable indexes.
const black_indexer: [64]u8 = blk:
{
    const us = Color.BLACK;
    var result: [64]u8 = undefined;
    for (Square.all) |sq|
    {
        const r = sq.rank();
        const f = sq.file();
        const rank = if (us.e == .white) 7 - r else r;
        const file = if (f > 3) 7 - f else f;
        const idx: usize = @as(usize, rank) * 4 + file;
        result[sq.u] = idx;
    }
    break :blk result;
};

/// Values should be in 1...100 indicating a percentage.\
/// i16 is chosen to avoid typecasting during calculations.
pub const Weight = enum(Value)
{
    ToMove = 20,

    // Territory: i16,
    // OpenPosition: i16, // engines in general are not the strongest (or at least very boring) in 'blocked' positions.

    // PassedPawn: i16,
    // ProtectedPawn: i16,
    // BackwardPawn: i16,
    // IsolatedPawn: i16,
    // DoubledPawn: i16,
    // PawnIslands: i16,

    // KnightProtected: i16,
    // KnightMobility: i16,

    // BishopProtected: i16,
    // BishopMobility: i16,
    // BishopPair: i16,

    RooksConnected = 20,

    // QueenMobility: i16,

    KingSafety = 80, // protected by our pieces
    // KingAttack: i16, // our pieces close to enemy king or sliders pointing at it indirectly.
    // CastlingAvailability: i16,

    fn value(comptime self: Weight) Value
    {
        return @intFromEnum(self);
    }
};









pub fn matter(pos: *const Position) Value
{
    return
        PieceType.PAWN.material() * @popCount(pos.pawns(Color.WHITE)) +
        PieceType.KNIGHT.material() * @popCount(pos.knights(Color.WHITE)) +
        PieceType.BISHOP.material() * @popCount(pos.bishops(Color.WHITE)) +
        PieceType.ROOK.material() * @popCount(pos.rooks(Color.WHITE)) +
        PieceType.QUEEN.material() * @popCount(pos.queens(Color.WHITE));
}
