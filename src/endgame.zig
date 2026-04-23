// zig fmt: off

//! Evaluation scaling to guide the search.

const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const attacks = @import("attacks.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

const popcnt = funcs.popcnt;
const first_square = funcs.first_square;
const fmul = funcs.fmul;
const m48 = Material.encode_48;
const m96 = Material.encode_96;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const ScorePair = types.ScorePair;
const Position = position.Position;
const Material = position.Material;

pub fn scale(pos: *const Position, eval: i32) f32 {

    // Do not scale pawn endgames or scores that are already very close to zero.
    if (pos.phase() == 0 or is_drawish(eval)) {
        return 1.0;
    }

    const winner: Color = if (eval > 0) Color.WHITE else Color.BLACK;
    const loser: Color = winner.opp();
    const winner_pawn_count: i32 = pos.pawn_count(winner);

    // Avoid calculating material twice in certain situations below.
    var material_difference: ?i32 = null;
    var result: f32 = 1.00;

    // No pawns on the board.
    if (pos.all_pawns() == 0) {
        if (pos.by_color(loser) == pos.kings(loser)) {
            return scale_maybe_mate(pos, winner, loser);
        }
        else {
            material_difference = get_material_sum(pos, winner) - get_material_sum(pos, loser);
            if (scale_pawnless(pos, winner, loser, material_difference.?)) |r| {
                return r;
            }
        }
    }

    // Only scale down when we suspect there are little winning chances.
    if (winner_pawn_count <= 2) {
        if (material_difference == null) {
            material_difference = get_material_sum(pos, winner) - get_material_sum(pos, loser);
        }
        if (@abs(material_difference.?) <= types.simple_value_bishop) {
            result *= pawn_scales[@abs(winner_pawn_count)];
            if (winner_pawn_count == 1 and pos.minor_major_count(winner) == 1) {
                result *= 0.90;
            }
        }
    }
    if (is_opposite_colored_bishops_only_endgame(pos)) {
        result *= base_scale.drawish;
    }
    return result;
}

fn is_drawish(static_eval: i32) bool {
    return @abs(static_eval) <= 8;
}

/// Sum piece values, not counting pawns.
fn get_material_sum(pos: *const Position, us: Color) i32 {
    return
        types.simple_value_knight * pos.knight_count(us) +
        types.simple_value_bishop * pos.bishop_count(us) +
        types.simple_value_rook * pos.rook_count(us) +
        types.simple_value_queen * pos.queen_count(us);
}

/// Opposite colored bishops + pawns only.
fn is_opposite_colored_bishops_only_endgame(pos: *const Position) bool {
    // I don't like this phase dependancy, but I have nothing smarter now.
    const ocb: bool =
        pos.phase_by_color[0] == types.ph_minor and
        pos.phase_by_color[1] == types.ph_minor and
        pos.bishop_count(Color.WHITE) == 1 and
        pos.bishop_count(Color.BLACK) == 1 and
        pos.bishop_square(Color.WHITE).color().e != pos.bishop_square(Color.BLACK).color().e;
    return ocb;
}

/// Easy mate or nothing.
/// Assumes (1) loser has only a king (2) there are no pawns on the board.
fn scale_maybe_mate(pos: *const Position, winner: Color, loser: Color) f32 {
    if (pos.major_count(winner) >= 1) {
        return def_mating.scale(pos, winner, loser);
    }
    const m: u48 = pos.material.decode_side(winner);
    return switch (m) {
        K, KN, KB => 0.00,
        KNN => 0.01,
        KBN => kbn_mating.scale(pos, winner, loser), // scale_kbn_mate(pos, winner, loser),
        else => def_mating.scale(pos, winner, loser),
    };
}

/// Handle some pawnless positions.
/// Assumes (1) there are no pawns on the board (2) both sides have pieces.
fn scale_pawnless(pos: *const Position, winner: Color, loser: Color, material_difference: i32) ?f32 {
    if (material_difference == 0) {
        return base_scale.drawish;
    }

    const small_margin: i32 = comptime (1 + types.simple_value_queen - types.simple_value_rook); // 401
    const big_margin: i32 = comptime (types.simple_value_rook); // 500

    if (material_difference >= big_margin) {
        return base_scale.winning;
    }
    else if (material_difference >= small_margin) {
        return base_scale.good;
    }

    // Handle some cases with a smaller margin than 400.
    const m: u96 = pos.material.decode_both(winner, loser);

    return switch (m) {
        m96(KR, KB)   => base_scale.drawish, // 200
        m96(KR, KN)   => base_scale.drawish, // 200
        m96(KQ, KR)   => base_scale.winning, // 400
        m96(KBB, KB)  => base_scale.drawish, // 300
        m96(KBB, KN)  => base_scale.winning, // 300
        m96(KRR, KNN) => base_scale.winning, // 400
        m96(KRR, KBB) => base_scale.drawish, // 400
        else => null,
    };
}

const base_scale = struct {
    const mate: f32 = 2.00;
    const winning: f32 = 1.50;
    const good: f32 = 1.25;
    const drawish: f32 = 0.50;
};

/// Indexing by [number of winner pawns].
const pawn_scales: [3]f32 = .{
    0.12, 0.69, 0.93,
};

const def_mating = struct {
    /// Indexing by [manhattan distance to corner].
    const cornering_scales: [7]f32 = .{
        1.20, 1.15, 1.10, 1.05, 1.00, 0.90, 0.85,
    };

    /// Indexing by [manhattan distance between kings].
    const king_dist_scales: [15]f32 = .{
        1.10, 1.10, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84,
    };

    fn scale(pos: *const Position, winner: Color, loser: Color) f32 {
        const king_sq_winner: Square = pos.king_square(winner);
        const king_sq_loser: Square = pos.king_square(loser);
        const d: u8 = king_sq_loser.manhattan_distance_to_corner();
        const k: u4 = bitboards.manh(king_sq_winner, king_sq_loser);
        var factor: f32 = base_scale.mate;
        factor *= cornering_scales[d];
        factor *= king_dist_scales[k];
        return factor;
    }
};

const kbn_mating = struct {
    /// 'Distances' for white squares.
    const white_squares: [64]u8 = .{
        7, 7, 7, 7, 6, 4, 2, 0,
        7, 8, 8, 8, 7, 5, 3, 2,
        7, 8, 9, 9, 8, 6, 5, 4,
        7, 8, 9,10, 9, 8, 7, 6,
        6, 7, 8, 9,10, 9, 8, 7,
        4, 5, 6, 8, 9, 9, 8, 7,
        2, 3, 5, 7, 8, 8, 8, 7,
        0, 2, 4, 6, 7, 7, 7, 7
    };

    /// 'Distances' for white squares.
    const black_squares: [64]u8 = .{
        0, 2, 4, 6, 7, 7, 7, 7,
        2, 3, 5, 7, 8, 8, 8, 7,
        4, 5, 6, 8, 9, 9, 8, 7,
        6, 7, 8, 9,10, 9, 8, 7,
        7, 8, 9,10, 9, 8, 7, 6,
        7, 8, 9, 9, 8, 6, 5, 4,
        7, 8, 8, 8, 7, 5, 3, 2,
        7, 7, 7, 7, 6, 4, 2, 0
    };

    /// Indexing by [white_squares or black_squares].
    const cornering_scales: [11]f32 = .{
        1.20, 1.16, 1.12, 1.10, 1.06, 1.01, 0.97, 0.93, 0.89, 0.85, 0.81,
    };

    /// Search needs guiding for this one: push the king to a specific colored corner.
    fn scale(pos: *const Position, winner: Color, loser: Color) f32 {
        const king_sq_winner: Square = pos.king_square(winner);
        const king_sq_loser: Square = pos.king_square(loser);
        const corner_color: Color = pos.bishop_square(winner).color();
        const c: u8 = cornering_idx(king_sq_loser, corner_color);
        const k: u4 = bitboards.manh(king_sq_winner, king_sq_loser);
        var factor: f32 = base_scale.mate;
        factor *= cornering_scales[c];
        factor *= def_mating.king_dist_scales[k];
        return factor;
    }

    fn cornering_idx(sq: Square, corner_color: Color) u8 {
        return switch (corner_color.e) {
            .white => white_squares[sq.u],
            .black => black_squares[sq.u],
        };
    }
};

const K: u48 = m48(0, 0, 0, 0, 0);

const KP: u48 = m48(1, 0, 0, 0, 0);
const KN: u48 = m48(0, 1, 0, 0, 0);
const KB: u48 = m48(0, 0, 1, 0, 0);
const KR: u48 = m48(0, 0, 0, 1, 0);
const KQ: u48 = m48(0, 0, 0, 0, 1);

const KPP: u48 = m48(2, 0, 0, 0, 0);
const KNN: u48 = m48(0, 2, 0, 0, 0);
const KBN: u48 = m48(0, 1, 1, 0, 0);
const KBB: u48 = m48(0, 0, 2, 0, 0);
const KRR: u48 = m48(0, 0, 0, 2, 0);
const KRP: u48 = m48(1, 0, 0, 1, 0);
const KQP: u48 = m48(1, 0, 0, 0, 1);

const KRPP: u48 = m48(2, 0, 0, 1, 0);
const KRRP: u48 = m48(1, 0, 0, 2, 0);
const KQPP: u48 = m48(2, 0, 0, 0, 1);
