const types = @import("types.zig");

const SmallValue = types.SmallValue;
const Value = types.Value;

const ScorePair = types.ScorePair;
const pair = types.pair;

// Piece values.
pub const piece_value_table: [6]ScorePair = .{
    pair(75, 141), pair(301, 321), pair(331, 356), pair(439, 612), pair(874, 1080), pair(0, 0),
};

/// By [square distance between our king and our passed pawn]
pub const king_passed_pawn_distance_table: [8]ScorePair = .{
    pair(0, 0), pair(-9, 63), pair(-13, 48), pair(-3, 27), pair(0, 17), pair(3, 15), pair(15, 13), pair(5, 10),
};

/// By [square distance between their king and our passed pawn]
pub const enemy_king_passed_pawn_distance_table: [8]ScorePair = .{
    pair(0, 0), pair(-76, 4), pair(-11, 7), pair(-7, 30), pair(1, 37), pair(3, 42), pair(5, 47), pair(-9, 44),
};

/// By [rank]. Pawns horizontally next to eachother.
pub const pawn_phalanx_bonus: [8]ScorePair = .{
    pair(0, 0), pair(5, 1), pair(12, 7), pair(18, 17), pair(44, 59), pair(98, 192), pair(-46, 414), pair(0, 0),
};

/// By [rank].
pub const passed_pawn_bonus: [8]ScorePair = .{
    pair(0, 0), pair(-10, -74), pair(-10, -60), pair(-8, -28), pair(18, 7), pair(11, 75), pair(28, 66), pair(0, 0),
};

/// By [rank].
pub const protected_pawn_bonus: [8]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(20, 11), pair(13, 7), pair(13, 15), pair(23, 43), pair(150, 37), pair(0, 0),
};

/// By [file].
pub const doubled_pawn_penalty: [8]ScorePair = .{
    pair(-7, -45), pair(13, -37), pair(2, -25), pair(1, -15), pair(-8, -6), pair(-7, -19), pair(11, -32), pair(-3, -49)
};

/// By [file].
pub const isolated_pawn_penalty: [8]ScorePair = .{
    pair(-8, 12), pair(-1, -12), pair(-11, -4), pair(-7, -13), pair(-11, -15), pair(-4, -6), pair(1, -12), pair(-7, 8),
};

pub const king_cannot_reach_passed_pawn_bonus = pair(-344, 199);

pub const bishop_pair_bonus: ScorePair = pair(18, 58);

/// EXPERIMENTAL
pub const bad_bishop_penalty: ScorePair = pair(-5, -30);

pub const tempo_bonus: ScorePair = pair(30, 27);

/// By [popcnt].
pub const knight_mobility_table: [9]ScorePair = .{
    pair(-25, 8), pair(-11, 4), pair(-2, 24), pair(3, 33), pair(7, 39), pair(11, 47), pair(16, 46), pair(19, 44), pair(21, 37),
};

/// By [popcnt].
pub const bishop_mobility_table: [14]ScorePair = .{
    pair(-31, -16), pair(-21, -24), pair(-12, -8), pair(-7, 4), pair(-1, 13), pair(2, 22), pair(4, 26), pair(6, 30),
    pair(7, 34), pair(12, 32), pair(21, 30), pair(28, 32), pair(29, 44), pair(35, 27),
};

/// By [popcnt].
pub const rook_mobility_table: [15]ScorePair = .{
    pair(-36, -31), pair(-26, -4), pair(-21, 1), pair(-22, 16), pair(-22, 20), pair(-18, 23), pair(-16, 28), pair(-13, 31),
    pair(-9, 35), pair(-7, 38), pair(-3, 40), pair(-2, 45), pair(2, 47), pair(4, 46), pair(1, 45)
};

/// By [popcnt].
pub const queen_mobility_table: [28]ScorePair = .{
    pair(-69, -126), pair(-15, -237), pair(-22, -28), pair(-16, 17), pair(-16, 57), pair(-14, 76), pair(-11, 89), pair(-10, 102),
    pair(-8, 110), pair(-5, 111), pair(-3, 116), pair(-3, 123), pair(1, 120), pair(1, 124), pair(3, 124), pair(7, 121),
    pair(6, 125), pair(9, 122), pair(17, 112), pair(30, 98), pair(37, 90), pair(77, 61), pair(75, 55), pair(91, 32),
    pair(135, 15), pair(239, -65), pair(160, -18), pair(85, 1),
};

/// By [piece][squarecount].
pub const attack_power: [6][8]ScorePair = .{
    .{ pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0) }, // pawn
    .{ pair(0, 0), pair(12, 0), pair(27, -10), pair(63, -30), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0) }, // knight
    .{ pair(0, 0), pair(13, 1), pair(29, -2), pair(61, -11), pair(68, -40), pair(0, 0), pair(0, 0), pair(0, 0) }, // bishop
    .{ pair(0, 0), pair(21, -21), pair(36, -22), pair(54, -16), pair(81, -19), pair(90, -29), pair(0, 0), pair(0, 0) }, // rook
    .{ pair(0, 0), pair(2, 12), pair(12, 23), pair(31, 28), pair(75, 11), pair(110, 6), pair(169, -15), pair(229, -62) }, // queen
    .{ pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0) } // king
};

/// By [square] (only on rank 4,5,6).
pub const knight_outpost_table: [64]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 1
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 2
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 3
    pair(9, 10), pair(6, 3), pair(2, 16), pair(8, 18), pair(7, 21), pair(-7, 19), pair(2, 9), pair(1, 9), // rank 4
    pair(1, 14), pair(13, 16), pair(17, 20), pair(9, 34), pair(13, 23), pair(11, 19), pair(14, 14), pair(-4, 24), // rank 5
    pair(18, 23), pair(16, 13), pair(36, 21), pair(43, 22), pair(47, 28), pair(31, 43), pair(43, 17), pair(-3, 39),  // rank 6
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 7
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 8
};

/// By [square] (only on rank 4,5,6).
pub const bishop_outpost_table: [64]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 1
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 2
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 3
    pair(-23, 31), pair(10, 6), pair(3, 20), pair(18, 14), pair(23, 24), pair(2, 10), pair(14, 0), pair(-47, 10), // rank 4
    pair(-20, -8), pair(18, 7), pair(11, 3), pair(24, 13), pair(16, 5), pair(15, 1), pair(14, 9), pair(18, -21), // rank 5
    pair(-10, 15), pair(20, 0), pair(30, 2), pair(42, 3), pair(67, -14), pair(45, 0), pair(26, -10), pair(-16, -26), // rank 6
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 7
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 8
};

/// By open / halfopen + file
pub const rook_on_file_bonus: [2][8]ScorePair = .{
    .{ pair(22, 5), pair(19, 3), pair(17, 9), pair(18, 8), pair(19, 13), pair(31, 3), pair(39, 3), pair(66, -1) },
    .{ pair(9, 33), pair(9, 9), pair(11, 9), pair(18, 0), pair(14, 1), pair(14, -2), pair(21, 0), pair(16, 19) },
};

// By the 3x4 king area (exclusing the king square). King on #7.
pub const pawn_protection_table: [12]ScorePair = .{
    pair(13, -6), pair(17, -6), pair(9, -3), // king rank + 2
    pair(21, -11), pair(18, -11), pair(20, -11), // king rank + 1
    pair(31, -4), pair(0, 0), pair(28, -3), // king rank + 0
    pair(-8, 5), pair(-5, 0), pair(-7, 7), // king rank - 1
};

// By the 3x7 king area (exclusing the king square). King on #19.
pub const pawn_storm_table: [21]ScorePair = .{
    pair(-6, 1), pair(-13, 3), pair(-7, 3), // king rank + 6
    pair(-5, 0), pair(-14, 5), pair(-8, 3), // king rank + 5
    pair(1, -7), pair(-11, 0), pair(0, -5), // king rank + 4
    pair(14, -9), pair(-2, -5), pair(12, -6), // king rank + 3
    pair(23, -9), pair(24, 0), pair(16, -9), // king rank + 2
    pair(0, 0), pair(15, -31), pair(0, 0), // king rank + 1
    pair(56, -49), pair(0, 0), pair(51, -47), // king rank + 0
};

/// By [file][open].
pub const king_on_file_penalty: [2][8]ScorePair = .{
    .{ pair(-55, -13), pair(-67, -7), pair(-37, -9), pair(-24, -9), pair(-20, -2), pair(-36, 0), pair(-49, 3), pair(-34, 5) }, // open
    .{ pair(-3, 45), pair(-31, 19), pair(-17, 12), pair(3, -5), pair(-2, -7), pair(-3, 3), pair(-27, 23), pair(-15, 34) }, // half open
};

/// By [threatenedpiece][is_protected]
pub const threatened_by_pawn_penalty: [6][2]ScorePair = .{
    .{ pair(-17, 13), pair(-11, 17) },
    .{ pair(-66, -20), pair(-66, -25) },
    .{ pair(-55, -44), pair(-64, -66) },
    .{ pair(-91, 3), pair(-79, -30) },
    .{ pair(-74, 37), pair(-82, 20) },
    .{ pair(0, 0), pair(0, 0) },
};

/// By [threatenedpiece][is_protected]
pub const threatened_by_knight_penalty: [6][2]ScorePair = .{
    .{ pair(-1, -20), pair(10, -12) },
    .{ pair(-24, -43), pair(-4, -59) },
    .{ pair(-41, -31), pair(-27, -33) },
    .{ pair(-79, -4), pair(-58, -27) },
    .{ pair(-54, 30), pair(-55, 17) },
    .{ pair(0, 0), pair(0, 0) },
};

/// By [threatenedpiece][is_protected]
pub const threatened_by_bishop_penalty: [6][2]ScorePair = .{
    .{ pair(-10, -20), pair(1, -9) },
    .{ pair(-47, -23), pair(-21, -24) },
    .{ pair(16, -136), pair(27, -130) },
    .{ pair(-79, -7), pair(-45, -36) },
    .{ pair(-81, -16), pair(-59, -85) },
    .{ pair(0, 0), pair(0, 0) },

};

/// By [threatenedpiece][is_protected]
pub const threatened_by_rook_penalty: [6][2]ScorePair = .{
    .{ pair(-2, -25), pair(10, -11) }, // pawn
    .{ pair(-42, -24), pair(-3, -13) }, // knight
    .{ pair(-30, -29), pair(-14, -3) }, // bishop
    .{ pair(-3, -62), pair(13, -38) }, // rook
    .{ pair(-83, 9), pair(-62, -38) }, // queen
    .{ pair(0, 0), pair(0, 0) }, // king
};

/// By [piece]
pub const pawn_push_threat_table: [13]ScorePair = .{
    pair(0, 0), pair(16, 30), pair(21, 18), pair(27, 10), pair(23, -6), pair(55, -6), // w_pawn...w_king
    pair(0, 0), pair(16, 30), pair(21, 18), pair(27, 10), pair(23, -6), pair(55, -6), // b_pawn...b_king
    pair(0, 0), // no_piece TODO: remove. we should never access this.
};

/// By [piecetype]
pub const safe_check_bonus: [6]ScorePair = .{
  //pair(0, 0), pair(47, 7), pair(19, 21), pair(59, -2), pair(28, 13), pair(0, 0), // original
    pair(0, 0), pair(54, 14), pair(26, 28), pair(66, 5), pair(35, 20), pair(0, 0), // EXPERIMENTAL: bit higher because of eval bug.
};

/// By [piecetype][square]
pub const piece_square_table: [6][64]ScorePair = .{
    .{ // pawn
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 1
        pair(-25, -25), pair(-29, -8), pair(-18, -28), pair(-21, -32), pair(-12, -25), pair(1, -33), pair(-4, -23), pair(-17, -41), // rank 2
        pair(-34, -29), pair(-39, -13), pair(-26, -35), pair(-21, -34), pair(-14, -35), pair(-19, -38), pair(-19, -25), pair(-18, -43), // rank 3
        pair(-24, -26), pair(-32, -5), pair(-15, -34), pair(-5, -41), pair(-2, -42), pair(-1, -41), pair(-20, -16), pair(-13, -39), // rank 4
        pair(-19, -11), pair(-22, -4), pair(-13, -30), pair(-10, -44), pair(11, -46), pair(6, -42), pair(-15, -12), pair(-10, -28), // rank 5
        pair(5, 0), pair(-16, 21), pair(19, -23), pair(27, -53), pair(43, -57), pair(65, -43), pair(18, 4), pair(6, -4),  // rank 6
        pair(44, 67), pair(25, 70), pair(14, 71), pair(53, 28), pair(47, 30), pair(41, 43), pair(-32, 86), pair(-27, 83), // rank 7
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), // rank 8
    },
    .{ // knight
        pair(-61, 13), pair(-23, 8), pair(-20, 12), pair(-8, 17), pair(0, 15), pair(2, 4), pair(-20, 14), pair(-26, 5), // rank 1
        pair(-31, 8), pair(-18, 16), pair(-7, 16), pair(7, 16), pair(9, 15), pair(4, 13), pair(3, 5), pair(-3, 18), // rank 2
        pair(-22, 7), pair(-1, 15), pair(10, 20), pair(19, 31), pair(31, 29), pair(18, 14), pair(18, 10), pair(0, 12), // rank 3
        pair(-7, 20), pair(15, 19), pair(25, 31), pair(31, 28), pair(30, 34), pair(42, 16), pair(32, 15), pair(8, 15), // rank 4
        pair(7, 18), pair(16, 22), pair(32, 25), pair(39, 25), pair(34, 27), pair(44, 26), pair(24, 22), pair(33, 5), // rank 5
        pair(-6, 6), pair(10, 14), pair(7, 27), pair(19, 19), pair(27, 15), pair(59, -9), pair(7, 5), pair(20, -13), // rank 6
        pair(-28, 4), pair(-12, 16), pair(-1, 11), pair(9, 9), pair(7, -2), pair(39, -6), pair(1, 7), pair(-6, -14), // rank 7
        pair(-131, -47), pair(-101, -6), pair(-69, 10), pair(-26, -8), pair(-8, -3), pair(-39, -28), pair(-106, -4), pair(-99, -66), // rank 8
    },
    .{ // bishop
        pair(-9, -4), pair(8, 15), pair(-9, 11), pair(-10, 10), pair(2, 5), pair(-11, 16), pair(4, 1), pair(17, -22), // rank 1
        pair(-3, 9), pair(0, -5), pair(8, -1), pair(-7, 10), pair(4, 9), pair(11, 3), pair(22, 0), pair(11, -9), // rank 2
        pair(-18, 8), pair(6, 13), pair(2, 13), pair(-1, 17), pair(2, 22), pair(6, 13), pair(10, 7), pair(3, 0), // rank 3
        pair(-14, 7), pair(-24, 17), pair(-6, 12), pair(1, 12), pair(2, 6), pair(-6, 11), pair(-14, 15), pair(-3, 1), // rank 4
        pair(-18, 13), pair(-4, 15), pair(-2, 13), pair(5, 18), pair(2, 14), pair(8, 16), pair(-9, 15), pair(-12, 11), // rank 5
        pair(-8, 14), pair(-2, 10), pair(-12, 15), pair(-8, 6), pair(-14, 10), pair(17, 15), pair(11, 13), pair(5, 14), // rank 6
        pair(-21, 0), pair(-15, 10), pair(-14, 6), pair(-28, 10), pair(-29, 5), pair(-19, 4), pair(-46, 17), pair(-33, 1), // rank 7
        pair(-29, 16), pair(-52, 17), pair(-53, 10), pair(-87, 16), pair(-84, 15), pair(-73, 5), pair(-38, 8), pair(-62, 5), // rank 8
    },
    .{ // rook
        pair(-26, 27), pair(-22, 25), pair(-18, 28), pair(-12, 22), pair(-7, 14), pair(-12, 21), pair(-17, 18), pair(-26, 13), // rank 1
        pair(-37, 24), pair(-29, 30), pair(-17, 28), pair(-18, 27), pair(-11, 17), pair(-15, 20), pair(-2, 9), pair(-36, 10), // rank 2
        pair(-36, 30), pair(-28, 34), pair(-23, 29), pair(-24, 33), pair(-15, 24), pair(-18, 25), pair(7, 8), pair(-18, 6), // rank 3
        pair(-32, 36), pair(-27, 44), pair(-20, 38), pair(-14, 37), pair(-14, 30), pair(-29, 39), pair(-6, 27), pair(-29, 24), // rank 4
        pair(-21, 42), pair(1, 41), pair(3, 41), pair(-2, 38), pair(8, 24), pair(13, 25), pair(13, 28), pair(-14, 27), // rank 5
        pair(-23, 40), pair(9, 39), pair(1, 38), pair(-1, 33), pair(22, 22), pair(34, 22), pair(50, 22), pair(-3, 23), // rank 6
        pair(-18, 41), pair(-12, 51), pair(-3, 49), pair(13, 37), pair(-3, 36), pair(12, 39), pair(16, 35), pair(-8, 35), // rank 7
        pair(-19, 45), pair(-22, 53), pair(-23, 56), pair(-28, 54), pair(-18, 43), pair(-2, 47), pair(-10, 49), pair(-27, 46), // rank 8
    },
    .{ // queen
        pair(-18, 55), pair(-15, 50), pair(-11, 56), pair(-9, 58), pair(-4, 45), pair(-13, 37), pair(-10, 32), pair(-1, 27), // rank 1
        pair(-8, 48), pair(-5, 51), pair(-3, 55), pair(-1, 66), pair(0, 65), pair(6, 38), pair(15, 17), pair(16, 4), // rank 2
        pair(-9, 52), pair(-7, 75), pair(-9, 89), pair(-13, 92), pair(-9, 93), pair(-1, 80), pair(13, 62), pair(6, 52), // rank 3
        pair(-11, 65), pair(-17, 96), pair(-12, 95), pair(-10, 110), pair(-4, 99), pair(-9, 91), pair(7, 76), pair(3, 72), // rank 4
        pair(-11, 67), pair(-2, 79), pair(-2, 88), pair(-10, 105), pair(-5, 104), pair(10, 82), pair(14, 83), pair(14, 67), // rank 5
        pair(-5, 61), pair(-5, 70), pair(-10, 100), pair(-9, 103), pair(-11, 114), pair(19, 92), pair(22, 68), pair(16, 68), // rank 6
        pair(-9, 57), pair(-20, 76), pair(-22, 107), pair(-34, 119), pair(-50, 142), pair(-11, 104), pair(-8, 85), pair(30, 72), // rank 7
        pair(-39, 68), pair(-35, 66), pair(-28, 88), pair(1, 75), pair(-14, 84), pair(-5, 81), pair(36, 28), pair(-13, 65), // rank 8
    },
    .{ // king
        pair(12, -67), pair(35, -40), pair(26, -24), pair(-37, -1), pair(5, -24), pair(-23, -11), pair(18, -36), pair(20, -75), // rank 1
        pair(27, -42), pair(2, -10), pair(-3, 0), pair(-24, 11), pair(-29, 12), pair(-18, 1), pair(0, -17), pair(12, -42), // rank 2
        pair(-26, -30), pair(23, -14), pair(-23, 9), pair(-40, 24), pair(-36, 19), pair(-32, 7), pair(-10, -13), pair(-48, -25), // rank 3
        pair(-43, -27), pair(7, -7), pair(-39, 18), pair(-81, 38), pair(-78, 32), pair(-41, 13), pair(-39, -3), pair(-119, -10), // rank 4
        pair(-36, -12), pair(12, 5), pair(-28, 26), pair(-76, 43), pair(-76, 37), pair(-48, 30), pair(-32, 9), pair(-131, 6), // rank 5
        pair(-72, 6), pair(68, 16), pair(6, 32), pair(-31, 45), pair(5, 42), pair(59, 31), pair(-22, 37), pair(-48, 4), // rank 6
        pair(-91, 6), pair(-4, 27), pair(-26, 33), pair(68, 15), pair(-5, 28), pair(-36, 49), pair(-34, 40), pair(-97, 16), // rank 7
        pair(74, -91), pair(82, -49), pair(60, -27), pair(-46, 9), pair(-19, -7), pair(-91, 11), pair(-24, -7), pair(101, -114), // rank 8
    }
};
