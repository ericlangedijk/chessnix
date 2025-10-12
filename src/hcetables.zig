const types = @import("types.zig");

const Value = types.Value;

pub const ScorePair = struct {
    mg: Value,
    eg: Value,

    pub const empty: ScorePair = .{ .mg = 0, .eg = 0 };

    pub fn inc(self: *ScorePair, sp: ScorePair) void {
        self.mg += sp.mg;
        self.eg += sp.eg;
    }

    pub fn add(self: ScorePair, other: ScorePair) ScorePair {
        return .{ .mg = self.mg + other.mg, .eg = self.eg + other.eg };
    }

    pub fn sub(self: ScorePair, other: ScorePair) ScorePair {
        return .{ .mg = self.mg - other.mg, .eg = self.eg - other.eg };
    }
};

// [is_our_piece][pawn_position][piece][piece_position]
//template <typename T>
//using PawnRelativePSQT = SideTable<SquareTable<PieceTable<SquareTable<T>>>>;

fn pair(mg: Value, eg: Value) ScorePair {
    return .{ .mg = mg, .eg = eg };
}

pub const king_pp_distance_table: [8]ScorePair = .{
    pair(0, 0), pair(-2, 84), pair(-6, 73), pair(-6, 61), pair(3, 50), pair(1, 41), pair(13, 33), pair(11, 26)
};

pub const enemy_king_pp_distance_table: [8]ScorePair = .{
    pair(0, 0), pair(-15, 2), pair(-22, 28), pair(-4, 51), pair(2, 62), pair(13, 68), pair(15, 74), pair(12, 91)
};

/// By rank.
pub const pawn_phalanx_bonus: [8]ScorePair = .{
    pair(0, 0), pair(9, 14), pair(20, 17), pair(23, 46), pair(47, 98), pair(112, 185), pair(75, 244), pair(0, 0)
};

/// By rank.
pub const passed_pawn_bonus: [8]ScorePair = .{
    pair(0, 0), pair(-28, -101), pair(-23, -85), pair(-9, -56), pair(19, -20), pair(10, 30), pair(154, 163), pair(0, 0)
};

/// By rank.
pub const defended_pawn_bonus: [8]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(6, -8), pair(4, 19), pair(16, 37), pair(54, 75), pair(218, 112), pair(0, 0)
};

/// By file.
pub const doubled_pawn_penalty: [8]ScorePair = .{
    pair(-34, -61), pair(-14, -15), pair(-3, 11), pair(-17, -10), pair(-14, -6), pair(-12, -9), pair(-14, 6), pair(-23, -45)
};

/// By file.
pub const isolated_pawn_penalty: [8]ScorePair = .{
    pair(-4, -3), pair(-10, -10), pair(-11, -6), pair(-7, -2), pair(-7, -6), pair(0, -11), pair(-7, -7), pair(5, -5)
};

pub const king_cannot_reach_pp_bonus = pair(-475, 103);

pub const bishop_pair_bonus: ScorePair = pair(23, 85);

/// By popcnt.
pub const knight_mobility_table: [9]ScorePair = .{
    pair(12, 38), pair(20, 24), pair(22, 30), pair(23, 36), pair(24, 41), pair(25, 51), pair(27, 54), pair(31, 52),
    pair(32, 48)
};

/// By popcnt.
pub const bishop_mobility_table: [14]ScorePair = .{
    pair(-8, 6), pair(1, -32), pair(4, -15), pair(8, -4), pair(10, 6), pair(13, 17), pair(13, 21), pair(12, 23),
    pair(16, 27), pair(21, 23), pair(26, 22), pair(38, 14), pair(43, 30), pair(95, -3)
};

/// By popcnt.
const rook_mobility_table: [15]ScorePair = .{
    pair(-49, -99), pair(-35, -88), pair(-33, -57), pair(-30, -26), pair(-29, -9), pair(-25, -5), pair(-20, 1), pair(-14, 5),
    pair(-11, 9), pair(-7, 14), pair(-4, 18), pair(3, 20), pair(6, 20), pair(29, 8), pair(66, -15),
};

/// By piece / popcnt squares
pub const attack_power: [6][8]ScorePair = .{
    .{
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0)
    },
    .{
        pair(0, 0), pair(24, 3), pair(42, -3), pair(107, -27), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0)
    },
    .{
        pair(0, 0), pair(12, 1), pair(35, -3), pair(76, -10), pair(101, -24), pair(0, 0), pair(0, 0), pair(0, 0)
    },
    .{
        pair(0, 0), pair(26, -13), pair(43, -12), pair(70, -12), pair(123, -23), pair(200, -45), pair(0, 0), pair(0, 0)
    },
    .{
        pair(0, 0), pair(8, 24), pair(25, 51), pair(53, 62), pair(116, 54), pair(171, 50), pair(299, -1), pair(370, -1)
    },
    .{
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0)
    }
};

/// Squares on relative rank 4,5,6
pub const knight_outpost_table: [64]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(5, 48), pair(54, 23), pair(7, 29), pair(10, 29), pair(24, 26), pair(45, 18), pair(49, 40), pair(77, 51),
    pair(12, 35), pair(13, 35), pair(11, 25), pair(17, 29), pair(8, 25), pair(2, 42), pair(12, 29), pair(40, 55),
    pair(-5, 19), pair(-6, 17), pair(7, 27), pair(5, 18), pair(-9, 33), pair(9, 27), pair(17, 17), pair(-4, 30),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
};

/// Squares on relative rank 4,5,6
pub const bishop_outpost_table: [64]ScorePair = .{
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(-21, -3), pair(26, -7), pair(10, 10), pair(16, 2), pair(13, -26), pair(49, -20), pair(85, 5), pair(-19, -22),
    pair(-18, 27), pair(17, 6), pair(-13, 29), pair(12, 36), pair(12, 21), pair(-16, 23), pair(16, -12), pair(58, 13),
    pair(23, 22), pair(-1, 2), pair(17, 10), pair(-4, 22), pair(-14, 15), pair(6, 25), pair(6, -13), pair(9, 46),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
    pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
};


pub const normal_piece_square_table: [6][64] ScorePair = .{
    .{
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0),
        pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0), pair(0, 0)
    },
    .{
        pair(-20, -114), pair(-42, -217), pair(-24, -175), pair(-43, -200), pair(-45, -205), pair(-32, -202), pair(-10, -195), pair(-20, -150),
        pair(-25, -181), pair(-25, -225), pair(7, -225), pair(-57, -192), pair(-59, -203), pair(-58, -224), pair(-39, -220), pair(-56, -227),
        pair(-26, -243), pair(-43, -217), pair(4, -222), pair(-15, -223), pair(-24, -203), pair(-5, -209), pair(-40, -219), pair(-9, -246),
        pair(-16, -215), pair(-3, -233), pair(25, -225), pair(-28, -217), pair(31, -213), pair(17, -223), pair(9, -238), pair(6, -217),
        pair(-14, -228), pair(-23, -235), pair(17, -212), pair(18, -223), pair(43, -220), pair(27, -221), pair(35, -234), pair(32, -236),
        pair(-10, -228), pair(-28, -223), pair(-9, -221), pair(-4, -211), pair(0, -218), pair(58, -237), pair(37, -228), pair(-31, -235),
        pair(-18, -231), pair(1, -216), pair(-68, -244), pair(4, -223), pair(-19, -221), pair(-9, -238), pair(-10, -256), pair(-15, -249),
        pair(-25, -229), pair(-19, -215), pair(-20, -213), pair(-21, -185), pair(0, -212), pair(-31, -214), pair(-36, -244), pair(-12, -180)
    },
    .{
        pair(-24, -167), pair(-16, -198), pair(-49, -199), pair(-43, -161), pair(-60, -186), pair(-54, -205), pair(-71, -246), pair(-51, -176),
        pair(-15, -218), pair(-59, -213), pair(-66, -233), pair(-64, -211), pair(-95, -220), pair(-48, -240), pair(-69, -198), pair(-40, -219),
        pair(-41, -237), pair(-51, -250), pair(-56, -237), pair(-26, -245), pair(-30, -240), pair(-27, -237), pair(-68, -244), pair(-40, -250),
        pair(-55, -232), pair(-36, -262), pair(-32, -252), pair(-39, -236), pair(-44, -241), pair(27, -265), pair(-60, -249), pair(-38, -264),
        pair(-7, -273), pair(-16, -261), pair(-27, -257), pair(-29, -251), pair(-26, -242), pair(37, -269), pair(-11, -268), pair(-51, -252),
        pair(-7, -268), pair(-53, -256), pair(-64, -253), pair(-72, -243), pair(19, -265), pair(-36, -252), pair(-10, -277), pair(-42, -251),
        pair(46, -298), pair(-41, -279), pair(-58, -271), pair(-26, -251), pair(-17, -255), pair(-22, -269), pair(-4, -278), pair(-19, -313),
        pair(3, -293), pair(-25, -291), pair(-55, -269), pair(-47, -277), pair(-28, -254), pair(-33, -271), pair(-28, -265), pair(-17, -299)
    },
    .{
        pair(-44, -333), pair(4, -353), pair(-21, -357), pair(68, -390), pair(15, -365), pair(-30, -354), pair(52, -365), pair(-39, -348),
        pair(-77, -336), pair(-99, -329), pair(-88, -332), pair(-59, -337), pair(-106, -322), pair(-87, -343), pair(-57, -348), pair(-83, -338),
        pair(2, -357), pair(-43, -345), pair(37, -364), pair(-64, -341), pair(-47, -344), pair(-12, -363), pair(-111, -334), pair(-46, -362),
        pair(-15, -351), pair(-81, -339), pair(-70, -345), pair(-49, -343), pair(-56, -345), pair(15, -369), pair(-63, -349), pair(-53, -350),
        pair(-63, -335), pair(-78, -338), pair(-112, -334), pair(-88, -348), pair(-60, -348), pair(-76, -350), pair(-56, -354), pair(-78, -343),
        pair(-102, -328), pair(-138, -317), pair(-76, -343), pair(-91, -337), pair(-100, -337), pair(-47, -357), pair(-97, -333), pair(-69, -344),
        pair(-126, -329), pair(-115, -325), pair(-81, -320), pair(-77, -331), pair(-109, -329), pair(-112, -327), pair(-94, -335), pair(-107, -330),
        pair(-109, -317), pair(-95, -338), pair(-60, -332), pair(-89, -333), pair(-94, -344), pair(-74, -351), pair(-143, -331), pair(-91, -338)
    },
    .{
        pair(-16, -62), pair(-56, -80), pair(-12, -77), pair(27, -57), pair(-52, -62), pair(1, -75), pair(-66, -25), pair(-64, -65),
        pair(-19, -41), pair(-54, -71), pair(-16, -46), pair(-9, -2), pair(-1, -6), pair(-54, -54), pair(-5, -13), pair(-39, -72),
        pair(-29, -48), pair(-28, -49), pair(-8, -22), pair(-58, -114), pair(17, 13), pair(18, 19), pair(5, -4), pair(-29, -54),
        pair(-45, -84), pair(-11, -4), pair(-27, -47), pair(-29, -61), pair(-50, -33), pair(-4, -36), pair(-17, -75), pair(-14, -38),
        pair(-25, -41), pair(-10, -13), pair(-16, -35), pair(9, 4), pair(-21, -46), pair(0, -19), pair(-52, -99), pair(-59, -88),
        pair(-23, -42), pair(-23, -39), pair(-18, -39), pair(25, 36), pair(-19, -30), pair(9, 16), pair(1, 5), pair(-4, -10),
        pair(-29, -61), pair(-53, -92), pair(-40, -65), pair(1, -6), pair(-2, -7), pair(6, 16), pair(-9, -21), pair(-2, -2),
        pair(-7, -14), pair(-11, -26), pair(-30, -58), pair(-26, -55), pair(5, 4), pair(-25, -50), pair(-2, -4), pair(-6, -13)
    },
    .{
        pair(45, 42), pair(-36, 19), pair(2, 4), pair(-37, -4), pair(11, -5), pair(-39, -1), pair(-38, -11), pair(-24, 41),
        pair(-9, 16), pair(-18, 15), pair(-19, 16), pair(-15, -7), pair(0, -4), pair(6, -3), pair(17, -14), pair(-33, 8),
        pair(-33, 28), pair(28, 24), pair(7, 10), pair(86, -25), pair(43, -21), pair(82, -17), pair(86, -17), pair(-81, 19),
        pair(28, 37), pair(8, 21), pair(14, 8), pair(62, -21), pair(25, -13), pair(39, -15), pair(-13, -6), pair(-52, 6),
        pair(12, 25), pair(-66, 21), pair(-10, 6), pair(38, -17), pair(23, -16), pair(36, -22), pair(-2, -16), pair(-58, 13),
        pair(-42, 33), pair(-35, 3), pair(47, -14), pair(32, -14), pair(32, -17), pair(45, -24), pair(-7, -28), pair(-65, 11),
        pair(5, 33), pair(-15, 6), pair(47, -15), pair(-9, -8), pair(5, -20), pair(40, -29), pair(33, -34), pair(-25, 3),
        pair(5, 36), pair(-65, 6), pair(-10, -3), pair(1, -24), pair(-16, -13), pair(-4, -26), pair(-65, -17), pair(-56, 9)
    }
};
