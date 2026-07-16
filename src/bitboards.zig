// zig fmt: off

//! Lots of bitboards.

const std = @import("std");
const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const int = funcs.int;

const Axis = types.Axis;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;


// --- Const stuff ---
pub const bb_rank_1: u64 = 0x00000000000000ff;
pub const bb_rank_2: u64 = 0x000000000000ff00;
pub const bb_rank_3: u64 = 0x0000000000ff0000;
pub const bb_rank_4: u64 = 0x00000000ff000000;
pub const bb_rank_5: u64 = 0x000000ff00000000;
pub const bb_rank_6: u64 = 0x0000ff0000000000;
pub const bb_rank_7: u64 = 0x00ff000000000000;
pub const bb_rank_8: u64 = 0xff00000000000000;

pub const bb_file_a: u64 = 0x0101010101010101;
pub const bb_file_b: u64 = 0x0202020202020202;
pub const bb_file_c: u64 = 0x0404040404040404;
pub const bb_file_d: u64 = 0x0808080808080808;
pub const bb_file_e: u64 = 0x1010101010101010;
pub const bb_file_f: u64 = 0x2020202020202020;
pub const bb_file_g: u64 = 0x4040404040404040;
pub const bb_file_h: u64 = 0x8080808080808080;

pub const bb_full: u64 = 0xffffffffffffffff;
pub const bb_border = bb_rank_1 | bb_rank_8 | bb_file_a | bb_file_h;
pub const bb_white_squares: u64 = 0b01010101_10101010_01010101_10101010_01010101_10101010_01010101_10101010;
pub const bb_black_squares: u64 = ~bb_white_squares;
pub const bb_white_side: u64 = bb_rank_1 | bb_rank_2 | bb_rank_3 | bb_rank_4;
pub const bb_black_side: u64 = bb_rank_5 | bb_rank_6 | bb_rank_7 | bb_rank_8;
pub const bb_queenside: u64 = bb_file_a | bb_file_b | bb_file_d | bb_file_d;
pub const bb_kingside: u64 = bb_file_e | bb_file_f | bb_file_g | bb_file_h;
pub const bb_center_16: u64 = (bb_file_c | bb_file_d | bb_file_e | bb_file_f) & (bb_rank_3 | bb_rank_4 | bb_rank_5 | bb_rank_6);
pub const bb_center_4: u64 = (bb_file_d | bb_file_e) & (bb_rank_4 | bb_rank_5);
pub const bb_corners: u64 = bb_a1 | bb_h1 | bb_a8 | bb_h8;

pub const rank_bitboards: [8]u64 = .{ bb_rank_1, bb_rank_2, bb_rank_3, bb_rank_4, bb_rank_5, bb_rank_6, bb_rank_7, bb_rank_8 };
pub const file_bitboards: [8]u64 = .{ bb_file_a, bb_file_b, bb_file_c, bb_file_d, bb_file_e, bb_file_f, bb_file_g, bb_file_h };

// Bitboards of each square
pub const bb_a1: u64 = 0x0000000000000001;
pub const bb_b1: u64 = 0x0000000000000002;
pub const bb_c1: u64 = 0x0000000000000004;
pub const bb_d1: u64 = 0x0000000000000008;
pub const bb_e1: u64 = 0x0000000000000010;
pub const bb_f1: u64 = 0x0000000000000020;
pub const bb_g1: u64 = 0x0000000000000040;
pub const bb_h1: u64 = 0x0000000000000080;

pub const bb_a2: u64 = 0x0000000000000100;
pub const bb_b2: u64 = 0x0000000000000200;
pub const bb_c2: u64 = 0x0000000000000400;
pub const bb_d2: u64 = 0x0000000000000800;
pub const bb_e2: u64 = 0x0000000000001000;
pub const bb_f2: u64 = 0x0000000000002000;
pub const bb_g2: u64 = 0x0000000000004000;
pub const bb_h2: u64 = 0x0000000000008000;

pub const bb_a3: u64 = 0x0000000000010000;
pub const bb_b3: u64 = 0x0000000000020000;
pub const bb_c3: u64 = 0x0000000000040000;
pub const bb_d3: u64 = 0x0000000000080000;
pub const bb_e3: u64 = 0x0000000000100000;
pub const bb_f3: u64 = 0x0000000000200000;
pub const bb_g3: u64 = 0x0000000000400000;
pub const bb_h3: u64 = 0x0000000000800000;

pub const bb_a4: u64 = 0x0000000001000000;
pub const bb_b4: u64 = 0x0000000002000000;
pub const bb_c4: u64 = 0x0000000004000000;
pub const bb_d4: u64 = 0x0000000008000000;
pub const bb_e4: u64 = 0x0000000010000000;
pub const bb_f4: u64 = 0x0000000020000000;
pub const bb_g4: u64 = 0x0000000040000000;
pub const bb_h4: u64 = 0x0000000080000000;

pub const bb_a5: u64 = 0x0000000100000000;
pub const bb_b5: u64 = 0x0000000200000000;
pub const bb_c5: u64 = 0x0000000400000000;
pub const bb_d5: u64 = 0x0000000800000000;
pub const bb_e5: u64 = 0x0000001000000000;
pub const bb_f5: u64 = 0x0000002000000000;
pub const bb_g5: u64 = 0x0000004000000000;
pub const bb_h5: u64 = 0x0000008000000000;

pub const bb_a6: u64 = 0x0000010000000000;
pub const bb_b6: u64 = 0x0000020000000000;
pub const bb_c6: u64 = 0x0000040000000000;
pub const bb_d6: u64 = 0x0000080000000000;
pub const bb_e6: u64 = 0x0000100000000000;
pub const bb_f6: u64 = 0x0000200000000000;
pub const bb_g6: u64 = 0x0000400000000000;
pub const bb_h6: u64 = 0x0000800000000000;

pub const bb_a7: u64 = 0x0001000000000000;
pub const bb_b7: u64 = 0x0002000000000000;
pub const bb_c7: u64 = 0x0004000000000000;
pub const bb_d7: u64 = 0x0008000000000000;
pub const bb_e7: u64 = 0x0010000000000000;
pub const bb_f7: u64 = 0x0020000000000000;
pub const bb_g7: u64 = 0x0040000000000000;
pub const bb_h7: u64 = 0x0080000000000000;

pub const bb_a8: u64 = 0x0100000000000000;
pub const bb_b8: u64 = 0x0200000000000000;
pub const bb_c8: u64 = 0x0400000000000000;
pub const bb_d8: u64 = 0x0800000000000000;
pub const bb_e8: u64 = 0x1000000000000000;
pub const bb_f8: u64 = 0x2000000000000000;
pub const bb_g8: u64 = 0x4000000000000000;
pub const bb_h8: u64 = 0x8000000000000000;

// --- Computed stuff ---
pub const bb_north: [Square.count]u64 = compute_direction_bitboards(.north);
pub const bb_south: [Square.count]u64 = compute_direction_bitboards(.south);
pub const bb_west: [Square.count]u64 = compute_direction_bitboards(.west);
pub const bb_east: [Square.count]u64 = compute_direction_bitboards(.east);
pub const bb_northwest: [Square.count]u64 = compute_direction_bitboards(.north_west);
pub const bb_southeast: [Square.count]u64 = compute_direction_bitboards(.south_east);
pub const bb_northeast: [Square.count]u64 = compute_direction_bitboards(.north_east);
pub const bb_southwest: [Square.count]u64 = compute_direction_bitboards(.south_west);
pub const bb_bishop: [Square.count]u64 = compute_bishop_bitboards();
pub const bb_rook: [Square.count]u64 = compute_rook_bitboards();
pub const bb_queen: [Square.count]u64 = compute_queen_bitboards();

pub const adjacent_square_masks: [Square.count]u64 = compute_adjacent_square_masks();
pub const passed_pawn_masks_white: [Square.count]u64 = compute_passed_pawn_masks_white();
pub const passed_pawn_masks_black: [Square.count]u64 = compute_passed_pawn_masks_black();
pub const adjacent_file_masks: [Square.count]u64 = compute_adjacent_file_masks();

/// Using the `Direction` enum order.
pub const direction_bitboards: [Direction.count][*]const u64 = .{
    &bb_north, &bb_east, &bb_south, &bb_west, &bb_northwest, &bb_northeast, &bb_southeast, &bb_southwest,
};


// --- Public functions ---

pub fn iterator(bitboard: u64) BitboardIterator {
    return BitboardIterator.init(bitboard);
}

pub const BitboardIterator = struct {
    state: u64,

    pub fn init(bitboard: u64) BitboardIterator {
        return .{ .state = bitboard };
    }

    pub fn next(self: *BitboardIterator) ?Square {
        if (self.state == 0) {
            return null;
        }
        const res = @ctz(self.state);
        self.state &= self.state -% 1;
        return .{ .u = @intCast(res)};
    }

    pub fn peek(self: *const BitboardIterator) ?Square {
        if (self.state == 0) {
            return null;
        }
        const res = @ctz(self.state);
        return .{ .u = @intCast(res)};
    }
};

pub fn popcnt(bitboard: u64) u7 {
    return @popCount(bitboard);
}

/// Convenience function.
pub fn ipopcnt(comptime T: anytype, bitboard: u64) T {
    return @intCast(@popCount(bitboard));
}

pub fn contains_square(bitboard: u64, sq: Square) bool {
    return funcs.test_bit_64(bitboard, sq.u);
}

/// Unsafe lsb. Assumes bitboard != 0.
pub fn first_square(bitboard: u64) Square {
    if (lib.is_paranoid) {
        assert(bitboard != 0);
    }
    //const lsb: u6 = @truncate(@ctz(bitboard));
    const lsb: u6 = @intCast(@ctz(bitboard));
    return .{ .u = lsb };
}

/// Unsafe lsb. Assumes bitboard != 0.
pub fn last_square(bitboard: u64) Square {
    if (lib.is_paranoid) {
        assert(bitboard != 0);
    }
    const msb: u6 = int(u6, 63) - int(u6, @clz(bitboard));
    return .{ .u = msb };
}

pub fn first_square_or_null(bitboard: u64) ?Square {
    if (bitboard == 0) return null;
    const lsb: u6 = @intCast(@ctz(bitboard));
    return .{ .u = lsb };
}

pub fn last_square_or_null(bitboard: u64) ?Square {
    if (bitboard == 0) return null;
    const msb: u6 = int(u6, 63) - int(u6, @clz(bitboard));
    return .{ .u = msb };
}

/// Deprecated. BitboardIterator is preferred.
pub fn bitloop(bitboard: *u64) ?Square {
    if (bitboard.* == 0) return null;
    defer bitboard.* &= (bitboard.* - 1);
    return .{ .u = @intCast(@ctz(bitboard.*)) };
}

/// Note: This requires x86-64 and the BMI2 instruction set.
pub fn get_nth_set_bit_or_null(bitboard: u64, n: u6) ?u6 {
    // Return null if we are asking for a bit that doesn't exist
    if (@popCount(bitboard) <= n) return null;

    const nth_bit = @as(u64, 1) << n;

    // PDEP (Parallel Bit Deposit) magic via inline assembly
    const isolated = asm (
        "pdep %[mask], %[val], %[out]"
        : [out] "=r" (-> u64),
        : [val] "r" (nth_bit),
          [mask] "r" (bitboard),
    );

    return @intCast(@ctz(isolated));
}

/// Note: This requires x86-64 and the BMI2 instruction set.
pub fn get_nth_set_bit(bitboard: u64, n: u6) u6 {
    const nth_bit = @as(u64, 1) << n;
    // PDEP (Parallel Bit Deposit) magic via inline assembly
    const isolated = asm (
        "pdep %[mask], %[val], %[out]"
        : [out] "=r" (-> u64),
        : [val] "r" (nth_bit),
          [mask] "r" (bitboard),
    );
    return @intCast(@ctz(isolated));
}

pub fn clear_square(bitboard: *u64, sq: Square) void {
    bitboard.* &= ~sq.to_bitboard();
}

pub fn relative_rank_456_bitboard(comptime us: Color) u64 {
    return if (us.e == .white) bb_rank_4 | bb_rank_5 | bb_rank_6 else bb_rank_3 | bb_rank_4 | bb_rank_5;
}

pub fn relative_side_bitboard(comptime us: Color) u64 {
    return if (us.e == .white) bb_white_side else bb_black_side;
}

pub fn relative_rank_bitboard(us: Color, rank: u3) u64 {
    return if (us.e == .white) rank_bitboards[rank] else rank_bitboards[7 - rank];
}

pub fn get_passed_pawn_mask(comptime us: Color, sq: Square) u64 {
    return switch (us.e) {
        .white => passed_pawn_masks_white[sq.u],
        .black => passed_pawn_masks_black[sq.u],
    };
}

pub fn forward_file(comptime us: Color, sq: Square) u64 {
    return if (us.e == .white) bb_north[sq.u] else bb_south[sq.u];
}

pub fn shift_bitboard(u: u64, comptime dir: Direction) u64 {
    return switch (dir) {
        .north      => (u & ~bb_rank_8) << 8,
        .east       => (u & ~bb_file_h) << 1,
        .south      => (u & ~bb_rank_1) >> 8,
        .west       => (u & ~bb_file_a) >> 1,
        .north_west => (u & ~bb_file_a) << 7,
        .north_east => (u & ~bb_file_h) << 9,
        .south_east => (u & ~bb_file_a) >> 7,
        .south_west => (u & ~bb_file_a) >> 9,
        else => unreachable,
    };
}

pub fn mirror_vertically(u: u64) u64 {
    return
        ( (u & 0x00000000000000ff) << 56) |
        ( (u & 0x000000000000ff00) << 40) |
        ( (u & 0x0000000000ff0000) << 24) |
        ( (u & 0x00000000ff000000) << 8 ) |
        ( (u & 0x000000ff00000000) >> 8 ) |
        ( (u & 0x0000ff0000000000) >> 24) |
        ( (u & 0x00ff000000000000) >> 40) |
        ( (u & 0xff00000000000000) >> 56);
}

// --- Computing ---

fn compute_direction_bitboards(comptime dir: Direction) [Square.count]u64 {
    @setEvalBranchQuota(8000);
    var bb: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        bb[sq.u] = sq.ray_bitboard(dir);
    }
    return bb;
}

fn compute_bishop_bitboards() [Square.count]u64 {
    var bb: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        bb[sq.u] = bb_northeast[sq.u] | bb_northwest[sq.u] | bb_southeast[sq.u] | bb_southwest[sq.u];
    }
    return bb;
}

fn compute_rook_bitboards() [Square.count]u64 {
    var bb: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        bb[sq.u] = bb_north[sq.u] | bb_south[sq.u] | bb_east[sq.u] | bb_west[sq.u];
    }
    return bb;
}

fn compute_queen_bitboards() [Square.count]u64 {
    var bb: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        bb[sq.u] =
            bb_northeast[sq.u] | bb_northwest[sq.u] | bb_southeast[sq.u] | bb_southwest[sq.u] |
            bb_north[sq.u] | bb_south[sq.u] | bb_east[sq.u] | bb_west[sq.u];
    }
    return bb;
}

fn compute_passed_pawn_masks_white() [Square.count]u64 {
    var pp: [Square.count]u64 = @splat(0);
    for (types.all_ranks) |rank| {
        for (types.all_files) |file| {
            var bb: u64 = 0;
            const sq: Square = .from_rank_file(rank, file);
            bb = bb_north[sq.u]; // square file
            if (file > types.file_a) bb |= bb_north[sq.sub(1).u]; // file forwards left of square
            if (file < types.file_h) bb |= bb_north[sq.add(1).u]; // file forwards right of square
            pp[sq.u] = bb;
        }
    }
    return pp;
}

fn compute_passed_pawn_masks_black() [Square.count]u64 {
    var pp: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        const black_sq: Square = sq.relative(.black);
        pp[black_sq.u] = mirror_vertically(passed_pawn_masks_white[sq.u]);
    }
    return pp;
}

fn compute_adjacent_square_masks() [Square.count]u64 {
    var ep: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        if (sq.next(.west)) |n| ep[sq.u] |= n.to_bitboard();
        if (sq.next(.east)) |n| ep[sq.u] |= n.to_bitboard();
    }
    return ep;
}

fn compute_adjacent_file_masks() [Square.count]u64 {
    var afm: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        const file: u3 = sq.coord.file;
        if (file > types.file_a) afm[sq.u] |= file_bitboards[file - 1];
        if (file < types.file_h) afm[sq.u] |= file_bitboards[file + 1];
    }
    return afm;
}

fn compute_backward_pawn_masks_black()[Square.count]u64 {
    var bpm: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        if (sq.coord.rank <= 1 or sq.coord.rank == 7) {
            continue;
        }
        var bb: u64 = passed_pawn_masks_white[sq.u] & ~file_bitboards[sq.coord.file];
        if (sq.coord.file > 0) {
            bb |= sq.sub(1).to_bitboard();
        }
        if (sq.coord.file < 7) {
            bb |= sq.add(1).to_bitboard();
        }
        bpm[sq.u] = bb;
    }
    return bpm;
}
