// zig fmt: off

//! Lots of bitboards.
//! NOTE: initialize *after* attacks.zig.

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;

////////////////////////////////////////////////////////////////
// Const stuff
////////////////////////////////////////////////////////////////

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
pub const bb_border_inner = (bb_rank_2 | bb_rank_7 | bb_file_b | bb_file_g) & ~bb_border;
pub const bb_center = bb_full & ~bb_border & ~bb_border_inner;
pub const bb_mini_center = bb_e4 | bb_d4 | bb_e5 | bb_d5;
pub const bb_black_squares: u64 = 0b01010101_10101010_10101010_01010101_10101010_01010101_10101010_01010101;
pub const bb_white_squares: u64 = ~bb_black_squares;

pub const rank_bitboards: [8]u64 = .{ bb_rank_1, bb_rank_2, bb_rank_3, bb_rank_4, bb_rank_5, bb_rank_6, bb_rank_7, bb_rank_8 };
pub const file_bitboards: [8]u64 = .{ bb_file_a, bb_file_b, bb_file_c, bb_file_d, bb_file_e, bb_file_f, bb_file_g, bb_file_h };

// rank and file indexes
pub const rank_1 : u3 = 0;
pub const rank_2 : u3 = 1;
pub const rank_3 : u3 = 2;
pub const rank_4 : u3 = 3;
pub const rank_5 : u3 = 4;
pub const rank_6 : u3 = 5;
pub const rank_7 : u3 = 6;
pub const rank_8 : u3 = 7;

pub const file_a : u3 = 0;
pub const file_b : u3 = 1;
pub const file_c : u3 = 2;
pub const file_d : u3 = 3;
pub const file_e : u3 = 4;
pub const file_f : u3 = 5;
pub const file_g : u3 = 6;
pub const file_h : u3 = 7;

pub const ranks: [8]u3 = .{ rank_1, rank_2, rank_3, rank_4, rank_5, rank_6, rank_7, rank_8 };
pub const files: [8]u3 = .{ file_a, file_b, file_c, file_d, file_e, file_f, file_g, file_h };

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

// Array of bitboards of each square
pub const square_bitboards: [64]u64 = .{
    bb_a1, bb_b1, bb_c1, bb_d1, bb_e1, bb_f1, bb_g1, bb_h1,
    bb_a2, bb_b2, bb_c2, bb_d2, bb_e2, bb_f2, bb_g2, bb_h2,
    bb_a3, bb_b3, bb_c3, bb_d3, bb_e3, bb_f3, bb_g3, bb_h3,
    bb_a4, bb_b4, bb_c4, bb_d4, bb_e4, bb_f4, bb_g4, bb_h4,
    bb_a5, bb_b5, bb_c5, bb_d5, bb_e5, bb_f5, bb_g5, bb_h5,
    bb_a6, bb_b6, bb_c6, bb_d6, bb_e6, bb_f6, bb_g6, bb_h6,
    bb_a7, bb_b7, bb_c7, bb_d7, bb_e7, bb_f7, bb_g7, bb_h7,
    bb_a8, bb_b8, bb_c8, bb_d8, bb_e8, bb_f8, bb_g8, bb_h8,
};

////////////////////////////////////////////////////////////////
// Initialized stuff
////////////////////////////////////////////////////////////////

var bb_north: [64]u64 = @splat(0);
var bb_south: [64]u64 = @splat(0);
var bb_west: [64]u64 = @splat(0);
var bb_east: [64]u64 = @splat(0);
var bb_northwest: [64]u64 = @splat(0);
var bb_southeast: [64]u64 = @splat(0);
var bb_northeast: [64]u64 = @splat(0);
var bb_southwest: [64]u64 = @splat(0);

pub const ptr_bb_north: [*]const u64 = &bb_north;
pub const ptr_bb_south: [*]const u64 = &bb_south;
pub const ptr_bb_west: [*]const u64 =  &bb_west;
pub const ptr_bb_east: [*]const u64 =  &bb_east;
pub const ptr_bb_northwest: [*]const u64 =  &bb_northwest;
pub const ptr_bb_southeast: [*]const u64 = &bb_southeast;
pub const ptr_bb_northeast: [*]const u64 = &bb_northeast;
pub const ptr_bb_southwest: [*]const u64 =  &bb_southwest;

/// Using the `Orientation` enum order.
pub const direction_bitboards: [8][*]const u64 = .{
    ptr_bb_north,
    ptr_bb_east,
    ptr_bb_south,
    ptr_bb_west,
    ptr_bb_northwest,
    ptr_bb_northeast,
    ptr_bb_southeast,
    ptr_bb_southwest,
};

var pairs: [64 * 64]SquarePair = @splat(SquarePair.empty);
pub const ptr_pairs: [*]SquarePair = &pairs;

/// Information about a pair of squares.
/// * Mainly used for determining pinners and pinned pieces.
pub const SquarePair = struct
{
    const empty: SquarePair = .{};

    /// The bitboard of the squares in between two squares.
    in_between_bitboard: u64 = 0,
    /// The from-to direction.
    direction: ?Direction = null,
    /// The from-to or to-from orientation.
    orientation: ?Orientation = null,
    /// A mask for quick checking if a piece can cover this direction.\
    /// We can `and` the piecetype's numeric value with this mask.
    ///
    /// * orthoganal  = 100
    /// * diagonal    = 001
    /// * bishop      = 011 (value 3)
    /// * rook        = 100 (value 4)
    /// * queen       = 101 (value 5)
    ///
    mask: u3 = 0,
    /// The manhattan distance.
    distance: u3 = 0,
};

var isolated_pawn_masks: [64]u64 = @splat(0);
var passed_pawn_masks_white: [64]u64 = @splat(0);
var passed_pawn_masks_black: [64]u64 = @splat(0);
var ep_masks: [64]u64 =  @splat(0);
var king_areas: [64]u64 = @splat(0);

const ptr_passed_pawn_masks_white: [*]const u64 = &passed_pawn_masks_white;
const ptr_passed_pawn_masks_black: [*]const u64 = &passed_pawn_masks_black;
const ptr_ep_masks: [*]const u64 = &ep_masks;
const ptr_king_areas: [*]const u64 = &king_areas;

////////////////////////////////////////////////////////////////
// Funcs
////////////////////////////////////////////////////////////////

pub fn initialize() void {
    for (Square.all) |sq| {
        // Raw direction bitboards.
        bb_north[sq.u] = sq.ray_bitboard(.north);
        bb_east[sq.u] = sq.ray_bitboard(.east);
        bb_south[sq.u] = sq.ray_bitboard(.south);
        bb_west[sq.u] = sq.ray_bitboard(.west);
        bb_northwest[sq.u] = sq.ray_bitboard(.north_west);
        bb_northeast[sq.u] = sq.ray_bitboard(.north_east);
        bb_southeast[sq.u] = sq.ray_bitboard(.south_east);
        bb_southwest[sq.u] = sq.ray_bitboard(.south_west);
    }

    // Set directions.
    for (Square.all) |from| {
        inline for (Direction.all) |dir| {
            const ray = from.ray(dir);
            for (ray.slice()) |to| {
                const pair: *SquarePair = get_squarepair_mut(from, to);
                const ori = dir.to_orientation();
//                pair.distance = calc_distance(from, to);
                pair.direction = dir;
                pair.orientation = ori;
                if (ori == .diagmain or ori == .diaganti) pair.mask |= 0b001;
                if (ori == .horizontal or ori == .vertical) pair.mask |= 0b100;
            }
        }
    }

    // Set in-between bitboards
    for (Square.all) |from| {
        for (Square.all) |to| {
            const pair: *SquarePair = get_squarepair_mut(from, to);
            pair.distance = calc_distance(from, to);

            if (pair.direction) |dir| {
                const ray = from.ray(dir);
                for (ray.slice()) |sq| {
                    if (sq.u == to.u) break;
                    pair.in_between_bitboard |= sq.to_bitboard();
                }
            }
        }
    }

    // Passed pawn masks. TODO: just loop squares.
    for (ranks) |rank| {
        for (files) |file| {
            var bb: u64 = 0;
            const sq: Square = .from_rank_file(rank, file);
            const black_sq: Square = sq.relative(Color.BLACK);

            bb = ptr_bb_north[sq.u];
            if (file > file_a) bb |= ptr_bb_north[sq.sub(1).u];
            if (file < file_h) bb |= ptr_bb_north[sq.add(1).u];

            // Passed pawn mask.
            passed_pawn_masks_white[sq.u] = bb;
            passed_pawn_masks_black[black_sq.u] = funcs.mirror_vertically(bb);

            // Ep mask.
            if (rank == rank_4 or rank == rank_5) {
                if (sq.next(.west)) |n| ep_masks[sq.u] |= n.to_bitboard();
                if (sq.next(.east)) |n| ep_masks[sq.u] |= n.to_bitboard();
            }

            // Isolated pawn mask.
            if (file > file_a) isolated_pawn_masks[sq.u] |= file_bitboards[file - 1];
            if (file < file_h) isolated_pawn_masks[sq.u] |= file_bitboards[file + 1];
        }
    }

    // King areas's
    for (Square.all) |sq| {
        const rank = sq.rank();
        const file = sq.file();

        king_areas[sq.u] = sq.to_bitboard();

        // King area.
        king_areas[sq.u] |= attacks.get_king_attacks(sq);

        if (rank == 0) king_areas[sq.u] |= attacks.get_king_attacks(sq.add(8));
        if (rank == 7) king_areas[sq.u] |= attacks.get_king_attacks(sq.sub(8));

        if (file == 0) king_areas[sq.u] |= attacks.get_king_attacks(sq.add(1));
        if (file == 7) king_areas[sq.u] |= attacks.get_king_attacks(sq.sub(1));

        if (sq.e == Square.A1.e) king_areas[sq.u] |= Square.C3.to_bitboard();
        if (sq.e == Square.A8.e) king_areas[sq.u] |= Square.C6.to_bitboard();
        if (sq.e == Square.H1.e) king_areas[sq.u] |= Square.F3.to_bitboard();
        if (sq.e == Square.H8.e) king_areas[sq.u] |= Square.F6.to_bitboard();
    }

}

fn calc_distance(a: Square, b: Square) u3
{
    const ar: i32 = a.rank();
    const br: i32 = b.rank();

    const af: i32 = a.file();
    const bf: i32 = b.file();

    const d: u32 = @max
    (
        @abs(ar - br),
        @abs(af - bf)
    );
    return @truncate(@abs(d));
}

fn get_squarepair_mut(from: Square, to: Square) *SquarePair
{
    const idx: usize = from.idx() * 64 + to.idx();
    return &ptr_pairs[idx];
}

pub fn get_squarepair(from: Square, to: Square) *const SquarePair
{
    const idx: usize = from.idx() * 64 + to.idx();
    return &ptr_pairs[idx];
}

pub fn in_between_bitboard(from: Square, to: Square) u64
{
    return get_squarepair(from, to).in_between_bitboard;
}


pub fn get_isolated_pawn_mask(sq: Square) u64 {
    return isolated_pawn_masks[sq.u];
}

pub fn get_passed_pawn_mask(comptime us: Color, sq: Square) u64 {
    return switch (us.e) {
        .white => ptr_passed_pawn_masks_white[sq.u],
        .black => ptr_passed_pawn_masks_black[sq.u],
    };
}

/// `to` is the to-square of the  double pushed pawn (e4).
pub fn get_ep_mask(to: Square) u64 {
    return ptr_ep_masks[to.u];
}

pub fn get_king_area(to: Square) u64
{
    return ptr_king_areas[to.u];
}


