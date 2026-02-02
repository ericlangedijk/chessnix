// zig fmt: off

//! Lots of bitboards.

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const Axis = types.Axis;
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
pub const bb_white_squares: u64 = 0b01010101_10101010_01010101_10101010_01010101_10101010_01010101_10101010;
pub const bb_black_squares: u64 = ~bb_white_squares;
pub const bb_white_side: u64 = bb_rank_1 | bb_rank_2 | bb_rank_3 | bb_rank_4;
pub const bb_black_side: u64 = bb_rank_5 | bb_rank_6 | bb_rank_7 | bb_rank_8;
pub const bb_queenside: u64 = bb_file_a | bb_file_b | bb_file_d | bb_file_d;
pub const bb_kingside: u64 = bb_file_e | bb_file_f | bb_file_g | bb_file_h;
pub const bb_center: u64 = (bb_file_c | bb_file_d | bb_file_e | bb_file_f) & (bb_rank_3 | bb_rank_4 | bb_rank_5 | bb_rank_6);
pub const bb_mini_center: u64 = (bb_file_d | bb_file_e) & (bb_rank_4 | bb_rank_5);
pub const bb_colored_squares: [2]u64 = .{ bb_white_squares, bb_black_squares };
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

////////////////////////////////////////////////////////////////
// Computed stuff.
////////////////////////////////////////////////////////////////
pub const bb_north: [64]u64 = compute_direction_bitboards(.north);
pub const bb_south: [64]u64 = compute_direction_bitboards(.south);
pub const bb_west: [64]u64 = compute_direction_bitboards(.west);
pub const bb_east: [64]u64 = compute_direction_bitboards(.east);
pub const bb_northwest: [64]u64 = compute_direction_bitboards(.north_west);
pub const bb_southeast: [64]u64 = compute_direction_bitboards(.south_east);
pub const bb_northeast: [64]u64 = compute_direction_bitboards(.north_east);
pub const bb_southwest: [64]u64 = compute_direction_bitboards(.north_west);
pub const pairs: [64 * 64]SquarePair = compute_squarepairs();
pub const ep_masks: [64]u64 = compute_ep_masks(); // indexing on to-square (e2e4).
pub const passed_pawn_masks_white: [64]u64 = compute_passed_pawn_masks_white();
pub const passed_pawn_masks_black: [64]u64 = compute_passed_pawn_masks_black();
pub const adjacent_file_masks: [64]u64 = compute_adjacent_file_masks();
pub const king_areas: [64]u64 = compute_king_areas();

/// (hce) By [square]
pub const king_areas_white: [64]u64 = compute_king_areas_white();
/// (hce) By [square]
pub const king_areas_black: [64]u64 = compute_king_areas_black();
/// (hce) Pawnstorm areas from the perspective of the white king. By [white-king-square]
pub const king_pawnstorm_areas_white: [64]u64 = compute_king_pawnstorm_areas_white();
/// (hce) Pawnstorm areas from the perspective of the black king. Indexing by [black-king-square]
pub const king_pawnstorm_areas_black: [64]u64 = compute_king_pawnstorm_areas_black();

/// Using the `Direction` enum order.
pub const direction_bitboards: [8][*]const u64 = .{
    &bb_north, &bb_east, &bb_south, &bb_west, &bb_northwest, &bb_northeast, &bb_southeast, &bb_southwest,
};

////////////////////////////////////////////////////////////////
// Compute.
////////////////////////////////////////////////////////////////
fn compute_direction_bitboards(comptime dir: Direction) [64]u64 {
    @setEvalBranchQuota(8000);
    var bb: [64]u64 = @splat(0);
    for (Square.all) |sq|{
        bb[sq.u] = sq.ray_bitboard(dir);
    }
    return bb;
}

fn compute_squarepairs() [64 * 64]SquarePair {
    @setEvalBranchQuota(128000);
    var sp: [64 * 64]SquarePair = @splat(.empty);

    // Set directions.
    for (Square.all) |from| {
        inline for (Direction.all) |dir| {
            const ray = from.ray(dir);
            for (ray.slice()) |to| {
                const idx: usize = from.idx() * 64 + to.idx();
                const ori = dir.to_orientation();
                sp[idx].direction = dir;
                sp[idx].orientation = ori;
                if (ori == .diagmain or ori == .diaganti) sp[idx].axis = .diag;
                if (ori == .horizontal or ori == .vertical) sp[idx].axis = .orth;
            }
        }
    }

    // Set ray bitboards
    for (Square.all) |from| {
        for (Square.all) |to| {
            const idx: usize = from.idx() * 64 + to.idx();
            // Distance
            sp[idx].dist = funcs.square_distance(from, to);
            // Ray.
            if (sp[idx].direction) |dir| {
                const ray = from.ray(dir);
                for (ray.slice()) |sq| {
                    sp[idx].ray |= sq.to_bitboard();
                    if (sq.u == to.u) break;
                }
            }
            // In between.
            sp[idx].in_between = sp[idx].ray & ~to.to_bitboard();
        }
    }
    return sp;
}

fn compute_passed_pawn_masks_white() [64]u64 {
    var pp: [64]u64 = @splat(0);
    for (ranks) |rank| {
        for (files) |file| {
            var bb: u64 = 0;
            const sq: Square = .from_rank_file(rank, file);
            bb = bb_north[sq.u]; // square file
            if (file > file_a) bb |= bb_north[sq.sub(1).u]; // file forwards left of square
            if (file < file_h) bb |= bb_north[sq.add(1).u]; // file forwards right of square
            pp[sq.u] = bb;
        }
    }
    return pp;
}

fn compute_passed_pawn_masks_black() [64]u64 {
    var pp: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        const black_sq: Square = sq.relative(Color.BLACK);
        pp[black_sq.u] = funcs.mirror_vertically(passed_pawn_masks_white[sq.u]);
    }
    return pp;
}

fn compute_ep_masks() [64]u64 {
    var ep: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        if (sq.coord.rank == rank_4 or sq.coord.rank == rank_5) {
            if (sq.next(.west)) |n| ep[sq.u] |= n.to_bitboard();
            if (sq.next(.east)) |n| ep[sq.u] |= n.to_bitboard();
        }
    }
    return ep;
}

fn compute_king_areas() [64]u64 {
    @setEvalBranchQuota(8000);
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        // ka[sq.u] |= sq.to_bitboard();
        if (sq.next(.north))|n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.east)) |n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.south))|n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.west)) |n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.north_west))|n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.north_east))|n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.south_east))|n| ka[sq.u] |= n.to_bitboard();
        if (sq.next(.south_west))|n| ka[sq.u] |= n.to_bitboard();
    }
    return ka;
}

fn compute_adjacent_file_masks() [64]u64 {
    var afm: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        const file: u3 = sq.coord.file;
        if (file > file_a) afm[sq.u] |= file_bitboards[file - 1];
        if (file < file_h) afm[sq.u] |= file_bitboards[file + 1];
    }
    return afm;
}

fn compute_backward_pawn_masks_black()[64]u64 {
    var bpm: [64]u64 = @splat(0);
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

/// hce eval
fn compute_king_areas_white() [64]u64 {
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.WHITE, .up);
    }
    return ka;
}

/// hce eval
fn compute_king_areas_black() [64]u64 {
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.BLACK, .up);
    }
    return ka;
}

/// hce eval
fn compute_king_pawnstorm_areas_white() [64]u64 {
    var ps: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = passed_pawn_masks_white[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.BLACK, .up);
    }
    return ps;
}

/// hce
fn compute_king_pawnstorm_areas_black() [64]u64 {
    var ps: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = passed_pawn_masks_black[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.WHITE, .up);
    }
    return ps;
}


/// Information about a pair of squares.
/// * Mainly used for determining pinners and pinned pieces.
pub const SquarePair = struct {
    /// The from-to ray bitboard **excluded** the from-square and **included** the to-square.
    ray: u64 = 0,

    // The bitboard of the squares in between the 2 squares.
    in_between: u64 = 0,
    /// The from-to direction.
    direction: ?Direction = null,
    /// The from-to or to-from orientation.
    orientation: ?Orientation = null,
    /// Diagonal or orthogonal (or none).
    axis: Axis = .none,
    /// Distance
    dist: u3 = 0,

    const empty: SquarePair = .{};
};

//////////////////////////////////////////////////////////////
// Funcs
////////////////////////////////////////////////////////////////
pub fn get_squarepair(from: Square, to: Square) *const SquarePair
{
    const idx: usize = from.idx() * 64 + to.idx();
    return &pairs[idx];
}

pub fn get_passed_pawn_mask(comptime us: Color, sq: Square) u64 {
    return switch (us.e) {
        .white => passed_pawn_masks_white[sq.u],
        .black => passed_pawn_masks_black[sq.u],
    };
}
