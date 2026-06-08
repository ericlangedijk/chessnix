// zig fmt: off

//! Lots of bitboards, lots of raw functions.

const std = @import("std");
const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;

const Axis = types.Axis;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;

/// Information about a pair of squares.
pub const SquarePair = struct {
    /// The from-to ray bitboard **excluded** the from-square and **included** the to-square.
    ray: u64 = 0,
    /// The from-to direction.
    direction: ?Direction = null,
    /// The from-to or to-from orientation.
    orientation: ?Orientation = null,
    /// Diagonal or orthogonal.
    axis: Axis = .none,
    /// Distance.
    dist: u3 = 0,
    /// Manhattan distance.
    manh: u4 = 0,

    const empty: SquarePair = .{};
};

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

pub const bits: [8]u8 = .{ 1, 2, 4, 8, 16, 32, 64, 128 };

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

pub const all_ranks: [8]u3 = .{ rank_1, rank_2, rank_3, rank_4, rank_5, rank_6, rank_7, rank_8 };
pub const all_files: [8]u3 = .{ file_a, file_b, file_c, file_d, file_e, file_f, file_g, file_h };

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
pub const bb_north: [64]u64 = Computing.compute_direction_bitboards(.north);
pub const bb_east: [64]u64 = Computing.compute_direction_bitboards(.east);
pub const bb_south: [64]u64 = Computing.compute_direction_bitboards(.south);
pub const bb_west: [64]u64 = Computing.compute_direction_bitboards(.west);
pub const bb_northwest: [64]u64 = Computing.compute_direction_bitboards(.north_west);
pub const bb_northeast: [64]u64 = Computing.compute_direction_bitboards(.north_east);
pub const bb_southeast: [64]u64 = Computing.compute_direction_bitboards(.south_east);
pub const bb_southwest: [64]u64 = Computing.compute_direction_bitboards(.south_west);

const pairs: [64 * 64]SquarePair = Computing.compute_squarepairs();

pub const small_rays: [8][8]u8 = Computing.compute_small_rays();

pub const ep_masks: [64]u64 = Computing.compute_ep_masks(); // indexing on to-square (e2e4).
pub const passed_pawn_masks_white: [64]u64 = Computing.compute_passed_pawn_masks_white();
pub const passed_pawn_masks_black: [64]u64 = Computing.compute_passed_pawn_masks_black();
pub const adjacent_file_masks: [64]u64 = Computing.compute_adjacent_file_masks();
pub const king_areas: [64]u64 = Computing.compute_king_areas();

pub const king_areas_white: [64]u64 = Computing.compute_king_areas_white();
pub const king_areas_black: [64]u64 = Computing.compute_king_areas_black();
pub const king_pawnstorm_areas_white: [64]u64 = Computing.compute_king_pawnstorm_areas_white();
pub const king_pawnstorm_areas_black: [64]u64 = Computing.compute_king_pawnstorm_areas_black();

/// Using the `Direction` enum order.
pub const direction_bitboards: [8][*]const u64 = .{
    &bb_north, &bb_east, &bb_south, &bb_west, &bb_northwest, &bb_northeast, &bb_southeast, &bb_southwest,
};

/// Just a wrapper to easily code-collapse this thing.
const Computing = struct {

    fn compute_direction_bitboards(comptime dir: Direction) [64]u64 {
        @setEvalBranchQuota(8000);
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
            ptr.* = sq.ray_bitboard(dir);
        }
        return result;
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
                sp[idx].manh = funcs.manhattan_distance(from, to);
                // Ray.
                if (sp[idx].direction) |dir| {
                    const ray = from.ray(dir);
                    for (ray.slice()) |sq| {
                        sp[idx].ray |= sq.to_bitboard();
                        if (sq.u == to.u) break;
                    }
                }
                // #deprecated
                // In between.
                // sp[idx].in_between = sp[idx].ray & ~to.to_bitboard();
            }
        }
        return sp;
    }

    fn compute_small_rays() [8][8]u8 {
        // Use the already computed first rank squarepairs.
        var result: [8][8]u8 = @splat(@splat(0));
        for (0..8) |f| {
            const from: u3 = @intCast(f);
            for (0..8) |t| {
                const to: u3 = @intCast(t);
                const from_sq: Square = Square.from_rank_file(0, from);
                const to_sq: Square = Square.from_rank_file(0, to);
                const pair: *const SquarePair = get_squarepair(from_sq, to_sq);
                const mask: u64 = pair.ray;
                const v: u8 = @intCast(mask & 0xff);
                result[f][t] = v;
            }
        }
        return result;
    }

    fn compute_passed_pawn_masks_white() [64]u64 {
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
            const file: u3 = sq.coord.file;
            var bb: u64 = bb_north[sq.u];
            if (file > file_a) bb |= bb_north[sq.sub(1).u]; // file forwards left of square
            if (file < file_h) bb |= bb_north[sq.add(1).u]; // file forwards right of square
            ptr.* = bb;
        }
        return result;
    }

    fn compute_passed_pawn_masks_black() [64]u64 {
        var result: [64]u64 = @splat(0);
        for (Square.all) |sq| {
            const black_sq: Square = sq.relative(Color.black);
            result[black_sq.u] = mirror_vertically(passed_pawn_masks_white[sq.u]);
        }
        return result;
    }

    fn compute_ep_masks() [64]u64 {
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
            if (sq.coord.rank == rank_4 or sq.coord.rank == rank_5) {
                if (sq.next(.west)) |n| ptr.* |= n.to_bitboard();
                if (sq.next(.east)) |n| ptr.* |= n.to_bitboard();
            }
        }
        return result;
    }

    fn compute_king_areas() [64]u64 {
        @setEvalBranchQuota(8000);
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
            // ka[sq.u] |= sq.to_bitboard(); // TODO: include or not??
            if (sq.next(.north))|n| ptr.* |= n.to_bitboard();
            if (sq.next(.east)) |n| ptr.* |= n.to_bitboard();
            if (sq.next(.south))|n| ptr.* |= n.to_bitboard();
            if (sq.next(.west)) |n| ptr.* |= n.to_bitboard();
            if (sq.next(.north_west))|n| ptr.* |= n.to_bitboard();
            if (sq.next(.north_east))|n| ptr.* |= n.to_bitboard();
            if (sq.next(.south_east))|n| ptr.* |= n.to_bitboard();
            if (sq.next(.south_west))|n| ptr.* |= n.to_bitboard();
        }
        return result;
    }

    fn compute_adjacent_file_masks() [64]u64 {
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
            const file: u3 = sq.coord.file;
            if (file > file_a) ptr.* |= file_bitboards[file - 1];
            if (file < file_h) ptr.* |= file_bitboards[file + 1];
        }
        return result;
    }

    fn compute_backward_pawn_masks_black()[64]u64 {
        var result: [64]u64 = @splat(0);
        for (Square.all, &result) |sq, *ptr| {
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
            ptr.* = bb;
        }
        return result;
    }

    /// hce eval
    fn compute_king_areas_white() [64]u64 {
        var ka: [64]u64 = @splat(0);
        for (Square.all) |sq| {
            ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
            // Go 1 rank further.
            ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.white, .up);
        }
        return ka;
    }

    /// hce eval
    fn compute_king_areas_black() [64]u64 {
        var ka: [64]u64 = @splat(0);
        for (Square.all) |sq| {
            ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
            // Go 1 rank further.
            ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.black, .up);
        }
        return ka;
    }

    /// hce eval
    fn compute_king_pawnstorm_areas_white() [64]u64 {
        var ps: [64]u64 = @splat(0);
        for (Square.all) |sq| {
            ps[sq.u] = passed_pawn_masks_white[sq.u];
            // Include the squares next to the king.
            ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.black, .up);
        }
        return ps;
    }

    /// hce
    fn compute_king_pawnstorm_areas_black() [64]u64 {
        var ps: [64]u64 = @splat(0);
        for (Square.all) |sq| {
            ps[sq.u] = passed_pawn_masks_black[sq.u];
            // Include the squares next to the king.
            ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.white, .up);
        }
        return ps;
    }
};

pub fn get_squarepair(from: Square, to: Square) *const SquarePair {
    const idx: usize = from.idx() * 64 + to.idx();
    return &pairs[idx];
}

/// Convenience function.
pub fn dist(a: Square, b: Square) u3 {
    return get_squarepair(a, b).dist;
}

/// Convenience function.
pub fn manh(a: Square, b: Square) u4 {
    return get_squarepair(a, b).manh;
}

pub fn get_passed_pawn_mask(comptime us: Color, sq: Square) u64 {
    return switch (us.e) {
        .white => passed_pawn_masks_white[sq.u],
        .black => passed_pawn_masks_black[sq.u],
    };
}

pub fn relative_rank(us: Color, rank: u3) u3 {
    return if (us.e == .white) rank else 7 - rank;
}

pub fn forward_file(comptime us: Color, sq: Square) u64 {
    return if (us.e == .white) bb_north[sq.u] else bb_south[sq.u];
}

pub fn relative_rank_7(us: Color) u3 {
    return if (us.e == .white) rank_7 else rank_2;
}

pub fn relative_rank_3_bitboard(us: Color) u64 {
    return if (us.e == .white) bb_rank_3 else bb_rank_6;
}

pub fn relative_rank_7_bitboard(us: Color) u64 {
    return if (us.e == .white) bb_rank_7 else bb_rank_2;
}

pub fn relative_rank_8_bitboard(us: Color) u64 {
    return if (us.e == .white) bb_rank_8 else bb_rank_1;
}

pub fn relative_rank_bb(us: Color, rank: u3) u64 {
    return if (us.e == .white) rank_bitboards[rank] else rank_bitboards[7 - rank];
}

/// Shift the bitboard in a certain direction taking borders into account.
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
    };
}

pub fn mirror_vertically(u: u64) u64 {
    // TODO: @bitreverse?
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

pub fn popcnt(bitboard: u64) u7 {
    return @popCount(bitboard);
}

pub fn contains_square(bitboard: u64, sq: Square) bool {
    return bitboard & sq.to_bitboard() != 0;
}

pub fn clear_square(bitboard: *u64, sq: Square) void {
    bitboard.* &= ~sq.to_bitboard();
}

pub fn first_square_or_null(bitboard: u64) ?Square {
    if (bitboard == 0) return null;
    return .{ .u = @intCast(@ctz(bitboard)) };
}

pub fn last_square_or_null(bitboard: u64) ?Square {
    if (bitboard == 0) return null;
    return .{ .u = @intCast(63 - @clz(bitboard)) };
}

/// Unsafe lsb. Assumes bitboard != 0.
pub fn first_square(bitboard: u64) Square {
    if (comptime lib.is_paranoid) {
        assert(bitboard != 0);
    }
    const lsb: u6 = @intCast(@ctz(bitboard));
    return .{ .u = lsb };
}

/// Fastest bit loop I can produce.
pub fn bitloop(bitboard: *u64) ?Square {
    if (bitboard.* == 0) return null;
    defer bitboard.* &= (bitboard.* - 1);
    return .{ .u = @intCast(@ctz(bitboard.*)) };
}

/// Not used.
/// Note: This requires x86-64 and the BMI2 instruction set.
fn get_nth_set_bit_or_null(bitboard: u64, n: u6) ?u6 {
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

/// Not used.
/// Note: This requires x86-64 and the BMI2 instruction set.
fn get_nth_set_bit(bitboard: u64, n: u6) u6 {
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

/// Not used.
pub fn pext(src: u64, mask: u64) u64 {
    if (@inComptime() or !std.Target.x86.featureSetHas(@import("builtin").cpu.model.features, .bmi2)) {
        var res: u64 = 0;
        var i: u6, var m: u64 = .{ 0, mask };
        while (m != 0) {
            res |= ((src >> @intCast(@ctz(m))) & 1) << i;
            i += 1;
            m &= m - 1;
        }
        return res;
    } else return asm ("pextq %[mask], %[src], %[res]"
        : [res] "=r" (-> u64),
        : [src] "r" (src),
          [mask] "r" (mask),
    );
}

/// Not used.
pub fn pdep(src: u64, mask: u64) u64 {
    if (@inComptime() or !std.Target.x86.featureSetHas(@import("builtin").cpu.model.features, .bmi2)) {
        var res: u64 = 0;
        var bit: u6 = 0;
        var m: u64 = mask;
        while (m != 0) {
            if (((src >> bit) & 1) != 0) {
                res |= m & -%m;
            }
            m &= m - 1;
            bit += 1;
        }
        return res;
    } else return asm ("pdepq %[mask], %[src], %[res]"
        : [res] "=r" (-> u64),
        : [src] "r" (src),
          [mask] "r" (mask),
    );
}

