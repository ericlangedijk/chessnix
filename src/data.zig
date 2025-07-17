//! All bitboards + magics + functions for move generation.

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const console = @import("console.zig");
const funcs = @import("funcs.zig");
const bits = @import("bits.zig");

const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;

const assert = std.debug.assert;

pub fn initialize() void
{
    // First determine byte-based local sliding attack masks, used for the big sliding attack tables.
    // Determine for each bit-position the 8-bit attackmask for each occuption. The from-bitpos is never included.
    // We use a compressed index becuase borders are excluded.
    var sliding_attacks: [8][64]u8 = std.mem.zeroes([8][64]u8);

    for (0..8) |b|
    {
        const bitpos: u3 = @truncate(b);
        for (0..255) |o|
        {
            const occ: u8 = @truncate(o);
            var attackmask: u8 = 0;

            // Scan bits forwards
            if (bitpos < 7)
            {
                for (bitpos + 1..8) |bit|
                {
                    const i: u3 = @truncate(bit);
                    const mask: u8 = @as(u8, 1) << i;
                    attackmask |= mask;
                    if (occ & mask != 0) break; // occupied
                }
            }
            // Scan bits backwards.
            if (bitpos > 0)
            {
                var i: u3 = bitpos;
                while (true)
                {
                    i -= 1;
                    const mask: u8 = @as(u8, 1) << i;
                    attackmask |= mask;
                    if (i == 0) break;
                    if (occ & mask != 0) break;  // occupied
                }
            }

            // compress the index
            const index : u8 = (occ & occ_index_mask) >> 1;
            sliding_attacks[bitpos][index] = attackmask;
        }
    }

    for (Square.all) |sq|
    {
        const idx: usize = sq.u;
        const file: u3 = sq.file();
        const rank: u3 = sq.rank();

        // Pawn hits. Fake pawnhits are generated for the first and last rank (for pawn tricks).
        if (sq.next(.north_east))|n| pawn_attacks_white[sq.u] |= n.to_bitboard();
        if (sq.next(.north_west))|n| pawn_attacks_white[sq.u] |= n.to_bitboard();
        if (sq.next(.south_east))|n| pawn_attacks_black[sq.u] |= n.to_bitboard();
        if (sq.next(.south_west))|n| pawn_attacks_black[sq.u] |= n.to_bitboard();

        // Knight.
        if (sq.next_twice(.north_west, .west))  |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_west, .north)) |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_east, .north)) |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_east, .east))  |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_east, .east))  |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_east, .south)) |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_west, .south)) |n| knight_attacks[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_west, .west))  |n| knight_attacks[sq.u] |= n.to_bitboard();

        // king
        if (sq.next(.north))|n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.east)) |n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.south))|n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.west)) |n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.north_west))|n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.north_east))|n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.south_east))|n| king_attacks[sq.u] |= n.to_bitboard();
        if (sq.next(.south_west))|n| king_attacks[sq.u] |= n.to_bitboard();

        // Raw direction bitboards.
        for (sq.ray(.north).slice()) |q| bb_north[sq.u] |= q.to_bitboard();
        for (sq.ray(.south).slice()) |q| bb_south[sq.u] |= q.to_bitboard();
        for (sq.ray(.west).slice())  |q| bb_west[sq.u] |= q.to_bitboard();
        for (sq.ray(.east).slice())  |q| bb_east[sq.u] |= q.to_bitboard();

        for (sq.rays(&.{.north, .east}).slice()) |q| bb_northeast[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.south, .west}).slice()) |q| bb_southwest[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.north, .west}).slice())  |q| bb_northwest[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.south, .east}).slice())  |q| bb_southeast[sq.u] |= q.to_bitboard();

        // Masks without borders and without square itself.
        for (sq.rays(&.{.west, .east}).slice()) |q| rank_masks[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.north, .south}).slice()) |q| file_masks[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.north_west, .south_east}).slice()) |q| diag_main_masks[sq.u] |= q.to_bitboard();
        for (sq.rays(&.{.north_east, .south_west}).slice()) |q| diag_anti_masks[sq.u] |= q.to_bitboard();
        rank_masks[sq.u] &= ~(bitboards.bb_file_a | bitboards.bb_file_h);
        file_masks[sq.u] &= ~(bitboards.bb_rank_1 | bitboards.bb_rank_1);
        diag_main_masks[sq.u] &= ~bitboards.bb_borders;
        diag_anti_masks[sq.u] &= ~bitboards.bb_borders;

        // Magics for each square, deduced from the precalculated ones.
        file_magics[sq.u] = magics.file_magics[file];
        diag_main_magics[sq.u] = magics.diag_main_magics[@as(u8, file) + rank];
        diag_anti_magics[sq.u] = magics.diag_anti_magics[@as(u8, file) + (7 - rank)];

        // Rank attacks.
        for (0..64) |occ|
        {
            const attack: u8 = sliding_attacks[file][occ];
            rank_attacks[idx * 64 + occ] = @as(u64, attack) << (@as(u6, rank) * 8); // shift into the correct rank
        }

        // File attacks.
        for (0..64) |occ|
        {
            const attackmask: u8 = sliding_attacks[7 - rank][occ];
            var bitboard: u64 = 0;
            for (0..8) |i|
            {
                const bitpos: u3 = @truncate(i);
                if (bits.test_bit_u8(attackmask, bitpos))
                {
                    const square: Square = .from_rank_file(7 - bitpos, file);
                    bitboard |= square.to_bitboard();
                }
            }
            file_attacks[idx * 64 + occ] = bitboard;
        }

        // Diagonal main attacks
        for (0..64) |occ|
        {
            const offset: u3 = @min(7 - rank, file);
            const attackmask: u8 = sliding_attacks[offset][occ];
            var bitboard: u64 = 0;

            // scan northwest (backwards from iffset)
            var q: Square = sq;
            var bitpos: u3 = offset;
            while (true)
            {
                if (bits.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.north_west) orelse break;
                assert(bitpos > 0);
                bitpos -= 1;
            }

            // scan southeast (forwards from offset)
            q = sq;
            bitpos = offset;
            while(true)
            {
                if (bits.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.south_east) orelse break;
                assert(bitpos < 7);
                bitpos += 1;
            }
            diag_main_attacks[sq.idx() * 64 + occ] = bitboard;
        }

        // Diagonal anti attacks.
        for (0..64) |occ|
        {
            const offset: u3 = @min(rank, file);
            const attackmask: u8 = sliding_attacks[offset][occ];
            var bitboard: u64 = 0;

            // scan southwest (backwards from offset)
            var q: Square = sq;
            var bitpos: u3 = offset;
            while (true)
            {
                if (bits.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.south_west) orelse break;
                assert(bitpos > 0);
                bitpos -= 1;
            }

            // scan northeast (forwards from offset)
            q = sq;
            bitpos = offset;
            while (true)
            {
                if (bits.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                //if (bitpos == 7) break;
                q = q.next(.north_east) orelse break;
                assert(bitpos < 7);
                bitpos += 1;
            }
            diag_anti_attacks[sq.idx() * 64 + occ] = bitboard;
        }
    }
}

/// This mask is used internally for compressed index. The borders are not needed in the story.
const occ_index_mask: u8 = 0b01111110;

/// Precomputed magics.
const magics = struct
{
    const file_magics: [8]u64 =
    .{
        0x8040201008040200,  // a-file
        0x4020100804020100,  // b-file
        0x2010080402010080,  // c-file
        0x1008040201008040,  // d-file
        0x0804020100804020,  // e-file
        0x0402010080402010,  // f-file
        0x0201008040201008,  // g-file
        0x0100804020100804   // h-file
    };

    const diag_main_magics: [15]u64 =
    .{
        0x0,                 //  0: a1
        0x0,                 //  1: a2-b1
        0x0101010101010100,  //  2: a3-c1
        0x0101010101010100,  //  3: a4-d1
        0x0101010101010100,  //  4: a5-e1
        0x0101010101010100,  //  5: a6-f1
        0x0101010101010100,  //  6: a7-g1
        0x0101010101010100,  //  7: a8-h1
        0x0080808080808080,  //  8: b8-h2
        0x0040404040404040,  //  9: c8-h3
        0x0020202020202020,  // 10: d8-h4
        0x0010101010101010,  // 11: e8-h5
        0x0008080808080808,  // 12: f8-h6
        0x0,                 // 13: g8-h7
        0x0                  // 14: h8
    };

    const diag_anti_magics: [15]u64 =
    .{
        0x0,                //  0: a8
        0x0,                //  1: a7-b8
        0x0101010101010100, //  2: a6-c8
        0x0101010101010100, //  3: a5-d8
        0x0101010101010100, //  4: a4-e8
        0x0101010101010100, //  5: a3-f8
        0x0101010101010100, //  6: a2-g8
        0x0101010101010100, //  7: a1-h8
        0x8080808080808000, //  8: b1-h7
        0x4040404040400000, //  9: c1-h6
        0x2020202020000000, // 10: d1-h5
        0x1010101000000000, // 11: e1-h4
        0x0808080000000000, // 12: f1-h3
        0x0,                // 13: g1-h2
        0x0                 // 14: h1
    };
};

// Movgen
var file_magics: [64]u64 = @splat(0);
var diag_main_magics: [64]u64 = @splat(0);
var diag_anti_magics: [64]u64 = @splat(0);

var rank_masks: [64]u64 = @splat(0);
var file_masks: [64]u64 = @splat(0);
var diag_main_masks: [64]u64 = @splat(0);
var diag_anti_masks: [64]u64 =  @splat(0);

var pawn_attacks_white: [64]u64 = @splat(0);
var pawn_attacks_black: [64]u64 = @splat(0);
var knight_attacks: [64]u64 =  @splat(0);
var king_attacks: [64]u64 = @splat(0);
var rank_attacks: [64 * 64]u64 = @splat(0);
var file_attacks: [64 * 64]u64 = @splat(0);
var diag_main_attacks: [64 * 64]u64 = @splat(0);
var diag_anti_attacks: [64 * 64]u64 = @splat(0);

// Raw
var bb_north: [64]u64 = @splat(0);
var bb_south: [64]u64 = @splat(0);
var bb_west: [64]u64 = @splat(0);
var bb_east: [64]u64 = @splat(0);
var bb_northwest: [64]u64 = @splat(0);
var bb_southeast: [64]u64 = @splat(0);
var bb_northeast: [64]u64 = @splat(0);
var bb_southwest: [64]u64 = @splat(0);

// Movgen. (Zig still sucks with array access, hence the pointers. When directly accessing the vars above an enormous amount of @memcpy calls are done.)
pub const ptr_file_magics: [*]u64 = &file_magics;
pub const ptr_diag_main_magics: [*]u64 = &diag_main_magics;
pub const ptr_diag_anti_magics: [*]u64 = &diag_anti_magics;

pub const ptr_rank_masks: [*]u64 = &rank_masks;
pub const ptr_file_masks: [*]u64 = &file_masks;
pub const ptr_diag_main_masks: [*]u64 = &diag_main_masks;
pub const ptr_diag_anti_masks: [*]u64 = &diag_anti_masks;

pub const ptr_pawn_attacks_white: [*]u64 = &pawn_attacks_white;
pub const ptr_pawn_attacks_black: [*]u64 = &pawn_attacks_black;
pub const ptr_knight_attacks: [*]u64 = & knight_attacks;
pub const ptr_king_attacks: [*]u64 = &king_attacks;
pub const ptr_rank_attacks: [*]u64 = &rank_attacks;
pub const ptr_file_attacks: [*]u64 = &file_attacks;
pub const ptr_diag_main_attacks: [*]u64 = &diag_main_attacks;
pub const ptr_diag_anti_attacks: [*]u64 = &diag_anti_attacks;

// Raw
pub const ptr_bb_north: [*]u64 = &bb_north;
pub const ptr_bb_south: [*]u64 = &bb_south;
pub const ptr_bb_west: [*]u64 =  &bb_west;
pub const ptr_bb_east: [*]u64 =  &bb_east;
pub const ptr_bb_northwest: [*]u64 =  &bb_northwest;
pub const ptr_bb_southeast: [*]u64 = &bb_southeast;
pub const ptr_bb_northeast: [*]u64 = &bb_northeast;
pub const ptr_bb_southwest: [*]u64 =  &bb_southwest;

/// Returns the raw index. Always <= 255.
fn index_of_original(comptime ori: Orientation, sq: Square, occ: u64) u64
{
    const idx: usize = sq.idx();

    return switch(ori)
    {
        .horizontal => ((occ >> (sq.u & 0b111000)) & occ_index_mask) >> 1, // just shifted to rank zero.
        .vertical => ((occ & ptr_file_masks[idx]) *% ptr_file_magics[idx]) >> 57,
        .diagmain => ((occ & ptr_diag_main_masks[idx]) *% ptr_diag_main_magics[idx]) >> 57,
        .diaganti => ((occ & ptr_diag_anti_masks[idx]) *% ptr_diag_anti_magics[idx]) >> 57,
    };
}

/// Calculate File Magic: calculating is faster than a table lookup.
inline fn compute_file_magic(sq: Square) u64
{
   const m = @as(u64, 0x8040201008040200);
   return m >> (sq.u & 7); // (sq.u & 7) == sq.file();
}

/// Correct but slower than lookup
// inline fn compute_diagmain_magic_slower(sq: Square) u64
// {
//     const index = @as(u4, sq.file()) + sq.rank();
//     return magics.diag_main_magics[index];
// }

fn index_of(comptime ori: Orientation, sq: Square, occ: u64) u64
{
    return switch(ori)
    {
        .horizontal => ((occ >> (sq.u & 0b111000)) & occ_index_mask) >> 1, // just shifted to rank zero.
        .vertical => ((occ & ptr_file_masks[sq.u]) *% compute_file_magic(sq)) >> 57,
        .diagmain => ((occ & ptr_diag_main_masks[sq.u]) *% ptr_diag_main_magics[sq.u]) >> 57,
        .diaganti => ((occ & ptr_diag_anti_masks[sq.u]) *% ptr_diag_anti_magics[sq.u]) >> 57,
    };
}

/// Returns attacks for one direction.
pub fn attacks_of(comptime ori: Orientation, sq: Square, occ: u64) u64
{
    const offset: u64 = sq.idx() * 64;
    const raw: u64 =  index_of(ori, sq, occ);

    return switch (ori)
    {
        .horizontal => ptr_rank_attacks[offset + raw],
        .vertical   => ptr_file_attacks[offset + raw],
        .diagmain   => ptr_diag_main_attacks[offset + raw],
        .diaganti   => ptr_diag_anti_attacks[offset + raw],
    };
}

/// A little speedup for combined attacks. We only need to calculate the offset once.
fn combined_attacks_of(comptime orientations: []const Orientation, sq: Square, occ: u64) u64
{
    var result: u64 = 0;
    const offset: u64 = sq.idx() * 64;

    inline for (orientations) |ori|
    {
        switch (ori)
        {
            .horizontal =>
            {
                const raw: u64 = index_of(ori, sq, occ);
                result |= ptr_rank_attacks[offset + raw];
            },
            .vertical =>
            {
                const raw: u64 = index_of(ori, sq, occ);
                result |= ptr_file_attacks[offset + raw];
            },
            .diagmain =>
            {
                const raw: u64 = index_of(ori, sq, occ);
                result |= ptr_diag_main_attacks[offset + raw];

            },
            .diaganti =>
            {
                const raw: u64 = index_of(ori, sq, occ);
                result |= ptr_diag_anti_attacks[offset + raw];
            },
        }
    }
    return result;
}

pub fn get_rank_attacks(sq: Square, occ: u64) u64
{
    return attacks_of(.horizontal, sq, occ);
}

pub fn get_file_attacks(sq: Square, occ: u64) u64
{
    return attacks_of(.vertical, sq, occ);
}

pub fn get_diagmain_attacks(sq: Square, occ: u64) u64
{
    return attacks_of(.diagmain, sq, occ);
}

pub fn get_diaganti_attacks(sq: Square, occ: u64) u64
{
    return attacks_of(.diaganti, sq, occ);
}

pub fn get_pawn_attacks(sq: Square, comptime us: Color) u64
{
    return switch(us.e)
    {
        .white => ptr_pawn_attacks_white[sq.u],
        .black => ptr_pawn_attacks_black[sq.u]
    };
}

pub fn get_knight_attacks(sq: Square) u64
{
    return ptr_knight_attacks[sq.u];
}

pub fn get_bishop_attacks(sq: Square, occ: u64) u64
{
    return combined_attacks_of(&.{ .diagmain, .diaganti }, sq, occ);
}

pub fn get_rook_attacks(sq: Square, occ: u64) u64
{
    return combined_attacks_of(&.{ .horizontal, .vertical }, sq, occ);
}

pub fn get_queen_attacks(sq: Square, occ: u64) u64
{
    return combined_attacks_of(&.{ .horizontal, .vertical, .diagmain, .diaganti }, sq, occ);
}

pub fn get_king_attacks(sq: Square) u64
{
    return ptr_king_attacks[sq.u];
}



