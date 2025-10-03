// zig fmt: off

//! All bitboards + magics + functions for move generation.
//! NOTE: initialize *before* bitboards.zig.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

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

    for (0..8) |b| {
        const one: u8 = 1;
        const bitpos: u3 = @truncate(b);
        for (0..256) |o| {
            const occ: u8 = @truncate(o);
            var attackmask: u8 = 0;

            // Scan bits forwards
            if (bitpos < 7) {
                var i: u3 = bitpos;
                while (true) {
                    i += 1;
                    const mask: u8 = one << i;
                    attackmask |= mask;
                    if (i == 7 or occ & mask != 0) break; // border or occupied
                }
            }
            // Scan bits backwards.
            if (bitpos > 0) {
                var i: u3 = bitpos;
                while (true) {
                    i -= 1;
                    const mask: u8 = one << i;
                    attackmask |= mask;
                    if (i == 0 or occ & mask != 0) break; // border or occupied
                }
            }
            // compress the index
            const index : u8 = (occ & occ_index_mask) >> 1;
            sliding_attacks[bitpos][index] = attackmask;
        }
    }

    for (Square.all) |sq| {
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

        const entry_file: *MagicEntry = &file_magics[sq.u];
        const entry_main: *MagicEntry = &main_magics[sq.u];
        const entry_anti: *MagicEntry = &anti_magics[sq.u];

        const bb_rank_1: u64 = 0x00000000000000ff;
        const bb_rank_8: u64 = 0xff00000000000000;
        const bb_file_a: u64 = 0x0101010101010101;
        const bb_file_h: u64 = 0x8080808080808080;
        const bb_border = bb_rank_1 | bb_rank_8 | bb_file_a | bb_file_h;

        // Masks without borders and without square itself.
        entry_file.mask = sq.rays_bitboard(&.{.north, .south}) & ~(bb_rank_1 | bb_rank_8);
        entry_main.mask = sq.rays_bitboard(&.{.north_west, .south_east}) & ~bb_border;
        entry_anti.mask = sq.rays_bitboard(&.{.north_east, .south_west}) & ~bb_border;

        // Magics for each square, deduced from the precalculated ones.
        entry_file.magic = PrecomputedMagics.file_magics[file];
        entry_main.magic = PrecomputedMagics.diag_main_magics[@as(u8, file) + rank];
        entry_anti.magic = PrecomputedMagics.diag_anti_magics[@as(u8, file) + (7 - rank)];

        // Rank attacks.
        for (0..64) |occ| {
            const attack: u8 = sliding_attacks[file][occ];
            rank_attacks[idx * 64 + occ] = @as(u64, attack) << (@as(u6, rank) * 8); // shift into the correct rank
        }

        // File attacks.
        for (0..64) |occ| {
            const attackmask: u8 = sliding_attacks[7 - rank][occ];
            var bitboard: u64 = 0;
            for (0..8) |i| {
                const bitpos: u3 = @truncate(i);
                if (funcs.test_bit_u8(attackmask, bitpos)) {
                    const square: Square = .from_rank_file(7 - bitpos, file);
                    bitboard |= square.to_bitboard();
                }
            }
            file_attacks[idx * 64 + occ] = bitboard;
        }

        // Diagonal main attacks.
        for (0..64) |occ| {
            const offset: u3 = @min(7 - rank, file);
            const attackmask: u8 = sliding_attacks[offset][occ];
            var bitboard: u64 = 0;
            // Scan northwest (backwards from offset).
            var q: Square = sq;
            var bitpos: u3 = offset;
            while (true) {
                if (funcs.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.north_west) orelse break;
                if (comptime lib.is_paranoid) assert(bitpos > 0);
                bitpos -= 1;
            }
            // Scan southeast (forwards from offset).
            q = sq;
            bitpos = offset;
            while(true) {
                if (funcs.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.south_east) orelse break;
                if (comptime lib.is_paranoid) assert(bitpos < 7);
                bitpos += 1;
            }
            diag_main_attacks[sq.idx() * 64 + occ] = bitboard;
        }

        // Diagonal anti attacks.
        for (0..64) |occ| {
            const offset: u3 = @min(rank, file);
            const attackmask: u8 = sliding_attacks[offset][occ];
            var bitboard: u64 = 0;
            // Scan southwest (backwards from offset).
            var q: Square = sq;
            var bitpos: u3 = offset;
            while (true) {
                if (funcs.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.south_west) orelse break;
                if (comptime lib.is_paranoid) assert(bitpos > 0);
                bitpos -= 1;
            }
            // Scan northeast (forwards from offset).
            q = sq;
            bitpos = offset;
            while (true) {
                if (funcs.test_bit_u8(attackmask, bitpos)) bitboard |= q.to_bitboard();
                q = q.next(.north_east) orelse break;
                if (comptime lib.is_paranoid) assert(bitpos < 7);
                bitpos += 1;
            }
            diag_anti_attacks[sq.idx() * 64 + occ] = bitboard;
        }
    }
}

/// This mask is used internally for compressed index. The borders are not needed in the story.
const occ_index_mask: u8 = 0b01111110;

/// Precomputed magics.
const PrecomputedMagics = struct {
    const file_magics: [8]u64 = .{
        0x8040201008040200,  // a-file
        0x4020100804020100,  // b-file
        0x2010080402010080,  // c-file
        0x1008040201008040,  // d-file
        0x0804020100804020,  // e-file
        0x0402010080402010,  // f-file
        0x0201008040201008,  // g-file
        0x0100804020100804   // h-file
    };

    const diag_main_magics: [15]u64 = .{
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

    const diag_anti_magics: [15]u64 = .{
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

const MagicEntry = struct {
    const empty: MagicEntry = .{};

    mask: u64 = 0,
    magic: u64 = 0,
    const shift: u6 = 57;

    fn attack_index(self: MagicEntry, occ: u64) u64 {
        return ((occ & self.mask) *% self.magic) >> shift;
    }
};

// Movgen.
var file_magics: [64]MagicEntry = @splat(.empty);
var main_magics: [64]MagicEntry = @splat(.empty);
var anti_magics: [64]MagicEntry = @splat(.empty);

var pawn_attacks_white: [64]u64 = @splat(0);
var pawn_attacks_black: [64]u64 = @splat(0);
var knight_attacks: [64]u64 =  @splat(0);
var king_attacks: [64]u64 = @splat(0);
var rank_attacks: [64 * 64]u64 = @splat(0);
var file_attacks: [64 * 64]u64 = @splat(0);
var diag_main_attacks: [64 * 64]u64 = @splat(0);
var diag_anti_attacks: [64 * 64]u64 = @splat(0);

// Movgen. (Pointers are faster than array access. Danger of @memcpy calls).
pub const ptr_file_magics: [*]const MagicEntry = &file_magics;
pub const ptr_main_magics: [*]const MagicEntry = &main_magics;
pub const ptr_anti_magics: [*]const MagicEntry = &anti_magics;

pub const ptr_pawn_attacks_white: [*]const u64 = &pawn_attacks_white;
pub const ptr_pawn_attacks_black: [*]const u64 = &pawn_attacks_black;
pub const ptr_knight_attacks: [*]const u64 = & knight_attacks;
pub const ptr_king_attacks: [*]const u64 = &king_attacks;
pub const ptr_rank_attacks: [*]const u64 = &rank_attacks;
pub const ptr_file_attacks: [*]const u64 = &file_attacks;
pub const ptr_diag_main_attacks: [*]const u64 = &diag_main_attacks;
pub const ptr_diag_anti_attacks: [*]const u64 = &diag_anti_attacks;

fn attack_index_of(comptime ori: Orientation, sq: Square, occ: u64) u64 {
    return switch(ori) {
        .horizontal => ((occ >> (sq.u & 0b111000)) & occ_index_mask) >> 1,
        .vertical   => ptr_file_magics[sq.u].attack_index(occ),
        .diagmain   => ptr_main_magics[sq.u].attack_index(occ),
        .diaganti   => ptr_anti_magics[sq.u].attack_index(occ),
    };
}

/// Returns attacks for one direction.
pub fn attacks_of(comptime ori: Orientation, sq: Square, occ: u64) u64 {
    const offset: u64 = sq.idx() * 64;
    const raw: u64 = attack_index_of(ori, sq, occ);

    return switch (ori) {
        .horizontal => ptr_rank_attacks[offset + raw],
        .vertical   => ptr_file_attacks[offset + raw],
        .diagmain   => ptr_diag_main_attacks[offset + raw],
        .diaganti   => ptr_diag_anti_attacks[offset + raw],
    };
}

/// A little speedup for combined attacks. We only need to calculate the offset once.
fn combined_attacks_of(comptime orientations: []const Orientation, sq: Square, occ: u64) u64 {
    const offset: u64 = sq.idx() * 64;
    var result: u64 = 0;

    inline for (orientations) |ori| {
        switch (ori) {
            .horizontal => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= ptr_rank_attacks[offset + raw];
            },
            .vertical => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= ptr_file_attacks[offset + raw];
            },
            .diagmain => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= ptr_diag_main_attacks[offset + raw];
            },
            .diaganti => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= ptr_diag_anti_attacks[offset + raw];
            },
        }
    }
    return result;
}

pub fn get_rank_attacks(sq: Square, occ: u64) u64 {
    return attacks_of(.horizontal, sq, occ);
}

pub fn get_file_attacks(sq: Square, occ: u64) u64 {
    return attacks_of(.vertical, sq, occ);
}

pub fn get_diagmain_attacks(sq: Square, occ: u64) u64 {
    return attacks_of(.diagmain, sq, occ);
}

pub fn get_diaganti_attacks(sq: Square, occ: u64) u64 {
    return attacks_of(.diaganti, sq, occ);
}

pub fn get_pawn_attacks(sq: Square, comptime us: Color) u64 {
    return switch(us.e) {
        .white => ptr_pawn_attacks_white[sq.u],
        .black => ptr_pawn_attacks_black[sq.u]
    };
}

pub fn get_knight_attacks(sq: Square) u64 {
    return ptr_knight_attacks[sq.u];
}

pub fn get_bishop_attacks(sq: Square, occ: u64) u64 {
    return combined_attacks_of(&.{ .diagmain, .diaganti }, sq, occ);
}

pub fn get_rook_attacks(sq: Square, occ: u64) u64 {
    return combined_attacks_of(&.{ .horizontal, .vertical }, sq, occ);
}

pub fn get_queen_attacks(sq: Square, occ: u64) u64 {
    return combined_attacks_of(&.{ .horizontal, .vertical, .diagmain, .diaganti }, sq, occ);
}

pub fn get_king_attacks(sq: Square) u64 {
    return ptr_king_attacks[sq.u];
}
