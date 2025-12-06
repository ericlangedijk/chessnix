// zig fmt: off

//! All bitboards + magics + functions for move generation.

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

const sliding_attacks: [8][64]u8 = compute_sliding_attackmasks();
const file_magics: [64]MagicEntry = compute_file_magics();
const main_magics: [64]MagicEntry = compute_diagmain_magics();
const anti_magics: [64]MagicEntry = compute_diaganti_magics();
const pawn_attacks_white: [64]u64 = compute_pawn_hits_white();
const pawn_attacks_black: [64]u64 = compute_pawn_hits_black();
const pawn_attack_white_and_black_combined: [64]u64 = compute_pawn_hits_white_and_black_combined();
const knight_attacks: [64]u64 = compute_knight_attacks();
const king_attacks: [64]u64 = compute_king_attacks();
const rank_attacks: [64 * 64]u64 = compute_rank_attacks();
const file_attacks: [64 * 64]u64 = compute_file_attacks();
const diag_main_attacks: [64 * 64]u64 = compute_diagmain_attacks();
const diag_anti_attacks: [64 * 64]u64 = compute_diaganti_attacks();

/// This mask is used internally for compressed index. The borders are not needed in the story.
const occ_index_mask: u8 = 0b01111110;

fn compute_sliding_attackmasks() [8][64]u8 {
    @setEvalBranchQuota(8000);
    // Determine byte-based local sliding attack masks, used for the big sliding attack tables.
    // Determine for each bit-position the 8-bit attackmask for each occuption. The from-bitpos is never included.
    // We use a compressed index becuase borders are excluded.
    var sa: [8][64]u8 = std.mem.zeroes([8][64]u8);

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
            sa[bitpos][index] = attackmask;
        }
    }
    return sa;
}

fn compute_file_magics() [64]MagicEntry {
    @setEvalBranchQuota(8000);
    const bb = @import("bitboards.zig");
    var fm: [64]MagicEntry = @splat(.{ .mask = 0, .magic = 0 });
    for (Square.all) |sq| {
        fm[sq.u].mask = sq.rays_bitboard(&.{.north, .south}) & ~(bb.bb_rank_1 | bb.bb_rank_8);
        fm[sq.u].magic = PrecomputedMagics.file_magics[sq.coord.file];
    }
    return fm;
}

fn compute_diagmain_magics() [64]MagicEntry {
    @setEvalBranchQuota(8000);
    const bb = @import("bitboards.zig");
    var dm: [64]MagicEntry = @splat(.{ .mask = 0, .magic = 0 });
    for (Square.all) |sq| {
        const file: u3 = sq.file();
        const rank: u3 = sq.rank();
        dm[sq.u].mask = sq.rays_bitboard(&.{.north_west, .south_east}) & ~bb.bb_border;
        dm[sq.u].magic = PrecomputedMagics.diag_main_magics[@as(u8, file) + rank];
    }
    return dm;
}

fn compute_diaganti_magics() [64]MagicEntry {
    @setEvalBranchQuota(8000);
    const bb = @import("bitboards.zig");
    var am: [64]MagicEntry = @splat(.{ .mask = 0, .magic = 0 });
    inline for (Square.all) |sq| {
        const file: u3 = sq.file();
        const rank: u3 = sq.rank();
        am[sq.u].mask = sq.rays_bitboard(&.{.north_east, .south_west}) & ~bb.bb_border;
        am[sq.u].magic = PrecomputedMagics.diag_anti_magics[@as(u8, file) + (7 - rank)];
    }
    return am;
}

fn compute_pawn_hits_white() [64]u64 {
    @setEvalBranchQuota(8000);
    var phw: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        // Pawn hits. Fake pawnhits are calculated for the first and last rank (for pawn tricks).
        if (sq.next(.north_east))|n| phw[sq.u] |= n.to_bitboard();
        if (sq.next(.north_west))|n| phw[sq.u] |= n.to_bitboard();
    }
    return phw;
}

fn compute_pawn_hits_black() [64]u64 {
    @setEvalBranchQuota(8000);
    var phb: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        // Pawn hits. Fake pawnhits are calculated for the first and last rank (for pawn tricks).
        if (sq.next(.south_east))|n| phb[sq.u] |= n.to_bitboard();
        if (sq.next(.south_west))|n| phb[sq.u] |= n.to_bitboard();
    }
    return phb;
}

fn compute_pawn_hits_white_and_black_combined() [64]u64 {
    var pac: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        pac[sq.u] = pawn_attacks_white[sq.u] | pawn_attacks_black[sq.u];
    }
    return pac;
}

fn compute_knight_attacks() [64]u64 {
    @setEvalBranchQuota(8000);
    var na: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        if (sq.next_twice(.north_west, .west))  |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_west, .north)) |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_east, .north)) |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.north_east, .east))  |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_east, .east))  |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_east, .south)) |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_west, .south)) |n| na[sq.u] |= n.to_bitboard();
        if (sq.next_twice(.south_west, .west))  |n| na[sq.u] |= n.to_bitboard();
    }
    return na;
}

fn compute_king_attacks() [64]u64 {
    @setEvalBranchQuota(8000);
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
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

fn compute_rank_attacks() [64 * 64]u64 {
    @setEvalBranchQuota(8000);
    var ra: [64 * 64]u64 = @splat(0);
    for (Square.all) |sq| {
        const idx: u64 = sq.u;
        for (0..64) |occ| {
            const attack: u8 = sliding_attacks[sq.coord.file][occ];
            ra[idx * 64 + occ] = @as(u64, attack) << (@as(u6, sq.coord.rank) * 8); // shift into the correct rank
        }
    }
    return ra;
}

fn compute_file_attacks() [64 * 64]u64 {
    @setEvalBranchQuota(132000);
    var fa: [64 * 64]u64 = @splat(0);
    for (Square.all) |sq| {
        const idx: u64 = sq.u;
        for (0..64) |occ| {
            const attackmask: u8 = sliding_attacks[7 - sq.coord.rank][occ];
            var bitboard: u64 = 0;
            for (0..8) |i| {
                const bitpos: u3 = @truncate(i);
                if (funcs.test_bit_u8(attackmask, bitpos)) {
                    const square: Square = .from_rank_file(7 - bitpos, sq.coord.file);
                    bitboard |= square.to_bitboard();
                }
            }
            fa[idx * 64 + occ] = bitboard;
        }
    }
    return fa;
}

fn compute_diagmain_attacks()[64 * 64]u64 {
    @setEvalBranchQuota(264000);
    var dma: [64 * 64]u64 = @splat(0);
    for (Square.all) |sq| {
        const idx: u64 = sq.u;
        for (0..64) |occ| {
            const offset: u3 = @min(7 - sq.coord.rank, sq.coord.file);
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
            dma[idx * 64 + occ] = bitboard;
        }
    }
    return dma;
}

fn compute_diaganti_attacks()[64 * 64]u64 {
    @setEvalBranchQuota(264000);
    var daa: [64 * 64]u64 = @splat(0);
    for (Square.all) |sq| {
        const idx: u64 = sq.u;
        for (0..64) |occ| {
            const offset: u3 = @min(sq.coord.rank, sq.coord.file);
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
            daa[idx * 64 + occ] = bitboard;
        }
    }
    return daa;
}

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

fn attack_index_of(comptime ori: Orientation, sq: Square, occ: u64) u64 {
    return switch(ori) {
        .horizontal => ((occ >> (sq.u & 0b111000)) & occ_index_mask) >> 1,
        .vertical   => file_magics[sq.u].attack_index(occ),
        .diagmain   => main_magics[sq.u].attack_index(occ),
        .diaganti   => anti_magics[sq.u].attack_index(occ),
    };
}

/// Returns attacks for one direction.
fn attacks_of(comptime ori: Orientation, sq: Square, occ: u64) u64 {
    const offset: u64 = sq.idx() * 64;
    const raw: u64 = attack_index_of(ori, sq, occ);

    return switch (ori) {
        .horizontal => rank_attacks[offset + raw],
        .vertical   => file_attacks[offset + raw],
        .diagmain   => diag_main_attacks[offset + raw],
        .diaganti   => diag_anti_attacks[offset + raw],
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
                result |= rank_attacks[offset + raw];
            },
            .vertical => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= file_attacks[offset + raw];
            },
            .diagmain => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= diag_main_attacks[offset + raw];
            },
            .diaganti => {
                const raw: u64 = attack_index_of(ori, sq, occ);
                result |= diag_anti_attacks[offset + raw];
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
        .white => pawn_attacks_white[sq.u],
        .black => pawn_attacks_black[sq.u]
    };
}

pub fn get_pawn_attacks_combined(sq: Square) u64 {
    return pawn_attack_white_and_black_combined[sq.u];
}

pub fn get_knight_attacks(sq: Square) u64 {
    return knight_attacks[sq.u];
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
    return king_attacks[sq.u];
}

pub fn get_piece_attacks(sq: Square, occ: u64, comptime pc: PieceType, comptime us: Color) u64 {
    return switch (pc.e) {
        .pawn => get_pawn_attacks(sq, us),
        .knight => get_knight_attacks(sq),
        .bishop => get_bishop_attacks(sq, occ),
        .rook => get_rook_attacks(sq, occ),
        .queen => get_queen_attacks(sq, occ),
        .king => get_king_attacks(sq),
        else => unreachable,
    };
}
