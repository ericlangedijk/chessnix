// zig fmt: off

//! Zobrist keys for position hashing.

const std = @import("std");
const lib = @import("lib.zig");
const rnd = @import("rnd.zig");
const types = @import("types.zig");

const assert = std.debug.assert;

const Piece = types.Piece;
const Square = types.Square;


pub fn initialize() void {
    var random: rnd.Random = .init(1);

    // First
    for (Piece.all) |pc| {
        for (Square.all) |sq| {
            piece_square_keys[pc.idx() * 64 + sq.idx()] = random.next_u64();
        }
    }

    // Second
    for (Square.all) |sq| {
        ep_keys[sq.u] = random.next_u64();
    }

    // For safe XOR invalid ep square.
    ep_keys[0] = 0;

    for (0..16) |i| {
        castling_keys[i] = random.next_u64();
    }

    btm_key = random.next_u64();
}



pub var piece_square_keys: [12 * 64]u64 = @splat(0);
pub var ep_keys: [64]u64 = @splat(0);
pub var castling_keys: [16]u64 = @splat(0);
pub var btm_key: u64 = 0;

const ptr_piece_square_keys: [*]const u64 = &piece_square_keys;
const ptr_ep_keys: [*]const u64 = &ep_keys;
pub const ptr_castling_keys: [*]const u64 = &castling_keys;

/// Don't call for no_piece.
pub fn piece_square(pc: Piece, sq: Square) u64
{
    if (comptime lib.is_paranoid) {
        assert(pc.e != .no_piece);
    }
    const idx: usize = pc.idx() * 64 + sq.u;
    return ptr_piece_square_keys[idx];
}

pub fn enpassant(sq: Square) u64
{
    return ptr_ep_keys[sq.u];
}

pub fn castling(castling_rights: u4) u64 {
    return ptr_castling_keys[castling_rights];
}

pub fn btm() u64 {
    return btm_key;
}

// // for (z.piece_square_keys, 0..) |k, i| {
// //     lib.io.debugprint("0x{x:0>16},\n", .{k});
// //     std.debug.assert(k == z.pc_sq[i]);
// // }

// // for (z.ep_keys, 0..) |k, i| {
// //     lib.io.debugprint("0x{x:0>16},\n", .{k});
// //     // _ = i;
// //     std.debug.assert(k == z.ee[i]);
// // }

// // for (z.castling_keys, 0..) |k, i| {
// //     lib.io.debugprint("0x{x:0>16},\n", .{k});
// //      _ = i;
// //     //std.debug.assert(k == z.ee[i]);
// // }


// // lib.io.debugprint("0x{x:0>16}  0x{x:0>16},\n", .{z.btm(), z.black_tm});
// // for (p.Layouts.all, 0..) |backrow, i| {
// //     lib.io.debugprint("{} {any}\n", .{i, backrow});
// // }
