// zig fmt: off

//! Zobrist keys for position hashing.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");

const uses = struct
{
    const rnd = @import("rnd.zig");
};

const Piece = types.Piece;
const Square = types.Square;

pub fn initialize() void
{
    var random: uses.rnd.Random = .init(0);
    for (0..16) |pc| {
        for (0..64) |sq| {
            piece_square_keys[pc * 64 + sq] = random.next_u64();
        }
    }

    for (0..8) |ep| {
        ep_keys[ep] = random.next_u64();
    }

    for (0..16) |i| {
        //if (i & (i - 1) != 0)
        if (i == 0 or std.math.isPowerOfTwo(i)) {
            castling_keys[i] = 0;
            var j: usize = 1;
            while (j < 16) : (j <<= 1) {
                if (i & j != 0) {
                    castling_keys[i] ^= castling_keys[i];
                }
            }
        }
        else {
            castling_keys[i] = random.next_u64();
        }
    }

    btm_key = random.next_u64();
}

var piece_square_keys: [16 * 64]u64 = @splat(0);
var ep_keys: [8]u64 = @splat(0);
var castling_keys: [16]u64 = @splat(0);
var btm_key: u64 = 0;

const ptr_piece_square_keys: [*]const u64 = &piece_square_keys;
const ptr_ep_keys: [*]const u64 = &ep_keys;
const ptr_castling_keys: [*]const u64 = &castling_keys;

pub fn piece_square(pc: Piece, sq: Square) u64
{
    const idx: usize = pc.idx() * 64 + sq.idx();
    return ptr_piece_square_keys[idx];
}

pub fn enpassant(file: u3) u64
{
    return ptr_ep_keys[file];
}

pub fn castling(castling_rights: u4) u64
{
    return ptr_castling_keys[castling_rights];
}

pub fn btm() u64
{
    return btm_key;
}

