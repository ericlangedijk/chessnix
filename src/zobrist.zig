// zig fmt: off

//! Zobrist keys for position hashing.

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");

const Piece = types.Piece;
const Square = types.Square;

/// piece-square + ep + castling + btm.
const rnd_count: usize = 768 + 64 + 16 + 1;

const offset_ep: usize = 768;
const offset_castling: usize = 768 + 64;
const offset_btm: usize = 768 + 64 + 16;

/// We use one flat array for all.
const all_randoms: [rnd_count]u64 = compute_all_randoms();

fn compute_all_randoms() [rnd_count]u64 {
    @setEvalBranchQuota(8000);
    var table: [rnd_count] u64 = undefined;
    var randoms: utils.Random = .init(1);
    for (0..rnd_count) |i| {
        table[i] = randoms.next();
    }
    // For safe xor of an invalid ep square (a1).
    table[offset_ep] = 0;
    return table;
}

pub fn piece_square(pc: Piece, sq: Square) u64 {
    const idx: usize = @as(usize, pc.u) * 64 + sq.u;
    return all_randoms[idx];
}

pub fn enpassant(sq: Square) u64 {
    const idx: usize = offset_ep + sq.u;
    return all_randoms[idx];
}

pub fn castling(castling_rights: u4) u64 {
    const idx: usize = offset_castling + castling_rights;
    return all_randoms[idx];
}

pub fn btm() u64 {
    const idx: usize = offset_btm;
    return all_randoms[idx];
}
