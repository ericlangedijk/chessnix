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

/// We use one flat array for all.
const all_randoms: [rnd_count]u64 = compute_all_randoms();

const keys: Keys = compute_keys();

const Keys = struct {
    piece_square: [12 * 64]u64,
    enpassant: [64]u64,
    castling: [16]u64,
    btm: u64,
};

fn compute_keys() Keys {
    @setEvalBranchQuota(8000);
    var randoms: utils.Random = .init(1);
    var result: Keys = undefined;
    const flattened: *[rnd_count]u64 = @bitCast(keys);
    for (0..rnd_count) |i| {
        flattened[i] = randoms.next_u64();
    }
    // For safe xor of an invalid ep square (a1).
    result.enpassant[0] = 0;
    return result;
}

fn compute_all_randoms() [rnd_count]u64 {
    @setEvalBranchQuota(8000);
    var table: [rnd_count] u64 = undefined;
    var randoms: utils.Random = .init(1);
    for (0..rnd_count) |i| {
        table[i] = randoms.next_u64();
    }
    // For safe xor of an invalid ep square (a1).
    table[768] = 0;
    return table;
}

pub fn piece_square(pc: Piece, sq: Square) u64 {
    const idx: usize = pc.idx() * 64 + sq.u;
    return all_randoms[idx];
}

pub fn enpassant(sq: Square) u64 {
    const idx: usize = @as(usize, 768) + sq.u;
    return all_randoms[idx];
}

pub fn castling(castling_rights: u4) u64 {
    const idx: usize = @as(usize, 768) + 64 + castling_rights;
    return all_randoms[idx];
}

pub fn btm() u64 {
    const idx: usize = @as(usize, 768) + 64 + 16;
    return all_randoms[idx];
}
