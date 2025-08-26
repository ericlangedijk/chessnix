// zig fmt: off

//! All kind ofo helper masks.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const data = @import("data.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;

const assert = std.debug.assert;

/// data.zig must be initialized first.
pub fn initialize() void
{
    // Passed pawn masks. TODO: just loop squares.
    for (bitboards.ranks) |rank|
    {
        for (bitboards.files) |file|
        {
            var bb: u64 = 0;
            const sq: Square = .from_rank_file(rank, file);
            const black_sq: Square = sq.relative(Color.BLACK);

            bb = data.ptr_bb_north[sq.u];
            if (file > bitboards.file_a) bb |= data.ptr_bb_north[sq.sub(1).u];
            if (file < bitboards.file_h) bb |= data.ptr_bb_north[sq.add(1).u];

            // Passed pawn mask.
            passed_pawn_masks_white[sq.u] = bb;
            passed_pawn_masks_black[black_sq.u] = funcs.mirror_vertically(bb);

            // Ep mask.
            if (rank == bitboards.rank_4 or rank == bitboards.rank_5)
            {
                if (sq.next(.west)) |n| ep_masks[sq.u] |= n.to_bitboard();
                if (sq.next(.east)) |n| ep_masks[sq.u] |= n.to_bitboard();
            }

            // Isolated pawn mask.
            if (file > bitboards.file_a) isolated_pawn_masks[sq.u] |= bitboards.file_bitboards[file - 1];
            if (file < bitboards.file_h) isolated_pawn_masks[sq.u] |= bitboards.file_bitboards[file + 1];
        }
    }
}

var isolated_pawn_masks: [64]u64 = @splat(0);
var passed_pawn_masks_white: [64]u64 = @splat(0);
var passed_pawn_masks_black: [64]u64 = @splat(0);
var ep_masks: [64]u64 =  @splat(0);

const ptr_passed_pawn_masks_white: [*]const u64 = &passed_pawn_masks_white;
const ptr_passed_pawn_masks_black: [*]const u64 = &passed_pawn_masks_black;
const ptr_ep_masks: [*]const u64 = &ep_masks;

pub fn get_isolated_pawn_mask(sq: Square) u64
{
    return isolated_pawn_masks[sq.u];
}

pub fn get_passed_pawn_mask(comptime us: Color, sq: Square) u64
{
    return switch (us.e)
    {
        .white => ptr_passed_pawn_masks_white[sq.u],
        .black => ptr_passed_pawn_masks_black[sq.u],
    };
}

/// `to` is the to-square of the  double pushed pawn (e4).
pub fn get_ep_mask(to: Square) u64
{
    return ptr_ep_masks[to.u];
}
