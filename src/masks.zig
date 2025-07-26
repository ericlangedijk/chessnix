//! All kind ofo helper masks.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const data = @import("data.zig");
const console = @import("console.zig");
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
            const white_sq: Square = .from_rank_file(rank, file);
            const black_sq: Square = white_sq.relative(Color.BLACK);
            bb = data.ptr_bb_north[white_sq.u];
            if (white_sq.file() > bitboards.file_a) bb |= data.ptr_bb_north[white_sq.sub(1).u];
            if (white_sq.file() < bitboards.file_h) bb |= data.ptr_bb_north[white_sq.add(1).u];
            passed_pawn_masks_white[black_sq.u] = bb;
            passed_pawn_masks_black[black_sq.u] = funcs.mirror_vertically(bb);
            if (rank == bitboards.rank_4 or rank == bitboards.rank_5)
            {
                if (white_sq.next(.west)) |n| ep_masks[white_sq.u] |= n.to_bitboard();
                if (white_sq.next(.east)) |n| ep_masks[white_sq.u] |= n.to_bitboard();
            }
        }
    }
}

var passed_pawn_masks_white: [64]u64 = @splat(0);
var passed_pawn_masks_black: [64]u64 = @splat(0);
var ep_masks: [64]u64 =  @splat(0); // TODO: not initialized / used.

const ptr_passed_pawn_masks_white: [*]u64 = &passed_pawn_masks_white;
const ptr_passed_pawn_masks_black: [*]u64 = &passed_pawn_masks_black;
const ptr_ep_masks: [*]u64 = &ep_masks;

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
