//! Information about each square pair.

const std = @import("std");
const lib = @import("lib.zig");
const console = @import("console.zig");
const funcs = @import("funcs.zig");
const bits = @import("bits.zig");
const position = @import("position.zig");

const uses = struct
{
    const rnd = @import("rnd.zig");
};

const Orientation = position.Orientation;
const Direction = position.Direction;
const Color = position.Color;
const Piece = position.Piece;
const Square = position.Square;

var pairs: [64 * 64]SquarePair = @splat(SquarePair.empty);
pub const ptr_pairs: [*]SquarePair = &pairs;

/// Information about a pair of squares.
/// * Mainly used for determining pinners and pinned pieces.
pub const SquarePair = struct
{
    const empty: SquarePair = .{};

    /// The bitboard of the squares in between two squares.
    in_between_bitboard: u64 = 0,
    /// The from-to direction.
    direction: ?Direction = null,
    /// The from-to or to-from orientation.
    orientation: ?Orientation = null,
    /// A mask for quick checking if a piece can cover this direction.\
    /// We can `and` the piecetype's numeric value with this mask.
    ///
    /// * diagonal    = 001
    /// * orthoganal  = 100
    /// * both        = 101
    /// * bishop      = 011 (value 3)
    /// * rook        = 100 (value 4)
    /// * queen       = 101 (value 5)
    ///
    mask: u3 = 0,
};

pub fn initialize() void
{
    // Set directions.
    for (Square.all) |from|
    {
        inline for (Direction.all) |dir|
        {
            const ray = from.ray(dir);
            for (ray.slice()) |to|
            {
                const pair: *SquarePair = get_mut(from, to);
                const ori = dir.to_orientation();
                pair.direction = dir;
                pair.orientation = ori;
                if (ori == .diagmain or ori == .diaganti) pair.mask |= 0b001;
                if (ori == .horizontal or ori == .vertical) pair.mask |= 0b100;
            }
        }
    }

    // Set in-between bitboards
    for (Square.all) |from|
    {
        for (Square.all) |to|
        {
            const pair: *SquarePair = get_mut(from, to);
            if (pair.direction) |dir|
            {
                const ray = from.ray(dir);
                for (ray.slice()) |sq|
                {
                    if (sq.u == to.u) break;
                    pair.in_between_bitboard |= sq.to_bitboard();
                }
            }
        }
    }
}

fn get_mut(from: Square, to: Square) *SquarePair
{
    const idx: usize = from.idx() * 64 + to.idx();
    return &ptr_pairs[idx];
}

pub fn get(from: Square, to: Square) *const SquarePair
{
    const idx: usize = from.idx() * 64 + to.idx();
    return &ptr_pairs[idx];
}

pub fn in_between_bitboard(from: Square, to: Square) u64
{
    return get(from, to).in_between_bitboard;
}
