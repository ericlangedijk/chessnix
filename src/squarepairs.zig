// zig fmt: off

//! Information about each square pair.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;

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
    /// * orthoganal  = 100
    /// * diagonal    = 001
    /// * bishop      = 011 (value 3)
    /// * rook        = 100 (value 4)
    /// * queen       = 101 (value 5)
    ///
    mask: u3 = 0,
    /// The manhattan distance.
    distance: u3 = 0,
};

pub fn initialize() void
{
    // Set directions.
    for (Square.all) |from| {
        inline for (Direction.all) |dir| {
            const ray = from.ray(dir);
            for (ray.slice()) |to| {
                const pair: *SquarePair = get_mut(from, to);
                const ori = dir.to_orientation();
//                pair.distance = calc_distance(from, to);
                pair.direction = dir;
                pair.orientation = ori;
                if (ori == .diagmain or ori == .diaganti) pair.mask |= 0b001;
                if (ori == .horizontal or ori == .vertical) pair.mask |= 0b100;
            }
        }
    }

    // Set in-between bitboards
    for (Square.all) |from| {
        for (Square.all) |to| {
            const pair: *SquarePair = get_mut(from, to);
            pair.distance = calc_distance(from, to);

            if (pair.direction) |dir| {
                const ray = from.ray(dir);
                for (ray.slice()) |sq| {
                    if (sq.u == to.u) break;
                    pair.in_between_bitboard |= sq.to_bitboard();
                }
            }
        }
    }
}

fn calc_distance(a: Square, b: Square) u3
{
    const ar: i32 = a.rank();
    const br: i32 = b.rank();

    const af: i32 = a.file();
    const bf: i32 = b.file();

    const d: u32 = @max
    (
        @abs(ar - br),
        @abs(af - bf)
    );
    return @truncate(@abs(d));
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
