// zig fmt: off

//! Information about any combination of 2 squares.

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const Axis = types.Axis;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;


/// Information about a pair of squares.
/// - 'ray': the from-to ray bitboard **excluded** the from-square and **included** the to-square.
pub const SquarePair = struct {
    /// The from-to ray bitboard **excluded** the from-square and **included** the to-square.
    ray: u64 = 0,
    /// The from-to direction.
    direction: ?Direction = null,
    /// The from-to or to-from orientation.
    orientation: ?Orientation = null,
    /// Diagonal or orthogonal or nothing.
    axis: Axis = .no_axis,
    /// Distance between the squares.
    dist: u3 = 0,
    /// Manhattan distance between the squares.
    manh: u4 = 0,

    // TODO: 3 bytes left until size == 16. Add in the indexes for hce 3x4 and 3x7 areas into the hceterms arrays.

    const empty: SquarePair = .{};
};

pub const pairs: [Square.count * Square.count]SquarePair = compute_squarepairs();

fn compute_squarepairs() [Square.count * Square.count]SquarePair {
    @setEvalBranchQuota(128000);
    var sp: [Square.count * Square.count]SquarePair = @splat(.empty);

    // Set directions.
    for (Square.all) |from| {
        inline for (Direction.all) |dir| {
            const ray = from.ray(dir);
            for (ray.slice()) |to| {
                const idx: usize = from.idx() * 64 + to.idx();
                const ori = dir.to_orientation();
                sp[idx].direction = dir;
                sp[idx].orientation = ori;
                if (ori == .diagmain or ori == .diaganti) sp[idx].axis = .diag;
                if (ori == .horizontal or ori == .vertical) sp[idx].axis = .orth;
            }
        }
    }

    // Set ray bitboards
    for (Square.all) |from| {
        for (Square.all) |to| {
            const idx: usize = from.idx() * 64 + to.idx();
            // Distance
            sp[idx].dist = funcs.square_distance(from, to);
            sp[idx].manh = funcs.manhattan_distance(from, to);
            // Ray.
            if (sp[idx].direction) |dir| {
                const ray = from.ray(dir);
                for (ray.slice()) |sq| {
                    sp[idx].ray |= sq.to_bitboard();
                    if (sq.u == to.u) break;
                }
            }
            // #deprecated
            // In between.
            // sp[idx].in_between = sp[idx].ray & ~to.to_bitboard();
        }
    }
    return sp;
}

pub fn get(from: Square, to: Square) *const SquarePair {
    const idx: usize = from.idx() * 64 + to.idx();
    return &pairs[idx];
}

/// Convenience function.
pub fn dist(a: Square, b: Square) u3 {
    return get(a, b).dist;
}

/// Convenience function.
pub fn manh(a: Square, b: Square) u4 {
    return get(a, b).manh;
}


