// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");

const PieceType = types.PieceType;

//    see: https://en.wikipedia.org/wiki/Fischer_random_chess_numbering_scheme
//    N = 0...959
//    a) Divide N by 4, yielding quotient N2 and remainder B1.
//       Place a Bishop upon the bright square corresponding to B1 (0=b, 1=d, 2=f, 3=h).
//    b) Divide N2 by 4 again, yielding quotient N3 and remainder B2.
//       Place a second Bishop upon the dark square corresponding to B2 (0=a, 1=c, 2=e, 3=g).
//    c) Divide N3 by 6, yielding quotient N4 and remainder Q.
//       Place the Queen according to Q, where 0 is the first free square starting from a, 1 is the second, etc.
//    d) N4 will be a single digit, 0...9.
//       Ignoring Bishops and Queen, find the positions of two Knights within the remaining five spaces.
//       Place the Knights according to its value by consulting the following N5N table:
// Position 518 is the classical position.

/// Returns a chess 960 backrow.
//pub fn decode(number: usize) [8]PieceType {
pub fn decode(number: usize) [8]u4 {

    // The permutations for the knight: the indexes of the remaining 5 empty squares after bishops and queen.
    const knight_table: [10][2]usize = .{
        .{0, 1}, // 0
        .{0, 2}, // 1
        .{0, 3}, // 2
        .{0, 4}, // 3
        .{1, 2}, // 4
        .{1, 3}, // 5
        .{1, 4}, // 6
        .{2, 3}, // 7
        .{2, 4}, // 8
        .{3, 4}, // 9
    };

    const nr: usize = number % 960;
    var backrow: [8]?PieceType = @splat(null);//PieceType.NO_PIECETYPE);
    var divider: usize = nr;

    // bishop 1
    var rem = div_rem(&divider, 4);
    backrow[rem * 2 + 1] = PieceType.BISHOP;

    // bishop 2
    rem = div_rem(&divider, 4);
    backrow[rem * 2] = PieceType.BISHOP;

    // queen
    rem = div_rem(&divider, 6);
    backrow[get_empty_index(backrow, rem)] = PieceType.QUEEN;

    // knight 1
    const knightindex1 = knight_table[divider][0];
    backrow[get_empty_index(backrow, knightindex1)] = PieceType.KNIGHT;

    // knight 2
    const knightindex2 = knight_table[divider][1] - 1;
    backrow[get_empty_index(backrow, knightindex2)] = PieceType.KNIGHT;

    // rook, king, rook
    backrow[get_empty_index(backrow, 0)] = PieceType.ROOK;
    backrow[get_empty_index(backrow, 0)] = PieceType.KING;
    backrow[get_empty_index(backrow, 0)] = PieceType.ROOK;

    // convert
    var result: [8]u4 = undefined;
    inline for (0..8) |i| {
        result[i] = backrow[i].?.u;
    }
    return result;
}

fn div_rem(n: *usize, div: usize) usize {
    const remainder = n.* % div;
    n.* /= div;
    return remainder;
}

fn get_empty_index(backrow: [8]?PieceType, empty_index: usize) usize {
    var count: usize = 0;
    inline for (0..8) |i| {
        if (backrow[i] == null) {
            if (count == empty_index) return i;
            count += 1;
        }
    }
    unreachable;
}

// pub const all_backrows: [960][8]u4 = compute_all_backrows();

// fn compute_all_backrows() [960][8]u4 {
//     @setEvalBranchQuota(64000);
//     var rows: [960][8]u4 = undefined;
//     for (0..960) |i| {
//         rows[i] = decode(i);
//     }
//     return rows;
// }