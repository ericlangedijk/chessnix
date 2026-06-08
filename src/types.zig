// zig fmt: off

//! Basic types and consts, used almost everywhere.

const std = @import("std");
const utils = @import("utils.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const funcs = @import("funcs.zig");
const lib = @import("lib.zig");

const assert = std.debug.assert;
const io = lib.io;

pub const Axis = enum(u2) {
    none, orth, diag,
};

pub const Orientation = enum(u2) {
    horizontal, vertical, diagmain, diaganti,
};

pub const Direction = enum(u3) {
    north, east, south, west,
    north_west, north_east, south_east, south_west,

    pub const all: [8]Direction = .{ .north, .east, .south, .west, .north_west, .north_east, .south_east, .south_west };

    pub fn relative(self: Direction, comptime color: Color) Direction {
        if (color.e == .white) {
            return self;
        }
        return switch(self) {
            .north => .south,
            .east => .west,
            .south => .north,
            .west => .east,
            .north_west => .south_west,
            .north_east => .south_east,
            .south_east => .north_east,
            .south_west => .north_west,
        };
    }

    pub fn to_orientation(self: Direction) Orientation {
        return switch(self) {
            .north, .south => .vertical,
            .east, .west => .horizontal,
            .north_west, .south_east => .diagmain,
            .north_east, .south_west => .diaganti,
        };
    }
};

pub const Color = packed union {
    pub const count: usize = 2;

    pub const E = enum(u1) { white, black };

    /// The enum value
    e: E,
    /// The numeric value
    u: u1,

    pub const white: Color = .{ .e = .white };
    pub const black: Color = .{ .e = .black };

    pub const all: [Color.count]Color = .{ white, black };

    pub fn opp(self: Color) Color {
        return .{ .u  = self.u ^ 1 };
    }

    pub fn idx(self: Color) usize {
        return self.u;
    }
};

pub const PieceType = packed union {
    pub const count: usize = 6;

    /// Although 3 bits are enough 4 bits is easier for conversions.
    pub const E = enum(u4) {
        pawn = 0,
        knight = 1,
        bishop = 2,
        rook = 3,
        queen = 4,
        king = 5,
    };
    /// The enum value.
    e: E,
    /// The numeric value.
    u: u4,

    pub const all: [PieceType.count]PieceType = .{ pawn, knight, bishop, rook, queen, king };

    pub const pawn: PieceType = .{ .e = .pawn };
    pub const knight: PieceType = .{ .e = .knight };
    pub const bishop: PieceType = .{ .e = .bishop };
    pub const rook: PieceType = .{ .e = .rook };
    pub const queen: PieceType = .{ .e = .queen };
    pub const king: PieceType = .{ .e = .king };

    pub fn idx(self: PieceType) usize {
        return self.u;
    }

    pub fn value(self: PieceType) i32 {
        return piece_values[self.u];
    }

    pub fn simple_value(self: PieceType) i32 {
        return simple_piece_values[self.u];
    }

    // pub fn to_char(self: PieceType) u8 {
    //     return switch(self.e) {
    //         .pawn => 0,
    //         .knight => 'N',
    //         .bishop => 'B',
    //         .rook => 'R',
    //         .queen => 'Q',
    //         .king => 'K',
    //     };
    // }
};

pub const Piece = packed union {
    pub const count: usize = 12;
    pub const count_included_empty: usize = 13;

    /// To have convenient array indexing the values are just sequential.
    pub const E = enum(u4) {
        w_pawn   = 0,  // 0000
        w_knight = 1,  // 0001
        w_bishop = 2,  // 0010
        w_rook   = 3,  // 0011
        w_queen  = 4,  // 0100
        w_king   = 5,  // 0101
        b_pawn   = 6,  // 0110
        b_knight = 7,  // 0111
        b_bishop = 8,  // 1000
        b_rook   = 9,  // 1001
        b_queen  = 10, // 1010
        b_king   = 11, // 1011

        no_piece = 12, // 1010
    };

    /// The enum value.
    e: E,
    /// The numeric value.
    u: u4,

    pub const w_pawn   : Piece = .{ .e = .w_pawn };
    pub const w_knight : Piece = .{ .e = .w_knight };
    pub const w_bishop : Piece = .{ .e = .w_bishop };
    pub const w_rook   : Piece = .{ .e = .w_rook };
    pub const w_queen  : Piece = .{ .e = .w_queen };
    pub const w_king   : Piece = .{ .e = .w_king };

    pub const b_pawn   : Piece = .{ .e = .b_pawn };
    pub const b_knight : Piece = .{ .e = .b_knight };
    pub const b_bishop : Piece = .{ .e = .b_bishop };
    pub const b_rook   : Piece = .{ .e = .b_rook };
    pub const b_queen  : Piece = .{ .e = .b_queen };
    pub const b_king   : Piece = .{ .e = .b_king };

    pub const no_piece : Piece = .{ .e = .no_piece };

    /// All valid pieces.
    pub const all: [Piece.count]Piece = .{
        w_pawn, w_knight, w_bishop, w_rook, w_queen, w_king,
        b_pawn, b_knight, b_bishop, b_rook, b_queen, b_king,
    };

    pub fn init(pt: PieceType, side: Color) Piece {
        return if (side.e == .white) .{ .u = pt.u } else .{ .u = pt.u + 6 };
    }

    pub fn is_empty(self: Piece) bool {
        return self.e == .no_piece;
    }

    pub fn is_piece(self: Piece) bool {
        return self.e != .no_piece;
    }

    /// Don't call for no_piece.
    pub fn color(self: Piece) Color {
        if (comptime lib.is_paranoid) {
            assert(self.e != .no_piece);
        }
        return if (self.u < 6) Color.white else Color.black;
    }

    pub fn is_white(self: Piece) bool {
        return self.u < 6;
    }

    pub fn is_color(self: Piece, comptime us: Color) bool {
        return if (us.e == .white) self.u < 6 else self.u >= 6 and self.u <= 11;
    }

    /// Don't call for no_piece.
    pub fn piecetype(self: Piece) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.e != .no_piece);
        }
        return if (self.u < 6) .{ .u = self.u } else .{.u = self. u - 6 };
    }

    pub fn piecetype_for_known_color(self: Piece, comptime us: Color) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.e != .no_piece and self.color().e == us.e);
        }
        return if (us.e == .white) .{ .u = self.u } else .{.u = self. u - 6 };
    }

    /// Used for flipping the board.
    pub fn opp(self: Piece) Piece {
        if (self.is_empty()) {
            return Piece.no_piece;
        }
        return if (self.color().e == .white ) .{ .u = self.u + 6 } else .{ .u = self.u - 6 };
    }

    pub fn is_pawn(self: Piece) bool {
        return self.piecetype().e == .pawn;
    }

    pub fn is_rook(self: Piece) bool {
        return self.piecetype().e == .rook;
    }

    pub fn is_minor(self: Piece) bool {
        const pt: PieceType.E = self.piecetype().e;
        return pt == .knight or pt == .bishop;
    }

    pub fn is_major(self: Piece) bool {
        const pt: PieceType.E = self.piecetype().e;
        return pt == .rook or pt == .queen;
    }

    pub fn is_king(self: Piece) bool {
        return self.piecetype().e == .king;
    }

    pub fn is_king_of_color(self: Piece, comptime us: Color) bool {
        return switch (us.e) {
            .white => self.e == .w_king,
            .black => self.e == .b_king,
        };
    }

    pub fn is_pawn_of_color(self: Piece, comptime us: Color) bool {
        return if (us.e == .white) self.e == .w_pawn else self.e == .b_pawn;
    }

    pub fn is_rook_of_color(self: Piece, comptime us: Color) bool {
        return if (us.e == .white) self.e == .w_rook else self.e == .b_rook;
    }

    pub fn is_minor_of_color(self: Piece, comptime us: Color) bool {
        return switch (us.e) {
            .white => self.e == .w_knight or self.e == .w_bishop,
            .black => self.e == .b_knight or self.e == .b_bishop,
        };
    }

    pub fn is_major_of_color(self: Piece, comptime us: Color) bool {
        return switch (us.e) {
            .white => self.e == .w_rook or self.e == .w_queen,
            .black => self.e == .b_rook or self.e == .b_queen,
        };
    }

    /// Returns the static exchange evaluation value. It is allowed to call this for no-piece.
    pub fn value(self: Piece) i32 {
        return piece_values[self.u];
    }

    pub fn simple_value(self: Piece) i32 {
        return simple_piece_values[self.u];
    }

    pub fn to_print_char(self: Piece) u8 {
        var ch: u8 = switch(self.piecetype().e) {
            .pawn => 'P',
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
        };
        if (self.color().e == .black) ch = std.ascii.toLower(ch);
        return ch;
    }

    pub fn to_char(self: Piece) u8 {
        return "PNBRQKpnbrqk?"[self.u];
    }

    pub fn from_char(char: u8) ParsingError!Piece {
        return switch(char) {
            'P' => w_pawn,
            'N' => w_knight,
            'B' => w_bishop,
            'R' => w_rook,
            'Q' => w_queen,
            'K' => w_king,
            'p' => b_pawn,
            'n' => b_knight,
            'b' => b_bishop,
            'r' => b_rook,
            'q' => b_queen,
            'k' => b_king,
            else => ParsingError.InvalidFenPiece,
        };
    }
};

pub const CastleType = packed union {
    pub const count: usize = 2;

    pub const E = enum(u1) { short, long };

    /// The enum value
    e: E,
    /// The numeric value
    u: u1,

    pub const short: CastleType = .{ .e = .short };
    pub const long: CastleType = .{ .e = .long };

    pub const all: [CastleType.count]CastleType = .{ short, long };
};

pub const File = u3;
pub const Rank = u3;

pub const Coord = packed struct {
    /// file == x. raw bits: Square.u & 7 (0b000111)
    file: u3,
    /// rank == y. raw bits: Square.u >> 3  (0b111000)
    rank: u3,
};

pub const Square = packed union {
    pub const count: usize = 64;

    pub const E = enum(u6) {
        a1, b1, c1, d1, e1, f1, g1, h1,
        a2, b2, c2, d2, e2, f2, g2, h2,
        a3, b3, c3, d3, e3, f3, g3, h3,
        a4, b4, c4, d4, e4, f4, g4, h4,
        a5, b5, c5, d5, e5, f5, g5, h5,
        a6, b6, c6, d6, e6, f6, g6, h6,
        a7, b7, c7, d7, e7, f7, g7, h7,
        a8, b8, c8, d8, e8, f8, g8, h8,
    };
    /// The enum value.
    e: E,
    /// The numeric value
    u: u6,
    /// The file and rank bits match nicely.
    coord: Coord,

    pub const zero: Square = a1;

    pub const a1: Square = .{ .u = 0 };
    pub const b1: Square = .{ .u = 1 };
    pub const c1: Square = .{ .u = 2 };
    pub const d1: Square = .{ .u = 3 };
    pub const e1: Square = .{ .u = 4 };
    pub const f1: Square = .{ .u = 5 };
    pub const g1: Square = .{ .u = 6 };
    pub const h1: Square = .{ .u = 7 };
    pub const a2: Square = .{ .u = 8 };
    pub const b2: Square = .{ .u = 9 };
    pub const c2: Square = .{ .u = 10 };
    pub const d2: Square = .{ .u = 11 };
    pub const e2: Square = .{ .u = 12 };
    pub const f2: Square = .{ .u = 13 };
    pub const g2: Square = .{ .u = 14 };
    pub const h2: Square = .{ .u = 15 };
    pub const a3: Square = .{ .u = 16 };
    pub const b3: Square = .{ .u = 17 };
    pub const c3: Square = .{ .u = 18 };
    pub const d3: Square = .{ .u = 19 };
    pub const e3: Square = .{ .u = 20 };
    pub const f3: Square = .{ .u = 21 };
    pub const g3: Square = .{ .u = 22 };
    pub const h3: Square = .{ .u = 23 };
    pub const a4: Square = .{ .u = 24 };
    pub const b4: Square = .{ .u = 25 };
    pub const c4: Square = .{ .u = 26 };
    pub const d4: Square = .{ .u = 27 };
    pub const e4: Square = .{ .u = 28 };
    pub const f4: Square = .{ .u = 29 };
    pub const g4: Square = .{ .u = 30 };
    pub const h4: Square = .{ .u = 31 };
    pub const a5: Square = .{ .u = 32 };
    pub const b5: Square = .{ .u = 33 };
    pub const c5: Square = .{ .u = 34 };
    pub const d5: Square = .{ .u = 35 };
    pub const e5: Square = .{ .u = 36 };
    pub const f5: Square = .{ .u = 37 };
    pub const g5: Square = .{ .u = 38 };
    pub const h5: Square = .{ .u = 39 };
    pub const a6: Square = .{ .u = 40 };
    pub const b6: Square = .{ .u = 41 };
    pub const c6: Square = .{ .u = 42 };
    pub const d6: Square = .{ .u = 43 };
    pub const e6: Square = .{ .u = 44 };
    pub const f6: Square = .{ .u = 45 };
    pub const g6: Square = .{ .u = 46 };
    pub const h6: Square = .{ .u = 47 };
    pub const a7: Square = .{ .u = 48 };
    pub const b7: Square = .{ .u = 49 };
    pub const c7: Square = .{ .u = 50 };
    pub const d7: Square = .{ .u = 51 };
    pub const e7: Square = .{ .u = 52 };
    pub const f7: Square = .{ .u = 53 };
    pub const g7: Square = .{ .u = 54 };
    pub const h7: Square = .{ .u = 55 };
    pub const a8: Square = .{ .u = 56 };
    pub const b8: Square = .{ .u = 57 };
    pub const c8: Square = .{ .u = 58 };
    pub const d8: Square = .{ .u = 59 };
    pub const e8: Square = .{ .u = 60 };
    pub const f8: Square = .{ .u = 61 };
    pub const g8: Square = .{ .u = 62 };
    pub const h8: Square = .{ .u = 63 };

    pub const all: [Square.count]Square = .{
        a1, b1, c1, d1, e1, f1, g1, h1,
        a2, b2, c2, d2, e2, f2, g2, h2,
        a3, b3, c3, d3, e3, f3, g3, h3,
        a4, b4, c4, d4, e4, f4, g4, h4,
        a5, b5, c5, d5, e5, f5, g5, h5,
        a6, b6, c6, d6, e6, f6, g6, h6,
        a7, b7, c7, d7, e7, f7, g7, h7,
        a8, b8, c8, d8, e8, f8, g8, h8,
    };

    /// Top down squares for print.
    pub const all_for_printing: [Square.count]Square = .{
        a8, b8, c8, d8, e8, f8, g8, h8,
        a7, b7, c7, d7, e7, f7, g7, h7,
        a6, b6, c6, d6, e6, f6, g6, h6,
        a5, b5, c5, d5, e5, f5, g5, h5,
        a4, b4, c4, d4, e4, f4, g4, h4,
        a3, b3, c3, d3, e3, f3, g3, h3,
        a2, b2, c2, d2, e2, f2, g2, h2,
        a1, b1, c1, d1, e1, f1, g1, h1,
    };

    const colors: [Square.count]Color = .{
        Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white,
        Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black,
        Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white,
        Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black,
        Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white,
        Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black,
        Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white,
        Color.white, Color.black, Color.white, Color.black, Color.white, Color.black, Color.white, Color.black,
    };

    const manhattan_distances_to_center: [Square.count]u8 = .{
        6, 5, 4, 3, 3, 4, 5, 6,
        5, 4, 3, 2, 2, 3, 4, 5,
        4, 3, 2, 1, 1, 2, 3, 4,
        3, 2, 1, 0, 0, 1, 2, 3,
        3, 2, 1, 0, 0, 1, 2, 3,
        4, 3, 2, 1, 1, 2, 3, 4,
        5, 4, 3, 2, 2, 3, 4, 5,
        6, 5, 4, 3, 3, 4, 5, 6
    };

    const manhattan_distances_to_corner: [Square.count]u8 = .{
        0, 1, 2, 3, 3, 2, 1, 0,
        1, 2, 3, 4, 4, 3, 2, 1,
        2, 3, 4, 5, 5, 4, 3, 2,
        3, 4, 5, 6, 6, 5, 4, 3,
        3, 4, 5, 6, 6, 5, 4, 3,
        2, 3, 4, 5, 5, 4, 3, 2,
        1, 2, 3, 4, 4, 3, 2, 1,
        0, 1, 2, 3, 3, 2, 1, 0
    };

    pub fn from(index: u6) Square {
        return .{ .u = index };
    }

    pub fn from_usize(index: usize) Square {
        if (comptime lib.is_paranoid) {
            assert(index < 64);
        }
        //return .{ .u = @truncate(index) };
        return .{ .u = @intCast(index) };
    }

    pub fn idx(self: Square) usize {
        return self.u;
    }

    pub fn from_rank_file(r: u3, f: u3) Square {
        return .{ .coord = .{.file = f, .rank = r} };
    }

    pub fn file(self: Square) u3 {
        return self.coord.file;
    }

    pub fn rank(self: Square) u3 {
        return self.coord.rank;
    }

    pub fn color(self: Square) Color {
        const c: u1 = @intCast(((self.u ^ (self.coord.rank)) & 1) ^ 1); // #testing
        return .{ .u = c };
        // assert(c == colors[self.u].u);
        //return colors[self.u];
    }

    pub fn manhattan_distance_to_center(self: Square) u8 {
        return manhattan_distances_to_center[self.u];
    }

    pub fn manhattan_distance_to_corner(self: Square) u8 {
        return manhattan_distances_to_corner[self.u];
    }

    pub fn to_bitboard(self: Square) u64 {
        return @as(u64, 1) << self.u;
    }

    pub fn inc(self: *Square, d: i7) Square {
        self.u +%= d;
    }

    pub fn add(self: Square, d: u6) Square {
        return .{ .u = self.u + d };
    }

    pub fn sub(self: Square, d: u6) Square {
        return .{ .u = self.u - d };
    }

    /// Returns the square when `us` is white otherwise the vertically mirrored square.
    pub fn relative(self: Square, us: Color) Square {
        return if (us.e == .white) self else .{ .u = self.u ^ 56 };
    }

    pub fn flipped(self: Square) Square {
        return .{ .u = self.u ^ 56 };
    }

    /// Only used during initialization.
    pub fn next(self: Square, dir: Direction) ?Square {
        if (!@inComptime()) {
            @compileError("only in comptime");
        }
        switch(dir) {
            .north => return if (self.coord.rank < 7) self.add(8) else null,
            .east => return if (self.coord.file < 7) self.add(1) else null,
            .south => return if (self.coord.rank > 0) self.sub(8) else null,
            .west => return if (self.coord.file > 0) self.sub(1) else null,
            .north_west => return if (self.coord.rank < 7 and self.coord.file > 0) self.add(7) else null,
            .north_east => return if (self.coord.rank < 7 and self.coord.file < 7) self.add(9) else null,
            .south_east => return if (self.coord.rank > 0 and self.coord.file < 7) self.sub(7) else null,
            .south_west => return if (self.coord.rank > 0 and self.coord.file > 0) self.sub(9) else null,
        }
    }

    /// Only used during initialization (knight moves).
    pub fn next_twice(self: Square, dir1: Direction, dir2: Direction) ?Square {
        if (self.next(dir1)) |a| {
            if (a.next(dir2)) |b| {
                return b;
            }
        }
        return null;
    }

    /// Only used during initialization.
    /// * Returns a ray of squares in the direction `dir`, not including self.
    pub fn ray(self: Square, dir: Direction) utils.BoundedArray(Square, 8) {
        var result: utils.BoundedArray(Square, 8) = .{};
        var run: Square = self;
        while (true) {
            if (run.next(dir)) |n| {
                result.append_assume_capacity(n);
                run = n;
            }
            else break;
        }
        return result;
    }

    /// Only used during initialization.
    /// * Returns all rays for a range of directions, not including self.
    // pub fn rays(self: Square, comptime dirs: []const Direction) utils.BoundedArray(Square, 32) {
    //     var result: utils.BoundedArray(Square, 32) = .{};
    //     for (dirs) |d| {
    //         result.append_slice_assume_capacity(self.ray(d).slice());
    //     }
    //     return result;
    // }

    /// Only used during initialization.
    pub fn ray_bitboard(self: Square, dir: Direction) u64 {
        var bb: u64 = 0;
        var run: Square = self;
        while (run.next(dir)) |n| {
            bb |=  n.to_bitboard();
            run = n;
        }
        return bb;
    }

    /// Only used during initialization.
    pub fn rays_bitboard(self: Square, comptime dirs: []const Direction) u64 {
        var bb: u64 = 0;
        inline for (dirs) |d| {
            bb |= self.ray_bitboard(d);
        }
        return bb;
    }

    pub fn to_string(self: Square) []const u8 {
        return @tagName(self.e);
    }

    /// Garbage in garbage out. No crash.
    pub fn from_string(str: []const u8) Square {
        if (str.len < 2) return Square.zero;
        // This math can never crash
        const v: u6 = @truncate((str[1] -| '1') *% 8 +| (str[0] -| 'a')); // TODO: intcast instead of truncate?
        return .{ .u = v };
    }

    pub fn char_of_rank(self: Square) u8 {
        return @as(u8, '1') + self.rank();
    }

    pub fn char_of_file(self: Square) u8 {
        return @as(u8, 'a') + self.file();
    }
};

pub const Move = packed struct(u16) {
    pub const silent                   : u4 = 0b0000; // 0
    pub const double_push              : u4 = 0b0001; // 1
    pub const castle_short             : u4 = 0b0010; // 2
    pub const castle_long              : u4 = 0b0011; // 3
    pub const knight_promotion         : u4 = 0b0100; // 4
    pub const bishop_promotion         : u4 = 0b0101; // 5
    pub const rook_promotion           : u4 = 0b0110; // 6
    pub const queen_promotion          : u4 = 0b0111; // 7
    pub const capture                  : u4 = 0b1000; // 8
    pub const ep                       : u4 = 0b1001; // 9
    pub const knight_promotion_capture : u4 = 0b1100; // 12
    pub const bishop_promotion_capture : u4 = 0b1101; // 13
    pub const rook_promotion_capture   : u4 = 0b1110; // 14
    pub const queen_promotion_capture  : u4 = 0b1111; // 15

    pub const capture_mask             : u4 = 0b1000; // bit 3 = capture.
    pub const promotion_mask           : u4 = 0b0100; // bit 2 = promotion
    pub const noisy_mask               : u4 = capture_mask | promotion_mask;

    pub const castles: [CastleType.count]u4 = .{ castle_short, castle_long };

    /// 6 bits.
    from: Square = .zero,
    /// 6 bits.
    to: Square = .zero,
    /// Detail flags.
    flags: u4 = 0,

    pub const empty: Move = .{};

    pub fn init(from: Square, to: Square, flags: u4) Move {
        return .{ .from = from, .to = to, .flags = flags };
    }

    pub fn bitcast(self: Move) u16 {
        return @bitCast(self);
    }

    pub fn is_empty(self: Move) bool {
        return self == empty;
    }

    pub fn is_capture(self: Move) bool {
        return self.flags & capture != 0;
    }

    pub fn is_quiet(self: Move) bool {
        if (comptime lib.is_paranoid) {
            assert(!self.is_empty());
        }
        return self.flags & noisy_mask == 0;
    }

    pub fn is_noisy(self: Move) bool {
        return self.flags & noisy_mask != 0;
    }

    pub fn is_promotion(self: Move) bool {
        return self.flags & 0b0100 != 0;
    }

    pub fn is_ep(self: Move) bool {
        return self.flags == ep;
    }

    pub fn is_castle(self: Move) bool {
        return self.flags == castle_short or self.flags == castle_long;
    }

    pub fn flipped(self: Move) Move {
        if (self.is_empty()) {
            return self;
        }
        return. { .from = self.from.flipped(), .to = self.to.flipped(), .flags = self.flags };
    }

    /// Only valid when we are a promotion.
    pub fn promoted_to(self: Move) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.is_promotion());
        }
        return .{ .u = (self.flags & 0b0111) - 3 };
    }

    /// Returns a 12 bit indexer for the from / to squares.
    pub fn from_to(self: Move) u12 {
        return @truncate(bitcast(self));
    }

    /// Prints uci move to stdout.
    pub fn print_buffered(self: Move, is_960: bool) void {
        if (self.is_empty()) {
            io.print_buffered("0000", .{});
            return;
        }
        const from: Square = self.from;
        var to: Square = self.to;

        // Only in classic chess we need to decode our "king takes rook". In Chess960 this is default.
        if (!is_960) {
            if (self.flags == Move.castle_short) {
                const color: Color = if (to.u < 8) Color.white else Color.black;
                to = position.Castling.king_dest(color, CastleType.short);
            }
            else if (self.flags == Move.castle_long) {
                const color: Color = if (to.u < 8) Color.white else Color.black;
                to = position.Castling.king_dest(color, CastleType.long);
            }
        }

        io.print_buffered("{t}{t}", .{ from.e, to.e });

        if (self.is_promotion()) {
            const prom: PieceType = self.promoted_to();
            const ch: u8 = "?nbrq?"[prom.u];
            io.print_buffered("{u}", .{ ch });
        }
    }

    pub fn to_string(self: Move, is_960: bool) utils.BoundedArray(u8, 5) {
        var result: utils.BoundedArray(u8, 5) = .{};
        const from: Square = self.from;
        var to: Square = self.to;

        // Only in classic chess we need to decode our "king takes rook". In Chess960 this is default.
        if (!is_960) {
            if (self.flags == Move.castle_short) {
                const color: Color = if (to.u < 8) Color.white else Color.black;
                to = position.king_castle_destination_squares[color.u][CastleType.short.u];
            }
            else if (self.flags == Move.castle_long) {
                const color: Color = if (to.u < 8) Color.white else Color.black;
                to = position.king_castle_destination_squares[color.u][CastleType.long.u];
            }
        }

        result.print_assume_capacity("{t}", .{ from.e});
        result.print_assume_capacity("{t}", .{ to.e});

        if (self.is_promotion()) {
            const prom = self.promoted_to();
            const ch: u8 = "?nbrq?"[prom.u];
            result.print_assume_capacity("{u}", .{ ch });
        }
        return result;
    }
};

pub const ExtMove = packed struct {
    move: Move = .empty,
    /// Set during move generation.
    piece: Piece = Piece.no_piece,
    /// Set during move generation.
    captured: Piece = Piece.no_piece,
    /// Set by movepicker during search.
    score: i32 = 0,
    /// Set by movepicker during search.
    is_tt_move: bool = false,
    /// Set by movepicker during search.
    is_bad_capture: bool = false,

    pub const empty: ExtMove = .{};

    pub fn init(from: Square, to: Square, flags: u4, piece: Piece, captured: Piece) ExtMove {
        return .{
            .move = .{ .from = from, .to = to, .flags = flags },
            .piece = piece,
            .captured = captured,
        };
    }
};

/// Simple array wrapper.
pub fn ExtMoveList(max: u8) type {
    return struct {
        const Self = @This();

        extmoves: [max]ExtMove,
        count: u8,

        pub fn init() Self {
            return .{ .extmoves = undefined, .count = 0 };
        }

        /// Assumes witin bounds.
        pub fn add(self: *Self, ex: ExtMove) void {
            self.extmoves[self.count] = ex;
            self.count += 1;
        }

        /// Only adds if possible.
        pub fn try_add(self: *Self, ex: ExtMove) void {
            if (self.count < max) {
                self.add(ex);
            }
        }

        pub fn slice(self: *Self) []ExtMove {
            return self.extmoves[0..self.count];
        }
    };
}

pub const ScorePair = struct {
    /// Middlegame
    mg: i16,
    /// Endgame
    eg: i16,

    pub const empty: ScorePair = .{ .mg = 0, .eg = 0 };

    pub fn init(mg: i16, eg: i16) ScorePair {
        return .{ .mg = mg, .eg = eg };
    }

    pub fn inc(self: *ScorePair, sp: ScorePair) void {
        self.mg += sp.mg;
        self.eg += sp.eg;
    }

    pub fn dec(self: *ScorePair, sp: ScorePair) void {
        self.mg -= sp.mg;
        self.eg -= sp.eg;
    }

    pub fn add(self: ScorePair, other: ScorePair) ScorePair {
        return .{ .mg = self.mg + other.mg, .eg = self.eg + other.eg };
    }

    pub fn sub(self: ScorePair, other: ScorePair) ScorePair {
        return .{ .mg = self.mg - other.mg, .eg = self.eg - other.eg };
    }

    pub fn mul(self: ScorePair, m: u8) ScorePair {
        return .{ .mg = self.mg * m, .eg = self.eg * m };
    }

    pub fn fmul(self: ScorePair, factor: f32) ScorePair {
        return .{ .mg = funcs.fmul(self.mg, factor), .eg = funcs.fmul(self.eg, factor)};
    }
};

/// Easy initialization function for eval tables.
pub fn pair(mg: i16, eg: i16) ScorePair {
    return .{ .mg = mg, .eg = eg };
}

pub const ParsingError = error {
    /// Garbage piece inside fen string.
    InvalidFenPiece,
    /// Garbage or move not found.
    IllegalMove,
    /// Garbage promotion character.
    InvalidPromotionChar,
};

// --- Constants ---
pub const megabyte: usize = 1024 * 1024;
pub const million: usize = 1000 * 1000;

/// This is how far we go.
pub const max_game_length: usize = 1024;
/// The absoluta maximum number of moves in a position.
pub const max_move_count: u8 = 224;
/// The absoluta maximum number of noisy moves in a position.
pub const max_noisy_count: u8 = 128;
/// Our max search depth during search. All arrays are a bit oversized for safety.
pub const max_search_depth: u8 = 128;

// Scores for SEE and move ordering.
pub const value_pawn: i32 = 98;
pub const value_knight: i32 = 299;
pub const value_bishop: i32 = 300;
pub const value_rook: i32 = 533;
pub const value_queen: i32 = 921;
pub const value_king: i32 = 0;

const piece_values: [13]i32 = .{
    value_pawn, value_knight, value_bishop, value_rook, value_queen, value_king,
    value_pawn, value_knight, value_bishop, value_rook, value_queen, value_king,
    0,
};

pub const simple_value_pawn: i32 = 100;
pub const simple_value_knight: i32 = 300;
pub const simple_value_bishop: i32 = 300;
pub const simple_value_rook: i32 = 500;
pub const simple_value_queen: i32 = 900;
pub const simple_value_king: i32 = 0;

const simple_piece_values: [13]i32 = .{
    simple_value_pawn, simple_value_knight, simple_value_bishop, simple_value_rook, simple_value_queen, simple_value_king,
    simple_value_pawn, simple_value_knight, simple_value_bishop, simple_value_rook, simple_value_queen, simple_value_king,
    0,
};

pub const ph_minor: u8 = 1;
pub const ph_rook: u8 = 2;
pub const ph_queen: u8 = 4;

/// Game phase table. Indexing by [piece]
pub const phase_table: [13]u8 = .{
    0, 1, 1, 2, 4, 0,
    0, 1, 1, 2, 4, 0,
    0
};
pub const max_phase: u8 = 24;

pub fn phased_score(phase: u8, score: ScorePair) i32 {
    const ph: u8 = @min(max_phase, phase);
    const mg: i32 = score.mg;
    const eg: i32 = score.eg;
    return @divFloor(mg * ph + eg * (max_phase - ph), max_phase);
}
