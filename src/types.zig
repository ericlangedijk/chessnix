// zig fmt: off

//! Basic types and consts, used almost everywhere.

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const io = lib.io;

const Castling = position.Castling;

comptime {
    if (@sizeOf(Move) != 2) @compileError("Move size");
    if (@sizeOf(ExtMove) != 8) @compileError("ExtMove size");
}

/// Win Draw Loss
pub const WDL = packed union {
    e: E,
    u: u8,

    pub const E = enum(u8) {
        loss = 0,
        draw = 1,
        win = 2,
    };

    pub const loss: WDL = .{ .e = .loss };
    pub const draw: WDL = .{ .e = .draw };
    pub const win: WDL = .{ .e = .win };

    pub fn from_int(u: usize) WDL {
        return .{ .u = @intCast(u)};
    }

    pub fn flipped(self: WDL) WDL {
        return .{ .u = 2 - self.u };
    }
};

pub const Axis = enum(u2) {
    no_axis = 0,
    orth = 1,
    diag = 2,
};

pub const Orientation = enum(u2) {
    horizontal = 0,
    vertical = 1,
    diagmain = 2,
    diaganti = 3,
};

pub const Direction = enum(u3) {
    north = 0,
    east = 1,
    south = 2,
    west = 3,
    north_west = 4,
    north_east = 5,
    south_east = 6,
    south_west = 7,

    pub const count = 8;

    pub const all: [Direction.count]Direction = .{
        .north, .east, .south, .west, .north_west, .north_east, .south_east, .south_west
    };

    pub fn relative(self: Direction, comptime color: Color) Direction {
        if (color.e == .white) return self;
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

pub const Castle = packed union {
    e: E,
    u: u1,

    pub const count: usize = 2;

    pub const E = enum(u1) {
        short,
        long
    };

    pub const short: Castle = .{ .e = .short };
    pub const long: Castle = .{ .e = .long };

    pub const all: [Castle.count]Castle = .{ short, long };
};

pub const Color = packed union {
    e: E,
    u: u1,

    pub const count: usize = 2;

    pub const E = enum(u1) {
        white,
        black
    };

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
    e: E,
    u: u4,

    pub const count = 6;

    pub const E = enum(u4) {
        pawn = 0,
        knight = 1,
        bishop = 2,
        rook = 3,
        queen = 4,
        king = 5,
        no_piecetype = 6,
    };

    pub const pawn: PieceType = .{ .e = .pawn };
    pub const knight: PieceType = .{ .e = .knight };
    pub const bishop: PieceType = .{ .e = .bishop };
    pub const rook: PieceType = .{ .e = .rook };
    pub const queen: PieceType = .{ .e = .queen };
    pub const king: PieceType = .{ .e = .king };
    pub const no_piecetype: PieceType = .{ .e = .no_piecetype };

    pub const all: [count]PieceType = .{
        pawn, knight, bishop, rook, queen, king
    };

    pub fn from_int(u: usize) PieceType {
        return .{ .u = @intCast(u)};
    }

    pub fn idx(self: PieceType) usize {
        return self.u;
    }

    pub fn value(self: PieceType) i32 {
        return piece_values[self.u];
    }

    pub fn simple_value(self: PieceType) i32 {
        return simple_piece_values[self.u];
    }

    pub fn to_promotion_char(self: PieceType) u8 {
        return "?nbrq??"[self.u];
    }

    /// TODO: move this function to Move?
    pub fn to_promotion_move_flag(self: PieceType) u4 {
        return switch (self.e) {
            .knight => Move.knight_promotion,
            .bishop => Move.bishop_promotion,
            .rook => Move.rook_promotion,
            .queen => Move.queen_promotion,
            else => unreachable,
        };
        //return "?nbrq??"[self.u];
    }

    pub fn to_char(self: PieceType) u8 {
        return switch(self.e) {
            .pawn => 0,
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
            .no_piecetype => '?',
        };
    }

    // pub fn format(self: PieceType, writer: *std.io.Writer) std.io.Writer.Error!void {
    //     const chars: [7]u8 = comptime "?nbrq??";
    //     try writer.print("{u}", .{ chars[self.u] });
    // }
};

pub const Piece = packed union {
    e: E,
    u: u4,

    pub const count: usize = 12;

    pub const E = enum(u4) {
        white_pawn   = 0,
        white_knight = 1,
        white_bishop = 2,
        white_rook   = 3,
        white_queen  = 4,
        white_king   = 5,
        black_pawn   = 6,
        black_knight = 7,
        black_bishop = 8,
        black_rook   = 9,
        black_queen  = 10,
        black_king   = 11,
        no_piece     = 12,
    };


    /// All valid pieces.
    pub const all: [count]Piece = .{
        white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
        black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king,
    };

    pub const white_pawn   : Piece = .{ .e = .white_pawn };
    pub const white_knight : Piece = .{ .e = .white_knight };
    pub const white_bishop : Piece = .{ .e = .white_bishop };
    pub const white_rook   : Piece = .{ .e = .white_rook };
    pub const white_queen  : Piece = .{ .e = .white_queen };
    pub const white_king   : Piece = .{ .e = .white_king };
    pub const black_pawn   : Piece = .{ .e = .black_pawn };
    pub const black_knight : Piece = .{ .e = .black_knight };
    pub const black_bishop : Piece = .{ .e = .black_bishop };
    pub const black_rook   : Piece = .{ .e = .black_rook };
    pub const black_queen  : Piece = .{ .e = .black_queen };
    pub const black_king   : Piece = .{ .e = .black_king };
    pub const no_piece     : Piece = .{ .e = .no_piece };

    pub fn init(pt: PieceType, side: Color) Piece {
        return if (side.e == .white) .{ .u = pt.u } else .{ .u = pt.u + 6 };
    }

    pub fn is_empty(self: Piece) bool {
        return self.e == .no_piece;
    }

    pub fn color(self: Piece) Color {
        if (comptime lib.is_paranoid) {
            assert(self.e != .no_piece);
        }
        return if (self.u < 6) .white else .black;
    }

    pub fn is_white(self: Piece) bool {
        return self.u < 6;
    }

    pub fn is_color(self: Piece, comptime us: Color) bool {
        return if (us.e == .white) self.u < 6 else self.u >= 6 and self.u <= 11;
    }

    pub fn piecetype(self: Piece) PieceType {
        return if (self.u < 6) .{ .u = self.u } else .{.u = self. u - 6 };
    }

    /// Used for flipping the board.
    pub fn opp(self: Piece) Piece {
        if (self.is_empty()) {
            return .no_piece;
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
        const pt: PieceType = self.piecetype();
        return pt.e == .knight or pt.e == .bishop;
    }

    pub fn is_major(self: Piece) bool {
        const pt: PieceType = self.piecetype();
        return pt.e == .rook or pt.e == .queen;
    }

    pub fn is_king(self: Piece) bool {
        return self.piecetype().e == .king;
    }

    /// Micro optimization.
    pub fn is_pawn_of_color(self: Piece, us: Color) bool {
        return if (us.e == .white) self.e == .white_pawn else self.e == .black_pawn;
    }

    /// Micro optimization.
    pub fn is_rook_of_color(self: Piece, us: Color) bool {
        return if (us.e == .white) self.e == .white_rook else self.e == .black_rook;
    }

    /// Micro optimization.
    pub fn is_king_of_color(self: Piece, us: Color) bool {
        return if (us.e == .white) self.e == .white_king else self.e == .black_king;
    }

    /// Micro optimization.
    pub fn is_minor_of_color(self: Piece, us: Color) bool {
        return if (us.e == .white) self.e == .white_knight or self.e == .white_bishop else self.e == .black_knight or self.e == .black_bishop;
    }

    /// Micro optimization.
    pub fn is_major_of_color(self: Piece, us: Color) bool {
        return if (us.e == .white) self.e == .white_rook or self.e == .white_queen else self.e == .black_rook or self.e == .black_queen;
    }

    /// Returns the static exchange evaluation value. It is allowed to call this for no-piece.
    pub fn value(self: Piece) i32 {
        return piece_values[self.u];
    }

    pub fn simple_value(self: Piece) i32 {
        return simple_piece_values[self.u];
    }

    pub fn to_print_char(self: Piece) u8 {
        return switch (self.e) {
            .white_pawn => 'P',
            .white_knight => 'N',
            .white_bishop => 'B',
            .white_rook => 'R',
            .white_queen => 'Q',
            .white_king => 'K',

            .black_pawn => 'p',
            .black_knight => 'n',
            .black_bishop => 'b',
            .black_rook => 'r',
            .black_queen => 'q',
            .black_king => 'k',
            .no_piece => '?',
        };

        // var ch: u8 = switch(self.piecetype().e) {
        //     .pawn => 'P',
        //     .knight => 'N',
        //     .bishop => 'B',
        //     .rook => 'R',
        //     .queen => 'Q',
        //     .king => 'K',
        //     .none => '?',
        // };
        // if (self.color().e == .black) ch = std.ascii.toLower(ch);
        // return ch;
    }

    pub fn to_char(self: Piece) u8 {
        return "PNBRQKpnbrqk?"[self.u];
    }

    pub fn from_char(char: u8) ParsingError!Piece {
        return switch(char) {
            'P' => white_pawn,
            'N' => white_knight,
            'B' => white_bishop,
            'R' => white_rook,
            'Q' => white_queen,
            'K' => white_king,
            'p' => black_pawn,
            'n' => black_knight,
            'b' => black_bishop,
            'r' => black_rook,
            'q' => black_queen,
            'k' => black_king,
            else => ParsingError.InvalidFenPiece,
        };
    }
};

pub const rank_1 : u3 = 0;
pub const rank_2 : u3 = 1;
pub const rank_3 : u3 = 2;
pub const rank_4 : u3 = 3;
pub const rank_5 : u3 = 4;
pub const rank_6 : u3 = 5;
pub const rank_7 : u3 = 6;
pub const rank_8 : u3 = 7;

pub const file_a : u3 = 0;
pub const file_b : u3 = 1;
pub const file_c : u3 = 2;
pub const file_d : u3 = 3;
pub const file_e : u3 = 4;
pub const file_f : u3 = 5;
pub const file_g : u3 = 6;
pub const file_h : u3 = 7;

pub const all_ranks: [8]u3 = .{ rank_1, rank_2, rank_3, rank_4, rank_5, rank_6, rank_7, rank_8 };
pub const all_files: [8]u3 = .{ file_a, file_b, file_c, file_d, file_e, file_f, file_g, file_h };

pub const Coord = packed struct(u6) {
    file: u3, // file = x. square % 8 or square & 7 (0b000111)
    rank: u3, // rank = y. square / 8 or square >> 3  (0b111000)
};

pub const Square = packed union {
    e: E,
    u: u6,
    coord: Coord,

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

    pub const a1 = sq(0);  pub const b1 = sq(1);  pub const c1 = sq(2);  pub const d1 = sq(3);  pub const e1 = sq(4);  pub const f1 = sq(5);  pub const g1 = sq(6);  pub const h1 = sq(7);
    pub const a2 = sq(8);  pub const b2 = sq(9);  pub const c2 = sq(10); pub const d2 = sq(11); pub const e2 = sq(12); pub const f2 = sq(13); pub const g2 = sq(14); pub const h2 = sq(15);
    pub const a3 = sq(16); pub const b3 = sq(17); pub const c3 = sq(18); pub const d3 = sq(19); pub const e3 = sq(20); pub const f3 = sq(21); pub const g3 = sq(22); pub const h3 = sq(23);
    pub const a4 = sq(24); pub const b4 = sq(25); pub const c4 = sq(26); pub const d4 = sq(27); pub const e4 = sq(28); pub const f4 = sq(29); pub const g4 = sq(30); pub const h4 = sq(31);
    pub const a5 = sq(32); pub const b5 = sq(33); pub const c5 = sq(34); pub const d5 = sq(35); pub const e5 = sq(36); pub const f5 = sq(37); pub const g5 = sq(38); pub const h5 = sq(39);
    pub const a6 = sq(40); pub const b6 = sq(41); pub const c6 = sq(42); pub const d6 = sq(43); pub const e6 = sq(44); pub const f6 = sq(45); pub const g6 = sq(46); pub const h6 = sq(47);
    pub const a7 = sq(48); pub const b7 = sq(49); pub const c7 = sq(50); pub const d7 = sq(51); pub const e7 = sq(52); pub const f7 = sq(53); pub const g7 = sq(54); pub const h7 = sq(55);
    pub const a8 = sq(56); pub const b8 = sq(57); pub const c8 = sq(58); pub const d8 = sq(59); pub const e8 = sq(60); pub const f8 = sq(61); pub const g8 = sq(62); pub const h8 = sq(63);

    pub const zero: Square = a1;

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

    /// Comptime only.
    inline fn sq(u: u6) Square {
        return .from(u);
    }

    pub fn from(index: u6) Square {
        return .{ .u = index };
    }

    pub fn from_int(u: usize) Square {
        if (comptime lib.is_paranoid) {
            assert(u < 64);
        }
        return .{ .u = @intCast(u) };
    }

    pub fn idx(self: Square) usize {
        return self.u;
    }

    pub fn as_u16(self: Square) u16 {
        return self.u;
    }

    pub fn from_rank_file(r: u3, f: u3) Square {
        return .{ .coord = .{.file = f, .rank = r} };
    }

    pub fn color(self: Square) Color {
        const c: u1 = @intCast(((self.u ^ (self.coord.rank)) & 1) ^ 1); // #testing
        return .{ .u = c };

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

    pub fn from_bitboard(bitboard: u64) Square {
        return funcs.first_square(bitboard);
        //return @as(u64, 1) << self.u;
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
    pub fn rays(self: Square, comptime dirs: []const Direction) utils.BoundedArray(Square, 32) {
        var result: utils.BoundedArray(Square, 32) = .{};
        for (dirs) |d| {
            result.append_slice_assume_capacity(self.ray(d).slice());
        }
        return result;
    }

    /// Only used during initialization.
    pub fn ray_bitboard(self: Square, dir: Direction) u64 {
        var bb: u64 = 0;
        var run: Square = self;
        while (run.next(dir)) |n| {
            bb |= n.to_bitboard();
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
        if (str.len < 2) return Square.a1;
        // This math can never crash
        const v: u6 = @truncate((str[1] -| '1') *% 8 +| (str[0] -| 'a'));
        return .{ .u = v };
    }

    pub fn char_of_rank(self: Square) u8 {
        return @as(u8, '1') + self.coord.rank;
    }

    pub fn char_of_file(self: Square) u8 {
        return @as(u8, 'a') + self.coord.file;
    }

    // pub fn format(self: Square, writer: *std.io.Writer) std.io.Writer.Error!void {
    //     try writer.print("{t}", .{ self.e });
    // }

};

pub const Move = packed struct(u16) {
    from: Square = .zero,
    to: Square = .zero,
    kind: u4 = 0,

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
    pub const promotion_mask           : u4 = 0b0100; // bit 2 = promotion.
    pub const noisy_mask               : u4 = capture_mask | promotion_mask;

    pub const empty: Move = .{};

    /// Sometimes we need something simple.
    pub const SimpleKind = enum(u2) {
        default,
        ep,
        castle,
        promotion,
    };

    pub fn init(from: Square, to: Square, kind: u4) Move {
        return .{ .from = from, .to = to, .kind = kind };
    }

    pub fn init_promotion(from: Square, to: Square, pt: PieceType, is_capt: bool) Move {
        const capt_flag = if (is_capt) capture else 0;
        const prom_flag = pt.to_promotion_move_flag();
        return .{ .from = from, .to = to, .kind = prom_flag | capt_flag };
    }

    pub fn bitcast(self: Move) u16 {
        return @bitCast(self);
    }

    pub fn castle(comptime ct: Castle) u4 {
        return if (ct.e == .short) castle_short else castle_long;
    }

    pub fn is_empty(self: Move) bool {
        return self == empty;
    }

    pub fn is_capture(self: Move) bool {
        return self.kind & capture_mask != 0;
    }

    pub fn is_quiet(self: Move) bool {
        if (comptime lib.is_paranoid) {
            assert(!self.is_empty());
        }
        return self.kind & noisy_mask == 0;
    }

    pub fn is_noisy(self: Move) bool {
        return self.kind & noisy_mask != 0;
    }

    pub fn is_promotion(self: Move) bool {
        return self.kind & promotion_mask != 0;
    }

    pub fn is_ep(self: Move) bool {
        return self.kind == ep;
    }

    pub fn is_castle(self: Move) bool {
        return self.kind == castle_short or self.kind == castle_long;
    }

    // pub fn flipped(self: Move) Move {
    //     if (self.is_empty()) {
    //         return self;
    //     }
    //     return. { .from = self.from.flipped(), .to = self.to.flipped(), .kind = self.kind };
    // }

    /// Only valid when we are a promotion.
    pub fn prom(self: Move) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.is_promotion());
        }
        return .{ .u = (self.kind & 0b0111) - 3 };
    }

    /// Returns promotion piece if this is a promotion move otherwise no_piece.
    pub fn prom_safe(self: Move) PieceType {
        if (!self.is_promotion()) {
            return .no_piecetype;
        }
        return .{ .u = (self.kind & 0b0111) - 3 };
    }

    /// Returns a 12 bit indexer for the from / to squares.
    pub fn from_to(self: Move) u12 {
        return @truncate(bitcast(self));
    }

    pub fn simple_kind(self: Move) SimpleKind {
        return switch (self.kind) {
            silent, double_push, capture => .default,
            castle_short, castle_long => .castle,
            ep => .ep,
            else => .promotion,
        };

    // pub const silent                   : u4 = 0b0000; // 0
    // pub const double_push              : u4 = 0b0001; // 1
    // pub const castle_short             : u4 = 0b0010; // 2
    // pub const castle_long              : u4 = 0b0011; // 3

    //     return switch (self.kind) {
    //         ep =>
    //             .ep,
    //         castle_short, castle_long =>
    //             .castle,
    //         knight_promotion, bishop_promotion, rook_promotion, queen_promotion,
    //         knight_promotion_capture, bishop_promotion_capture, rook_promotion_capture, queen_promotion_capture =>
    //             .promotion,
    //         else => .default,
    //     };
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
            if (self.kind == castle_short) {
                const color: Color = if (to.u < 8) .white else .black;
                to = Castling.king_dest(color, .short);
            }
            else if (self.kind == castle_long) {
                const color: Color = if (to.u < 8) .white else .black;
                to = Castling.king_dest(color, .long);
            }
        }

        io.print_buffered("{t}{t}", .{ from.e, to.e });

        if (self.is_promotion()) {
            const p: PieceType = self.prom();
            const ch: u8 = "?nbrq?"[p.u];
            io.print_buffered("{u}", .{ ch });
        }
    }

    pub fn to_string(self: Move, is_960: bool) utils.BoundedArray(u8, 5) {
        var result: utils.BoundedArray(u8, 5) = .{};
        const from: Square = self.from;
        var to: Square = self.to;

        // Only in classic chess we need to decode our "king takes rook". In Chess960 this is default.
        if (!is_960) {
            if (self.kind == castle_short) {
                const color: Color = if (to.u < 8) .white else .black;
                //to = position.king_castle_destination_squares[color.u][Castle.short.u]; // TODO: use func
                to = Castling.king_dest(color, .short);
            }
            else if (self.kind == castle_long) {
                const color: Color = if (to.u < 8) .white else .black;
                //to = position.king_castle_destination_squares[color.u][Castle.long.u]; // TODO: use func
                to = Castling.king_dest(color, .long);
            }
        }

        result.print_assume_capacity("{t}", .{ from.e });
        result.print_assume_capacity("{t}", .{ to.e });

        if (self.is_promotion()) {
            const p = self.prom();
            const ch: u8 = "?nbrq?"[p.u];
            result.print_assume_capacity("{u}", .{ ch });
        }
        return result;
    }

    pub fn format(self: Move, writer: *std.io.Writer) std.io.Writer.Error!void {
        try writer.print("{t}{t}", .{ self.from.e, self.to.e });
        if (self.is_promotion()) {
            try writer.print("{u}", .{ self.prom().to_promotion_char() });
        }
    }

    // pub fn format(self: Move, options: bool, writer: *std.io.Writer) std.io.Writer.Error!void {
    //     try writer.print("{t}{t}", .{ self.from.e, self.to.e });
    //     if (self.is_promotion()) {
    //         try writer.print("{u}", .{ self.prom().to_promotion_char() });
    //     }
    //     if (options) try writer.print(" with options", .{});
    // }

    pub fn fmt(self: Move) Formatter {
        return .{ .move = self };
    }

    pub const Formatter = struct {
        move: Move,

        pub fn format(self: Formatter, writer: *std.io.Writer) std.io.Writer.Error!void {
            try writer.print("{t}{t}", .{ self.move.from.e, self.move.to.e });
            if (self.move.is_promotion()) {
                try writer.print("{u}", .{ self.move.prom().to_promotion_char() });
            }
        }
    };
};

/// Extended move for the engine.
/// - `move`, `piece`, `captured` are set during move generation.
/// - `score` is set during search by the movepicker.
pub const ExtMove = packed struct(u64) {
    move: Move,
    piece: Piece,
    captured: Piece,
    alignment_padding: u8,
    score: i32,

    pub const empty: ExtMove = .{
        .move = .empty,
        .piece = .no_piece,
        .captured = .no_piece,
        .alignment_padding = 0,
        .score = 0,
    };

    pub fn init(from: Square, to: Square, kind: u4, piece: Piece, captured: Piece) ExtMove {
        return .{
            .move = .{ .from = from, .to = to, .kind = kind },
            .piece = piece,
            .captured = captured,
            .alignment_padding = 0,
            .score = 0,
        };
    }

    pub fn from_move(move: Move, piece: Piece, captured: Piece) ExtMove {
        return .{
            .move = move,
            .piece = piece,
            .captured = captured,
            .alignment_padding = 0,
            .score = 0,
        };
    }
};

/// Simple array wrapper.
pub fn ExtMoveList(max: u8) type {
    return struct {
        const Self = @This();
        pub const max_capacity: u8 = max;

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

        pub fn slice_const(self: *const Self) []const ExtMove {
            return self.extmoves[0..self.count];
        }
    };
}

/// Evaluation score. mg = opening or middlegame, eg = endgame.
/// Extern to guarantee field order during turning.
pub const ScorePair = extern struct {
    mg: i16,
    eg: i16,

    pub const empty: ScorePair = .{ .mg = 0, .eg = 0 };

    pub fn init(mg: i16, eg: i16) ScorePair {
        return .{ .mg = mg, .eg = eg };
    }

    pub fn inc(self: *ScorePair, delta: ScorePair) void {
        self.mg += delta.mg;
        self.eg += delta.eg;
    }

    pub fn dec(self: *ScorePair, delta: ScorePair) void {
        self.mg -= delta.mg;
        self.eg -= delta.eg;
    }

    pub fn add(self: ScorePair, delta: ScorePair) ScorePair {
        return .init(self.mg + delta.mg, self.eg + delta.eg);
    }

    pub fn sub(self: ScorePair, delta: ScorePair) ScorePair {
        return .init(self.mg - delta.mg, self.eg - delta.eg);
    }

    pub fn mul(self: ScorePair, m: u8) ScorePair {
        return .{ .mg = self.mg * m, .eg = self.eg * m };
    }

    pub fn fmul(self: ScorePair, factor: f32) ScorePair {
        return .{ .mg = funcs.fmul(self.mg, factor), .eg = funcs.fmul(self.eg, factor)};
    }

    pub fn format(self: ScorePair, writer: *std.io.Writer) std.io.Writer.Error!void {
        try writer.print("({}, {})", .{ self.mg, self.eg });
    }

};

/// Easy initialization function for eval tables.
pub fn pair(mg: i16, eg: i16) ScorePair {
    return .{ .mg = mg, .eg = eg };
}

pub const ParsingError = error {
    InvalidFenPiece,
    IllegalMove,
    InvalidPromotionChar,
};

// --- Constants ---
pub const megabyte: usize = 1024 * 1024;
pub const million: usize = 1000 * 1000;

pub const max_game_length: usize = 1024;
pub const max_move_count: u8 = 224;
pub const max_noisy_count: u8 = 128;
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

pub fn restrict_phase(phase: u8) u8 {
    return @min(max_phase, phase);
}

pub fn phased_score(phase: u8, score: ScorePair) i32 {
    const ph: u8 = @min(max_phase, phase);
    const mg: i32 = score.mg;
    const eg: i32 = score.eg;
    return @divFloor(mg * ph + eg * (max_phase - ph), max_phase);
}
