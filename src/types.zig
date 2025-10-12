// zig fmt: off

//! The basic types.

const std = @import("std");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const funcs = @import("funcs.zig");
const lib = @import("lib.zig");

const assert = std.debug.assert;

// Much used, we shorten.
pub const P = PieceType.PAWN;
pub const N = PieceType.KNIGHT;
pub const B = PieceType.BISHOP;
pub const R = PieceType.ROOK;
pub const Q = PieceType.QUEEN;
pub const K = PieceType.KING;

/// Used for evaluation.
pub const Value = i32;
/// Used for evaluation.
pub const Float = f32;

pub const Axis = enum(u2) {
    none, orth, diag,
};

pub const Orientation = enum(u2) {
    horizontal, vertical, diagmain, diaganti,
};

pub const Direction = enum(u3) {
    north, east, south, west, north_west, north_east, south_east, south_west,

    pub const all: [8]Direction = .{ .north, .east, .south, .west, .north_west, .north_east, .south_east, .south_west };

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

pub const CastleType = packed union {
    pub const Enum = enum(u1) { short, long };
    /// The enum value
    e: Enum,
    /// The numeric value
    u: u1,

    pub const all: [2]CastleType = .{ SHORT, LONG };
    pub const SHORT: CastleType = .{ .e = .short };
    pub const LONG: CastleType = .{ .e = .long };
};

pub const Color = packed union {
    pub const Enum = enum(u1) { white, black };
    /// The enum value
    e: Enum,
    /// The numeric value
    u: u1,

    pub const all: [2]Color = .{ WHITE, BLACK };
    pub const WHITE: Color = .{ .e = .white };
    pub const BLACK: Color = .{ .e = .black };

    pub fn opp(self: Color) Color {
        return .{ .u  = self.u ^ 1 };
    }

    pub fn idx(self: Color) usize {
        return self.u;
    }
};

pub const Coord = packed struct {
    /// file == x. raw bits: Square.u & 7 (0b000111)
    file: u3,
    /// rank == y. raw bits: Square.u >> 3  (0b111000)
    rank: u3,
};

pub const Square = packed union {
    pub const Enum = enum(u6) {
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
    e: Enum,
    /// The numeric value
    u: u6,
    /// The file and rank bits match nicely.
    coord: Coord,

    pub const all: [64]Square = .{
        A1, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8,
    };

    /// Top down squares for print.
    pub const all_for_printing: [64]Square = .{
        A8, B8, C8, D8, E8, F8, G8, H8,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A1, B1, C1, D1, E1, F1, G1, H1,
    };

    pub const zero: Square = A1;

    pub const A1: Square = .{ .u = 0 };
    pub const B1: Square = .{ .u = 1 };
    pub const C1: Square = .{ .u = 2 };
    pub const D1: Square = .{ .u = 3 };
    pub const E1: Square = .{ .u = 4 };
    pub const F1: Square = .{ .u = 5 };
    pub const G1: Square = .{ .u = 6 };
    pub const H1: Square = .{ .u = 7 };
    pub const A2: Square = .{ .u = 8 };
    pub const B2: Square = .{ .u = 9 };
    pub const C2: Square = .{ .u = 10 };
    pub const D2: Square = .{ .u = 11 };
    pub const E2: Square = .{ .u = 12 };
    pub const F2: Square = .{ .u = 13 };
    pub const G2: Square = .{ .u = 14 };
    pub const H2: Square = .{ .u = 15 };
    pub const A3: Square = .{ .u = 16 };
    pub const B3: Square = .{ .u = 17 };
    pub const C3: Square = .{ .u = 18 };
    pub const D3: Square = .{ .u = 19 };
    pub const E3: Square = .{ .u = 20 };
    pub const F3: Square = .{ .u = 21 };
    pub const G3: Square = .{ .u = 22 };
    pub const H3: Square = .{ .u = 23 };
    pub const A4: Square = .{ .u = 24 };
    pub const B4: Square = .{ .u = 25 };
    pub const C4: Square = .{ .u = 26 };
    pub const D4: Square = .{ .u = 27 };
    pub const E4: Square = .{ .u = 28 };
    pub const F4: Square = .{ .u = 29 };
    pub const G4: Square = .{ .u = 30 };
    pub const H4: Square = .{ .u = 31 };
    pub const A5: Square = .{ .u = 32 };
    pub const B5: Square = .{ .u = 33 };
    pub const C5: Square = .{ .u = 34 };
    pub const D5: Square = .{ .u = 35 };
    pub const E5: Square = .{ .u = 36 };
    pub const F5: Square = .{ .u = 37 };
    pub const G5: Square = .{ .u = 38 };
    pub const H5: Square = .{ .u = 39 };
    pub const A6: Square = .{ .u = 40 };
    pub const B6: Square = .{ .u = 41 };
    pub const C6: Square = .{ .u = 42 };
    pub const D6: Square = .{ .u = 43 };
    pub const E6: Square = .{ .u = 44 };
    pub const F6: Square = .{ .u = 45 };
    pub const G6: Square = .{ .u = 46 };
    pub const H6: Square = .{ .u = 47 };
    pub const A7: Square = .{ .u = 48 };
    pub const B7: Square = .{ .u = 49 };
    pub const C7: Square = .{ .u = 50 };
    pub const D7: Square = .{ .u = 51 };
    pub const E7: Square = .{ .u = 52 };
    pub const F7: Square = .{ .u = 53 };
    pub const G7: Square = .{ .u = 54 };
    pub const H7: Square = .{ .u = 55 };
    pub const A8: Square = .{ .u = 56 };
    pub const B8: Square = .{ .u = 57 };
    pub const C8: Square = .{ .u = 58 };
    pub const D8: Square = .{ .u = 59 };
    pub const E8: Square = .{ .u = 60 };
    pub const F8: Square = .{ .u = 61 };
    pub const G8: Square = .{ .u = 62 };
    pub const H8: Square = .{ .u = 63 };

    pub fn from(index: u6) Square {
        return .{ .u = index };
    }

    pub fn from_usize(index: usize) Square {
        if (comptime lib.is_paranoid) {
            assert(index < 64);
        }
        return .{ .u = @truncate(index) };
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

    pub fn to_bitboard(self: Square) u64 {
        return bitboards.square_bitboards[self.u]; // It seems this is a bit faster than 1 << square.
    }

    pub fn add(self: Square, d: u6) Square {
        return .{ .u = self.u + d };
    }

    pub fn sub(self: Square, d: u6) Square {
        return .{ .u = self.u - d };
    }

    /// Returns the square when `us` is white otherwise the vertically mirrored square.
    pub fn relative(self: Square, comptime us: Color) Square {
        return if (us.e == .white) self else .{ .u = self.u ^ 56 };
    }

    pub fn flipped(self: Square) Square {
        return .{ .u = self.u ^ 56 };
    }

    /// Only used during initialization.
    pub fn next(self: Square, dir: Direction) ?Square {
        switch(dir) {
            .north => return if (self.rank() < 7) self.add(8) else null,
            .east => return if (self.file() < 7) self.add(1) else null,
            .south => return if (self.rank() > 0) self.sub(8) else null,
            .west => return if (self.file() > 0) self.sub(1) else null,
            .north_west => return if (self.rank() < 7 and self.file() > 0) self.add(7) else null,
            .north_east => return if (self.rank() < 7 and self.file() < 7) self.add(9) else null,
            .south_east => return if (self.rank() > 0 and self.file() < 7) self.sub(7) else null,
            .south_west => return if (self.rank() > 0 and self.file() > 0) self.sub(9) else null,
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
    pub fn ray(self: Square, dir: Direction) lib.BoundedArray(Square, 8) {
        var result: lib.BoundedArray(Square, 8) = .{};
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
    pub fn rays(self: Square, comptime dirs: []const Direction) lib.BoundedArray(Square, 32) {
        var result: lib.BoundedArray(Square, 32) = .{};
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
        if (str.len < 2) return Square.A1;
        // This math can never crash
        const v: u6 = @truncate((str[1] -| '1') *% 8 +| (str[0] -| 'a'));
        return .{ .u = v };
    }

    pub fn char_of_rank(self: Square) u8 {
        return @as(u8, '1') + self.rank();
    }

    pub fn char_of_file(self: Square) u8 {
        return @as(u8, 'a') + self.file();
    }
};

pub const PieceType = packed union {
    /// Although 3 bits are enough 4 bits is easier for conversions.
    pub const Enum = enum(u4) {
        pawn = 0,
        knight = 1,
        bishop = 2,
        rook = 3,
        queen = 4,
        king = 5,
    };
    /// The enum value.
    e: Enum,
    /// The numeric value.
    u: u4,

    pub const all: [6]PieceType = .{ PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

    pub const PAWN: PieceType = .{ .e = .pawn };
    pub const KNIGHT: PieceType = .{ .e = .knight };
    pub const BISHOP: PieceType = .{ .e = .bishop };
    pub const ROOK: PieceType = .{ .e = .rook };
    pub const QUEEN: PieceType = .{ .e = .queen };
    pub const KING: PieceType = .{ .e = .king };

    pub fn idx(self: PieceType) usize {
        return self.u;
    }

    pub fn value(self: PieceType) Value {
        return piece_values[self.u];
    }

    /// Returns the material code.
    pub fn material(self: PieceType) Value {
        return piece_material_values[self.u];
    }

    pub fn to_char(self: PieceType) u8 {
        return switch(self.e) {
            .pawn => 0,
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
        };
    }
};

pub const Piece = packed union {
    /// To have convenient array indexing the values are just sequential.
    pub const Enum = enum(u4) {
        w_pawn = 0,
        w_knight = 1,
        w_bishop = 2,
        w_rook = 3,
        w_queen = 4,
        w_king = 5,

        b_pawn = 6,
        b_knight = 7,
        b_bishop = 8,
        b_rook = 9,
        b_queen = 10,
        b_king = 11,

        no_piece = 12,
    };
    /// The enum value.
    e: Enum,
    /// The numeric value.
    u: u4,

    /// All valid pieces.
    pub const all: [12]Piece = .{
        W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
        B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    };

    pub const W_PAWN   : Piece = .{ .e = .w_pawn };
    pub const W_KNIGHT : Piece = .{ .e = .w_knight };
    pub const W_BISHOP : Piece = .{ .e = .w_bishop };
    pub const W_ROOK   : Piece = .{ .e = .w_rook };
    pub const W_QUEEN  : Piece = .{ .e = .w_queen };
    pub const W_KING   : Piece = .{ .e = .w_king };

    pub const B_PAWN   : Piece = .{ .e = .b_pawn };
    pub const B_KNIGHT : Piece = .{ .e = .b_knight };
    pub const B_BISHOP : Piece = .{ .e = .b_bishop };
    pub const B_ROOK   : Piece = .{ .e = .b_rook };
    pub const B_QUEEN  : Piece = .{ .e = .b_queen };
    pub const B_KING   : Piece = .{ .e = .b_king };

    pub const NO_PIECE : Piece = .{ .e = .no_piece };

    pub fn create(pt: PieceType, side: Color) Piece {
        return if (side.e == .white) .{ .u = pt.u } else .{ .u = pt.u + 6 };
    }

    pub fn value(self: Piece) Value {
        return piece_values[self.u];
    }

    pub fn material(self: Piece) Value {
        return piece_material_values[self.u];
    }

    pub fn from_usize(u: usize) Piece {
        if (comptime lib.is_paranoid) {
            assert(u <= 11);
        }
        return .{ .u = @truncate(u)};
    }

    pub fn idx(self: Piece) usize {
        return self.u;
    }

    // pub fn bitcast(self: Piece) u4 {
    //     return @bitCast(self);
    // }

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
        return if (self.u < 6) Color.WHITE else Color.BLACK;
    }

    /// Don't call for no_piece.
    pub fn piecetype(self: Piece) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.e != .no_piece);
        }
        return if (self.u < 6) .{ .u = self.u } else .{.u = self. u - 6 };
    }

    pub fn opp(self: Piece) Piece {
        if (self.is_empty()) return Piece.NO_PIECE;
        return if (self.color().e == .white ) .{.u = self.u + 6} else .{ .u = self.u - 6 };
    }

    pub fn is_pawn(self: Piece) bool {
        return self.piecetype().e == .pawn;
    }

    // pub fn is_king(self: Piece) bool {
    //     return self.piecetype().e == .king;
    // }

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
        // TODO: make ascii lookup table?
        return switch(char) {
            'P' => W_PAWN,
            'N' => W_KNIGHT,
            'B' => W_BISHOP,
            'R' => W_ROOK,
            'Q' => W_QUEEN,
            'K' => W_KING,
            'p' => B_PAWN,
            'n' => B_KNIGHT,
            'b' => B_BISHOP,
            'r' => B_ROOK,
            'q' => B_QUEEN,
            'k' => B_KING,
            else => ParsingError.InvalidFenPiece,
        };
    }
};

pub const Move = packed struct(u16) {
    // bit 2 = promotion, bit 3 = capture.
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

    /// 6 bits.
    from: Square = .zero,
    /// 6 bits.
    to: Square = .zero,
    /// Detail flags.
    flags: u4 = 0,

    pub const empty: Move = .{};
    pub const nullmove: Move = @bitCast(@as(u16, 0xffff));

    pub fn create(from: Square, to: Square, flags: u4) Move {
        return .{ .from = from, .to = to, .flags = flags };
    }

    pub fn bitcast(self: Move) u16 {
        return @bitCast(self);
    }

    pub fn is_empty(self: Move) bool {
        return self.bitcast() == 0;
    }

    pub fn is_null_move(self: Move) bool {
        return self.bitcast() == 0xffff;
    }

    pub fn is_capture(self: Move) bool {
        return self.flags & 0b1000 != 0;
    }

    pub fn is_quiet(self: Move) bool {
        return self.flags & 0b1100 == 0; // no capture, no promotion.
    }

    pub fn is_promotion(self: Move) bool {
        return self.flags & 0b0100 != 0;
    }

    pub fn is_promotion_capture(self: Move) bool {
        return self.flags & 0b1100 != 0;
    }

    pub fn is_ep(self: Move) bool {
        return self.flags == ep;
    }

    pub fn is_castle(self: Move) bool {
        return self.flags == castle_short or self.flags == castle_long;
    }

    pub fn flipped(self: Move) Move {
        if (self.is_empty()) return self;
        return. { .from = self.from.flipped(), .to = self.to.flipped(), .flags = self.flags };
    }

    /// Only valid when we are a promotion.
    pub fn promoted_to(self: Move) PieceType {
        if (comptime lib.is_paranoid) {
            assert(self.is_promotion());
        }
        //return .{ .u = @truncate((self.flags & 0b0111) - 2) };
        return .{ .u = (self.flags & 0b0111) - 3 };
    }

    // /// UCI string
    // pub fn to_string(self: Move) lib.BoundedArray(u8, 5) {
    //     var result: lib.BoundedArray(u8, 5) = .{};
    //     const from: Square = self.from;
    //     var to: Square = self.to;

    //     if (self.type == .castle) {
    //         const color: Color = if (to.u < 8) Color.WHITE else Color.BLACK;
    //         const castletype: CastleType = self.info.castletype;
    //         // Change target square. We decode castling as "king takes rook".
    //         to = position.king_castle_destination_squares[color.u][castletype.u];
    //     }

    //     result.print_assume_capacity("{t}", .{ from.e});
    //     result.print_assume_capacity("{t}", .{ to.e});

    //     if (self.type == .promotion) {
    //         const ch: u8 = self.info.prom.to_uci_char();
    //         result.print_assume_capacity("{u}", .{ ch });
    //     }
    //     return result;
    // }

    // Zig-format for UCI move output (e2e4).
    pub fn format(self: Move, writer: *std.io.Writer) std.io.Writer.Error!void {
        if (self.is_empty()) {
            try writer.print("0000", .{});
            return;
        }
        const from: Square = self.from;
        var to: Square = self.to;

        if (self.flags == Move.castle_short) {
            const color: Color = if (to.u < 8) Color.WHITE else Color.BLACK;
            to = position.king_castle_destination_squares[color.u][CastleType.SHORT.u];
        }
        else if (self.flags == Move.castle_long) {
            const color: Color = if (to.u < 8) Color.WHITE else Color.BLACK;
            to = position.king_castle_destination_squares[color.u][CastleType.LONG.u];
        }

        try writer.print("{t}{t}", .{ from.e, to.e });

        if (self.is_promotion()) {
            const prom: PieceType = self.promoted_to();
            const ch: u8 = "?nbrq?"[prom.u];
            try writer.print("{u}", .{ ch });
        }
    }
};

pub const GamePhase = enum { Opening, Midgame, Endgame };

pub const ParsingError = error {
    /// Garbage piece inside fen string.
    InvalidFenPiece,
    /// Garbage or move not found.
    IllegalMove,
    /// Garbage promotion character.
    InvalidPromotionChar,
};

////////////////////////////////////////////////////////////////
/// Constants
////////////////////////////////////////////////////////////////

pub const max_game_length: usize = 1024;
pub const max_move_count: usize = 224;
pub const max_search_depth: u8 = 128;
pub const max_threads: u16 = 32;

pub const infinity: Value = 32000;
pub const mate: Value = 30000;
pub const mate_threshold = mate - 256;
pub const stalemate: Value = 0;
pub const draw: Value = 0;

pub const value_pawn: Value = 100;
pub const value_knight: Value = 317; // 305;
pub const value_bishop: Value = 333;
pub const value_rook: Value = 510; // 474;//// 463; // was: 563
pub const value_queen: Value = 950;
pub const value_king: Value = 0;

// Values used in Position stolen from Stockfish.
pub const material_pawn: Value = 126;
pub const material_knight: Value = 781;
pub const material_bishop: Value = 825;
pub const material_rook: Value = 1276;
pub const material_queen: Value = 2538;
pub const material_king: Value = 0;

/// The total material value in the starting position including pawns
pub const max_material_value: Value = 18620;

/// The threshold value for piece square tables.
pub const max_material_without_pawns: Value = 16604;

pub const opening_threshold: Value = 16604;
pub const midgame_threshold: Value = 15258;
pub const endgame_threshold: Value = 3915;

const piece_values: [12]Value = .{
    value_pawn, value_knight, value_bishop, value_rook, value_queen, value_king,
    value_pawn, value_knight, value_bishop, value_rook, value_queen, value_king,
};

const piece_material_values: [12]Value = .{
    material_pawn, material_knight, material_bishop, material_rook, material_queen, material_king,
    material_pawn, material_knight, material_bishop, material_rook, material_queen, material_king,
};

