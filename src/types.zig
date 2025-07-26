const std = @import("std");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const lib = @import("lib.zig");

const assert = std.debug.assert;

/// Used for evaluation.
pub const Value = i16;
/// Used for evaluation.
pub const Float = f32;

pub const Orientation = enum(u2)
{
    horizontal,
    vertical,
    diagmain,
    diaganti,
};

pub const Direction = enum(u3)
{
    pub const all: [8]Direction = .{.north, .east, .south, .west, .north_west, .north_east, .south_east, .south_west};

    north,
    east,
    south,
    west,
    north_west,
    north_east,
    south_east,
    south_west,

    pub fn to_i8(self: Direction) i8
    {
        return @intFromEnum(self);
    }

    pub fn relative(self: Direction, comptime color: Color) Direction
    {
        if (color.e == .white) return self;

        return switch(self)
        {
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

    pub fn to_orientation(self: Direction) Orientation
    {
        return switch(self)
        {
            .north, .south => .vertical,
            .east, .west => .horizontal,
            .north_west, .south_east => .diagmain,
            .north_east, .south_west => .diaganti,
        };
    }
};

pub const CastleType = enum(u1)
{
    pub const all: [2]CastleType = .{.short, .long };

    short, long,

    pub fn idx(self: CastleType) usize
    {
        return @intFromEnum(self);
    }
};

pub const Color = packed union
{
    pub const WHITE: Color = .{ .e = .white };
    pub const BLACK: Color = .{ .e = .black };

    const Enum = enum(u1) { white, black };

    /// The enum value
    e: Enum,
    /// The numeric value
    u: u1,

    pub fn opp(self: Color) Color
    {
        return .{ .u  = self.u ^ 1 };
    }

    pub fn idx(self: Color) usize
    {
        return self.u;
    }
};

pub const Square = packed union
{
    pub const all: [64]Square = get_all();
    pub const zero: Square = A1;
    pub const no_ep: Square = A1;

    pub const A1: Square = .{ .e = .a1 };
    pub const B1: Square = .{ .e = .b1 };
    pub const C1: Square = .{ .e = .c1 };
    pub const D1: Square = .{ .e = .d1 };
    pub const E1: Square = .{ .e = .e1 };
    pub const F1: Square = .{ .e = .f1 };
    pub const G1: Square = .{ .e = .g1 };
    pub const H1: Square = .{ .e = .h1 };
    pub const A2: Square = .{ .e = .a2 };
    pub const B2: Square = .{ .e = .b2 };
    pub const C2: Square = .{ .e = .c2 };
    pub const D2: Square = .{ .e = .d2 };
    pub const E2: Square = .{ .e = .e2 };
    pub const F2: Square = .{ .e = .f2 };
    pub const G2: Square = .{ .e = .g2 };
    pub const H2: Square = .{ .e = .h2 };
    pub const A3: Square = .{ .e = .a3 };
    pub const B3: Square = .{ .e = .b3 };
    pub const C3: Square = .{ .e = .c3 };
    pub const D3: Square = .{ .e = .d3 };
    pub const E3: Square = .{ .e = .e3 };
    pub const F3: Square = .{ .e = .f3 };
    pub const G3: Square = .{ .e = .g3 };
    pub const H3: Square = .{ .e = .h3 };
    pub const A4: Square = .{ .e = .a4 };
    pub const B4: Square = .{ .e = .b4 };
    pub const C4: Square = .{ .e = .c4 };
    pub const D4: Square = .{ .e = .d4 };
    pub const E4: Square = .{ .e = .e4 };
    pub const F4: Square = .{ .e = .f4 };
    pub const G4: Square = .{ .e = .g4 };
    pub const H4: Square = .{ .e = .h4 };
    pub const A5: Square = .{ .e = .a5 };
    pub const B5: Square = .{ .e = .b5 };
    pub const C5: Square = .{ .e = .c5 };
    pub const D5: Square = .{ .e = .d5 };
    pub const E5: Square = .{ .e = .e5 };
    pub const F5: Square = .{ .e = .f5 };
    pub const G5: Square = .{ .e = .g5 };
    pub const H5: Square = .{ .e = .h5 };
    pub const A6: Square = .{ .e = .a6 };
    pub const B6: Square = .{ .e = .b6 };
    pub const C6: Square = .{ .e = .c6 };
    pub const D6: Square = .{ .e = .d6 };
    pub const E6: Square = .{ .e = .e6 };
    pub const F6: Square = .{ .e = .f6 };
    pub const G6: Square = .{ .e = .g6 };
    pub const H6: Square = .{ .e = .h6 };
    pub const A7: Square = .{ .e = .a7 };
    pub const B7: Square = .{ .e = .b7 };
    pub const C7: Square = .{ .e = .c7 };
    pub const D7: Square = .{ .e = .d7 };
    pub const E7: Square = .{ .e = .e7 };
    pub const F7: Square = .{ .e = .f7 };
    pub const G7: Square = .{ .e = .g7 };
    pub const H7: Square = .{ .e = .h7 };
    pub const A8: Square = .{ .e = .a8 };
    pub const B8: Square = .{ .e = .b8 };
    pub const C8: Square = .{ .e = .c8 };
    pub const D8: Square = .{ .e = .d8 };
    pub const E8: Square = .{ .e = .e8 };
    pub const F8: Square = .{ .e = .f8 };
    pub const G8: Square = .{ .e = .g8 };
    pub const H8: Square = .{ .e = .h8 };

    pub const Enum = enum(u6)
    {
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

    fn get_all() [64]Square
    {
        var out: [64]Square = undefined;
        for (&out, 0..) |*q, square|
        {
            q.* = .{ .u = @truncate(square) };
        }
        return out;
    }

    pub fn from(index: u6) Square
    {
        return .{ .u = index };
    }

    pub fn from_usize(index: usize) Square
    {
        assert(index < 64);
        return .{ .u = @truncate(index) };
    }

    pub fn idx(self: Square) usize
    {
        return self.u;
    }

    /// rank == y, file == x.
    pub fn from_rank_file(r: u3, f: u3) Square
    {
        const v = @as(u6, r) * 8 + f;
        return .from(v);
    }

    /// file == x.
    pub fn file(self: Square) u3
    {
        return @truncate(self.u & 7);
    }

    /// rank == y.
    pub fn rank(self: Square) u3
    {
        return @truncate(self.u >> 3);
    }

    pub fn to_bitboard(self: Square) u64
    {
        return bitboards.bb_a1 << self.u;
        //  @as(u64, 1)
        //return bitboards.square_bitboards[self.u];
    }

    pub fn add(self: Square, d: u6) Square
    {
        return .{ .u = self.u + d };
    }

    pub fn sub(self: Square, d: u6) Square
    {
        return .{ .u = self.u - d };
    }

    /// Returns the square when `us` is white otherwise the vertically mirrored square.
    pub fn relative(self: Square, comptime us: Color) Square
    {
        return if (us.e == .white) self else .{ .u = self.u ^ 56 };
    }

    /// Only used during initialization.
    pub fn next(self: Square, dir: Direction) ?Square
    {
        switch(dir)
        {
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

    /// Only used during initialization.
    pub fn next_twice(self: Square, dir1: Direction, dir2: Direction) ?Square
    {
        if (self.next(dir1)) |a|
        {
            if (a.next(dir2)) |b| return b;
        }
        return null;
    }

    /// Only used during initialization.\
    /// Returns a ray of squares in the direction `dir`.
    /// * not including self.
    pub fn ray(self: Square, dir: Direction) std.BoundedArray(Square, 8)
    {
        var result: std.BoundedArray(Square, 8) = .{};
        var run: Square = self;
        while (true)
        {
            if (run.next(dir)) |n|
            {
                result.appendAssumeCapacity(n);
                run = n;
            }
            else break;
        }
        return result;
    }

    /// Only used during initialization.\
    /// Returns all rays for a range of directions.
    /// * not including self.
    pub fn rays(self: Square, comptime dirs: []const Direction) std.BoundedArray(Square, 32)
    {
        var result: std.BoundedArray(Square, 32) = .{};
        for (dirs) |d|
        {
            result.appendSliceAssumeCapacity(self.ray(d).slice());
        }
        return result;
    }

    /// Debug only.
    pub fn to_string(self: Square) []const u8
    {
        return @tagName(self.e);
    }

    pub fn char_of_rank(self: Square) u8
    {
        return @as(u8, '1') + self.rank();
    }
    pub fn char_of_file(self: Square) u8
    {
        return @as(u8, 'a') + self.file();
    }
};

pub const PieceType = packed union
{
    pub const NO_PIECETYPE: PieceType = .{ .e = .no_piecetype };
    pub const PAWN: PieceType = .{ .e = .pawn };
    pub const KNIGHT: PieceType = .{ .e = .knight };
    pub const BISHOP: PieceType = .{ .e = .bishop };
    pub const ROOK: PieceType = .{ .e = .rook };
    pub const QUEEN: PieceType = .{ .e = .queen };
    pub const KING: PieceType = .{ .e = .king };

    pub const Enum = enum(u3)
    {
        no_piecetype = 0,
        pawn = 1,
        knight = 2,
        bishop = 3,
        rook = 4,
        queen = 5,
        king = 6,
    };

    /// The enum value.
    e: Enum,
    /// The numeric value.
    u: u3,

    pub fn idx(self: PieceType) usize
    {
        return self.u;
    }

    pub fn value(self: PieceType) Value
    {
        return piece_values[self.u];
    }

    pub fn bitmask(self: PieceType) u6
    {
        if (comptime lib.is_paranoid) assert(self.u != 0);
        return @as(u6, 1) << (self.u - 1);
    }

    /// Returns the material code.
    pub fn material(self: PieceType) Value
    {
        return piece_material_values[self.u];
    }

    pub fn to_char(self: PieceType) u8
    {
        return switch(self.e)
        {
            .no_piecetype, .pawn => 0,
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
        };
    }
};

pub const Piece = packed union
{
    // Piece values are so that bit 3 indicates black.
    pub const NO_PIECE : Piece = .{ .e = .no_piece };
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

    pub const Enum = enum(u4)
    {
        no_piece = 0,
        w_pawn = 1,
        w_knight = 2,
        w_bishop = 3,
        w_rook = 4,
        w_queen = 5,
        w_king = 6,

        b_pawn = 9,
        b_knight = 10,
        b_bishop = 11,
        b_rook = 12,
        b_queen = 13,
        b_king = 14,
    };

    /// The enum value.
    e: Enum,
    /// The numeric value.
    u: u4,
    /// The piece type nicely matches the bits. Probably this trick will not be possible anymore in Zig 0.15+
    piecetype: PieceType,

    pub fn make(pt: PieceType, side: Color) Piece
    {
        const p: u4 = pt.u;
        const c: u4 = side.u;
        return .{ .u = p | c << 3 };
    }

    pub fn create_pawn(us: Color) Piece
    {
        return make(PieceType.PAWN, us);
    }

    pub fn create_rook(us: Color) Piece
    {
        return make(PieceType.ROOK, us);
    }

    pub fn create_king(us: Color) Piece
    {
        return make(PieceType.KING, us);
    }

    pub fn value(self: Piece) Value
    {
        return piece_values[self.u];
    }

    pub fn material(self: Piece) Value
    {
        return piece_material_values[self.u];
    }

    // pub fn bitcast(self: Piece) u4
    // {
    //     return @bitCast(self);
    // }

    pub fn from_usize(u: usize) Piece
    {
        if (comptime lib.is_paranoid) assert(u <= 14 and u != 7 and u != 8);
        return .{ .u = @truncate(u)};
    }

    pub fn idx(self: Piece) usize
    {
        return self.u;
    }

    pub fn is_empty(self: Piece) bool
    {
        return self.u == 0;
        //return self.bitcast() == 0;
    }

    pub fn bitcast(self: Piece) u4
    {
        return @bitCast(self);
    }

    pub fn is_piece(self: Piece) bool
    {
        return self.u != 0;
    }

    pub fn color(self: Piece) Color
    {
        return .{ .u = @truncate(self.u >> 3) };
        //if (self.u > 8) return .{ .e = .black } else .{ .e = .white }; // TODO: optimize??
    }

    // pub fn get_piece_type(self: Piece) PieceType
    // {
    //     // Filter out the color
    //     const value: u3 = @truncate(self.u & 0b111);
    //     return .{ .u = value };
    // }

    pub fn opp(self: Piece) Piece
    {
        return .{ .u = self.u ^ 8};
    }

    pub fn is_pawn(self: Piece) bool
    {
        return self.piecetype.e == .pawn;
    }

    pub fn to_print_char(self: Piece) u8
    {
        var ch: u8 = switch(self.piecetype.e)
        {
            .pawn => 'P',
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
            else => '?'
        };

        if (self.color().e == .black) ch = std.ascii.toLower(ch);
        return ch;
    }

    pub fn to_fen_char(self: Piece) u8
    {
        return switch (self.e)
        {
            .w_pawn   => 'P' ,
            .w_knight => 'N',
            .w_bishop => 'B',
            .w_rook   => 'R',
            .w_queen  => 'Q',
            .w_king   => 'K',
            .b_pawn   => 'p' ,
            .b_knight => 'n' ,
            .b_bishop => 'b' ,
            .b_rook   => 'r' ,
            .b_queen  => 'q' ,
            .b_king   => 'k' ,
            else => unreachable,
        };
    }

};

pub const MoveType = enum (u2)
{
    normal = 0,
    promotion = 1,
    enpassant = 2,
    castle = 3,
};

pub const Move = packed struct(u16)
{
    pub const empty: Move = .{ .from = .zero, .to = .zero, .prom = .knight, .movetype = .normal };

    /// Encoded promotion piece
    pub const Prom = enum(u2)
    {
        pub const no_prom: Prom = .knight; // we do not have more bits :)

        knight,
        bishop,
        rook,
        queen,

        pub fn bitcast(self: Prom) u2
        {
            return @intFromEnum(self);
        }

        pub fn to_piecetype(self: Prom) PieceType
        {
            const v: u3 = self.bitcast();
            return PieceType{ .u = v + 2};
        }

        pub fn to_piece(self: Prom, us: Color) Piece
        {
            return Piece.make(self.to_piecetype(), us);
        }
    };

    /// 6 bits
    from: Square,
    /// 6 bits
    to: Square,
    /// 2 bits
    prom: Prom,
    /// 2 bits
    movetype: MoveType,

    pub fn create(from: Square, to: Square) Move
    {
        return .{ .from = from, .to = to, .prom = .no_prom, .movetype = .normal};
    }

    pub fn create_promotion(from: Square, to: Square, prom: Prom) Move
    {
        return .{ .from = from, .to = to, .prom = prom, .movetype = .promotion};
    }

    pub fn create_enpassant(from: Square, to: Square) Move
    {
        return .{ .from = from, .to = to, .prom = .no_prom, .movetype = .enpassant};
    }

    pub fn create_castle(from: Square, to: Square) Move
    {
        return .{ .from = from, .to = to, .prom = .no_prom, .movetype = .castle};
    }

    /// Only valid if castling
    pub fn castle_type(self: Move) CastleType
    {
        return if (self.to.u > self.from.u) .short else .long;
    }

    pub fn promotion_piece(self: Move) PieceType
    {
        return self.prom.to_piecetype();
    }

    /// UCI string
    pub fn to_string(self: Move) std.BoundedArray(u8, 8)
    {
        var result: std.BoundedArray(u8, 8) = .{};
        const from: Square = self.from;
        var to: Square = self.to;

        if (self.movetype == .castle)
        {
            const castletype: CastleType = self.castle_type();
            const color: Color = if (to.rank() == 0) Color.WHITE else Color.BLACK;
            // Change target square. We decode castling as "king takes rook"
            to = funcs.king_castle_to_square(color, castletype);
        }

        result.appendSliceAssumeCapacity(@tagName(from.e));
        result.appendSliceAssumeCapacity(@tagName(to.e));

        if (self.movetype == .promotion)
        {
            result.appendAssumeCapacity('=');
            const ch: u8 = self.prom.to_piecetype().to_char();
            result.appendAssumeCapacity(ch);
        }
        return result;
    }

};

pub const max_move_count: usize = 224;
pub const max_search_depth: u8 = 128;

pub const infinity: Value = 32000;
pub const mate: Value = 28000;
pub const mate_threshold = mate - 256;
pub const stalemate: Value = 0;
pub const draw: Value = 0;

const value_pawn: Value = 100;
const value_knight: Value = 300;
const value_bishop: Value = 300;
const value_rook: Value = 500;
const value_queen: Value = 900;

// Values used in Position stolen from Stockfish.
pub const material_pawn: Value = 126;
pub const material_knight: Value = 781;
pub const material_bishop: Value = 825;
pub const material_rook: Value = 1276;
pub const material_queen: Value = 2538;

// startvalue: 18620 WITH pawns
// startvalue: 16604 WITHOUT pawns
pub const max_material_value_threshold: Value = 18620;
//pub const MIDGAME_THRESHOLD: i16 = 15258;
//pub const ENDGAME_THRESHOLD: i16 = 3915;


const piece_values: [15]Value =
.{
    0, // no_piece
    value_pawn, value_knight, value_bishop, value_rook, value_queen, 0,
    0, 0, // empty
    value_pawn, value_knight, value_bishop, value_rook, value_queen, 0,
};

const piece_material_values: [15]Value =
.{
    0, // no_piece
    material_pawn, material_knight, material_bishop, material_rook, material_queen, 0,
    0, 0, // empty
    material_pawn, material_knight, material_bishop, material_rook, material_queen, 0,
};



