const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const data = @import("data.zig");
const masks = @import("masks.zig");
const position = @import("position.zig");

const Value = types.Value;
const Float = types.Float;
const Color = types.Color;
const Square = types.Square;
const CastleType = types.CastleType;
const Position = position.Position;

const assert = std.debug.assert;

/// Enum for shifting pawn moves.
pub const PawnShift = enum(u2)
{
    up, northwest, northeast,
};

pub fn king_castle_to_square(us: Color, dir: CastleType) Square
{
    const squares: [2][2]Square = .{ .{.G1, .C1}, .{.G8, .C8} };
    return squares[us.u][dir.idx()];
}

pub fn rook_castle_to_square(us: Color, dir: CastleType) Square
{
    const squares: [2][2]Square = .{ .{.F1, .D1}, .{.F8, .D8} };
    return squares[us.u][dir.idx()];
}

pub fn relative_rank_7_bitboard(comptime us: Color) u64
{
    return if (us.e == .white) bitboards.bb_rank_7 else bitboards.bb_rank_2;
}

pub fn relative_rank_3_bitboard(comptime us: Color) u64
{
    return if (us.e == .white) bitboards.bb_rank_3 else bitboards.bb_rank_6;
}

pub fn relative_south_east(comptime us: Color) i7
{
    return if (us.e == .white) -7 else 7;
}

pub fn relative_south_west(comptime us: Color) i7
{
    return if (us.e == .white) -9 else 9;
}

pub fn pawns_shift(pawns: u64, comptime us: Color, comptime shift: PawnShift) u64
{
    // TODO: check symmetry for movgen.
    switch(us.e)
    {
        .white =>
        {
           return switch(shift)
            {
                .up => pawns << 8,
                .northwest => (pawns & ~bitboards.bb_file_a) << 7,
                .northeast => (pawns & ~bitboards.bb_file_h) << 9,
            };
        },
        .black =>
        {
            return switch(shift)
            {
                .up => pawns >> 8,
                .northwest => (pawns & ~bitboards.bb_file_h) >> 7,
                .northeast => (pawns & ~bitboards.bb_file_a) >> 9,
            };
        }
    }
}

pub fn is_supported_by_pawn(pos: *const Position, comptime us: Color, sq: Square) bool
{
    const them = comptime us.opp();
    return data.get_pawn_attacks(sq, them) & pos.pawns(us) != 0; // using inversion trick
}

pub fn is_passed_pawn(pos: *const Position, comptime us: Color, sq: Square) bool
{
    return switch (us.e)
    {
        .white => masks.get_passed_pawn_mask(us, sq) & pos.all_pawns() == 0,
        .black => masks.get_passed_pawn_mask(us, sq) & pos.all_pawns() == 0,
    };
}

// fn mirrorHorizontally(bits: u64) u64 {
//     var x = bits;
//     x = ((x & 0x5555555555555555) << 1) | ((x >> 1) & 0x5555555555555555);
//     x = ((x & 0x3333333333333333) << 2) | ((x >> 2) & 0x3333333333333333);
//     x = ((x & 0x0F0F0F0F0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0F);
//     return x;
// }

pub fn mirror_vertically(u: u64) u64
{
    var x = u;
    x = ((x & 0x00000000000000ff) << 56) |
        ((x & 0x000000000000ff00) << 40) |
        ((x & 0x0000000000ff0000) << 24) |
        ((x & 0x00000000ff000000) << 8)  |
        ((x & 0x000000ff00000000) >> 8)  |
        ((x & 0x0000ff0000000000) >> 24) |
        ((x & 0x00ff000000000000) >> 40) |
        ((x & 0xff00000000000000) >> 56);
    return x;
}

pub fn contains_square(bitboard: u64, sq: Square) bool
{
    //return bitboard & sq.to_bitboard() != 0;
    return test_bit_64(bitboard, sq.u);
}

pub fn first_square_or_null(bitboard: u64) ?Square
{
    if (bitboard == 0) return null;
    const lsb: u6 = @truncate(@ctz(bitboard));
    return Square.from(lsb);
}

/// Unsafe lsb
pub fn first_square(bitboard: u64) Square
{
    assert(bitboard != 0);
    const lsb: u6 = @truncate(@ctz(bitboard));
    return Square.from(lsb);
}

/// Unsafe pop lsb and clears that lsb from the bitboard.
pub fn pop_square(bitboard: *u64) Square
{
    assert(bitboard.* != 0);
    defer bitboard.* &= (bitboard.* - 1);
    return first_square(bitboard.*);
}

// EXPERIMENTAL
pub fn pop(bitboard: *u64) ?Square
{
    if (bitboard.* == 0) return null;
    defer bitboard.* &= (bitboard.* - 1);
    return Square.from(@truncate(@ctz(bitboard.*)));
}

pub fn clear_square(bitboard: *u64, sq: Square) void
{
    bitboard.* &= ~sq.to_bitboard();
}


/// Unsafe
pub fn lsb_u64(u: u64) u6
{
    assert(u != 0);
    return @truncate(@ctz(u));
}

pub fn test_bit_u8(u: u8, bit: u3) bool
{
    const one: u8 = @as(u8, 1) << bit;
    return u & one != 0;
}

pub fn test_bit_64(u: u64, bit: u6) bool
{
    const one: u64 = @as(u64, 1) << bit;
    return u & one != 0;
}


pub fn movenumber_to_ply(movenr: u16, to_move: Color) u16
{
    return @max(2 * (movenr - 1), 0) + (to_move.u);
}

pub fn ptr_add(T: type, ptr: *T, delta: comptime_int) *T
{
    return @ptrFromInt(@intFromPtr(ptr) + @sizeOf(T) * delta);
}

pub fn ptr_sub(T: type, ptr: *T, delta: comptime_int) *T
{
    return @ptrFromInt(@intFromPtr(ptr) - @sizeOf(T) * delta);
}

/// ### Utility function.
/// Calculates something per second.
pub fn nps(count: usize, elapsed_nanoseconds: u64) u64
{
    if (elapsed_nanoseconds == 0) return 0;
    const a: f64 = @floatFromInt(count);
    const b: f64 = @floatFromInt(elapsed_nanoseconds);
    const s: f64 =  (a * 1_000_000_000.0) / b;
    return @intFromFloat(s);
}

pub fn start_timer() std.time.Timer
{
    return std.time.Timer.start() catch @panic("timer issue");
}

















pub fn print_bitboard(bb: u64) void
{
    var y: u3 = 7;
    while (true)
    {
        var x: u3 = 0;
        while (true)
        {
            const square: Square = .from_rank_file(y, x);
            const b: u64 = square.to_bitboard();
            if (bb & b == 0) std.debug.print(". ", .{}) else std.debug.print("1 ", .{});

            if (x == 7) break;
            x += 1;
        }
        std.debug.print("\n", .{});
        if (y == 0) break;
        y -= 1;
    }
    std.debug.print("\n", .{});

}

pub fn print_bits(u: u8) void
{
    const x: std.bit_set.IntegerBitSet(8) = .{.mask = u};
    for (0..8) |bit|
    {
        if (x.isSet(bit)) std.debug.print("1", .{}) else std.debug.print(".", .{});
    }
    std.debug.print("\n", .{});
}

pub fn float(i: Value) f32
{
    return @floatFromInt(i);
}

pub fn int(f: Float) Value
{
    return @intFromFloat(f);
}
