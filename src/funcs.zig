// zig fmt: off

//! Chess related utilities.

const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const attacks = @import("attacks.zig");
const position = @import("position.zig");

const Value = types.Value;
const Float = types.Float;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;
const CastleType = types.CastleType;
const Position = position.Position;

const assert = std.debug.assert;
const wtf = lib.wtf;

/// Enum for shifting pawn moves.
pub const PawnShift = enum(u2) { up, northwest, northeast };

pub fn relative_rank_7_bitboard(us: Color) u64 {
    return if (us.e == .white) bitboards.bb_rank_7 else bitboards.bb_rank_2;
}

pub fn relative_rank_8_bitboard(us: Color) u64 {
    return if (us.e == .white) bitboards.bb_rank_8 else bitboards.bb_rank_1;
}

pub fn relative_rank_3_bitboard(us: Color) u64 {
    return if (us.e == .white) bitboards.bb_rank_3 else bitboards.bb_rank_6;
}

pub fn relative_rank(us: Color, rank: u3) u3 {
    return if (us.e == .white) rank else 7 - rank;
}

pub fn relative_rank_bb(us: Color, rank: u3) u64 {
    return if (us.e == .white) bitboards.rank_bitboards[rank] else bitboards.rank_bitboards[7 - rank];
}

pub fn relative_rank_7(us: Color) u3 {
    return if (us.e == .white) bitboards.rank_7 else bitboards.rank_2;
}

pub fn pawns_shift(pawns: u64, comptime us: Color, comptime shift: PawnShift) u64 {
    switch(us.e) {
        .white => {
           return switch(shift) {
                .up        => pawns << 8,
                .northwest => (pawns & ~bitboards.bb_file_a) << 7,
                .northeast => (pawns & ~bitboards.bb_file_h) << 9,
            };
        },
        .black => {
            return switch(shift) {
                .up        => pawns >> 8,
                .northwest => (pawns & ~bitboards.bb_file_h) >> 7, // southeast
                .northeast => (pawns & ~bitboards.bb_file_a) >> 9, // southwest
            };
        }
    }
}

/// Returns the from square of a moved pawn.
pub fn pawn_from(to: Square, comptime us: Color, comptime shift: PawnShift) Square {
    switch (us.e) {
        .white => {
            return switch(shift) {
                .up        => to.sub(8),
                .northwest => to.sub(7),
                .northeast => to.sub(9),
            };
        },
        .black => {
            return switch(shift) {
                .up        => to.add(8),
                .northwest => to.add(7), // southeast
                .northeast => to.add(9), // southwest
            };
        }
    }
}

// pub fn all_pawns_attacks(pawns: u64, comptime us: Color) u64 {
//     return switch(us.e) {
//         .white => ((pawns & ~bitboards.bb_file_a) << 7) | ((pawns & ~bitboards.bb_file_h) << 9),
//         .black => ((pawns & ~bitboards.bb_file_h) >> 7) | ((pawns & ~bitboards.bb_file_a) >> 9),
//     };
// }

// fn mirrorHorizontally(bits: u64) u64 {
//     var x = bits;
//     x = ((x & 0x5555555555555555) << 1) | ((x >> 1) & 0x5555555555555555);
//     x = ((x & 0x3333333333333333) << 2) | ((x >> 2) & 0x3333333333333333);
//     x = ((x & 0x0F0F0F0F0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0F);
//     return x;
// }

pub fn mirror_vertically(u: u64) u64 {
    var x = u;
    x = ( (x << 56)) |
        ( (x & 0x000000000000ff00) << 40) |
        ( (x & 0x0000000000ff0000) << 24) |
        ( (x & 0x00000000ff000000) << 8 ) |
        ( (x & 0x000000ff00000000) >> 8 ) |
        ( (x & 0x0000ff0000000000) >> 24) |
        ( (x & 0x00ff000000000000) >> 40) |
        ( (x >> 56));
    return x;
}

pub fn popcnt(bitboard: u64) u7 {
    return @popCount(bitboard);
}

pub fn popcnt_v(bitboard: u64) Value {
    return @popCount(bitboard);
}

pub fn contains_square(bitboard: u64, sq: Square) bool {
    return test_bit_64(bitboard, sq.u);
}

pub fn first_square_or_null(bitboard: u64) ?Square {
    if (bitboard == 0) return null;
    const lsb: u6 = @truncate(@ctz(bitboard));
    return Square.from(lsb);
}

/// Unsafe lsb
pub fn first_square(bitboard: u64) Square {
    if (comptime lib.is_paranoid) assert(bitboard != 0);
    const lsb: u6 = @truncate(@ctz(bitboard));
    return Square.from(lsb);
}

/// Unsafe pop lsb and clears that lsb from the bitboard.
pub fn pop_square(bitboard: *u64) Square {
    if (comptime lib.is_paranoid) assert(bitboard.* != 0);
    defer bitboard.* &= (bitboard.* - 1);
    return first_square(bitboard.*);
}

/// I cannot make this function as fast as a manual loop.
pub fn bit_loop(bitboard: *u64) ?Square {
    if (bitboard.* == 0) return null;
    defer bitboard.* &= (bitboard.* - 1);
    return Square.from(@truncate(@ctz(bitboard.*)));
}

pub fn clear_square(bitboard: *u64, sq: Square) void {
    bitboard.* &= ~sq.to_bitboard(); // TODO: XOR?
}

/// Unsafe
pub fn lsb_u64(u: u64) u6 {
    if (comptime lib.is_paranoid) assert(u != 0);
    return @truncate(@ctz(u));
}

pub fn test_bit_u8(u: u8, bit: u3) bool {
    const one: u8 = @as(u8, 1) << bit;
    return u & one != 0;
}

pub fn test_bit_64(u: u64, bit: u6) bool {
    const one: u64 = @as(u64, 1) << bit;
    return u & one != 0;
}

pub fn movenumber_to_ply(movenr: u16, stm: Color) u16 {
    return @max(2 * (movenr - 1), 0) + (stm.u); // TODO: make safe for underflow.
}

pub fn ply_to_movenumber(ply: u16, tomove: Color) u16 {
    return if (ply == 0) 1 else (ply - tomove.u) / 2 + 1;
}

/// Convert "mate in X moves" to an absolute "distance to mate".
/// * `mv` is always the matevalue from the perspective of white: negative -> white loses, positive -> white wins.
pub fn mate_to_dtm(mv: Value, stm: Color) Value {
    if (mv == 0) return 0;
    const white_wins: bool = mv > 0;
    const white_to_move: bool = stm.e == .white;
    return (if (white_wins) mv * 2 else -mv * 2) - @intFromBool(white_wins == white_to_move);
}

pub fn eql(input: []const u8, comptime line: []const u8) bool {
    return std.mem.eql(u8, input, line);
}

// TODO: smarter. there must be some std.mem function.
pub fn in(input: u8, comptime line: []const u8) bool {
    for (line) |e| { if (e == input) return true; } return false;
}

pub fn ptr_add(T: type, ptr: *const T, comptime delta: comptime_int) *T {
    return @ptrFromInt(@intFromPtr(ptr) + @sizeOf(T) * delta);
}

pub fn ptr_sub(T: type, ptr: *const T, comptime delta: comptime_int) *T {
    return @ptrFromInt(@intFromPtr(ptr) - @sizeOf(T) * delta);
}

/// Calculates something per second.
pub fn nps(count: usize, elapsed_nanoseconds: u64) u64 {
    if (elapsed_nanoseconds == 0) return 0;
    const a: f64 = @floatFromInt(count);
    const b: f64 = @floatFromInt(elapsed_nanoseconds);
    const s: f64 = (a * 1_000_000_000.0) / b;
    return @intFromFloat(s);
}

/// Calculates something in millions per second.
pub fn mnps(count: usize, elapsed_nanoseconds: u64) f64 {
    if (elapsed_nanoseconds == 0) return 0;
    const a: f64 = @floatFromInt(count);
    const b: f64 = @floatFromInt(elapsed_nanoseconds);
    const s: f64 = (a * 1_000.0) / b;
    return s;
}

//
// pub fn percentage(done: u64, total: u64) u64 {
//     if (total == 0) return 0;
//     const a: f64 = @floatFromInt(done);
//     const b: f64 = @floatFromInt(total);
//     const s: f64 = (a * 100) / b;
//     return @intFromFloat(s);
// }

pub fn float(i: Value) f32 {
    return @floatFromInt(i);
}

pub fn int(f: Float) Value {
    return @intFromFloat(f);
}

pub fn mul(i: Value, f: Float) Value {
    return @intFromFloat(float(i) * f);
}

pub fn div(i: Value, d: Value) Value {
    return @divTrunc(i, d);
}

pub fn percent(max: usize, count: usize) usize {
    if (max == 0) return 0;
    const c: f32 = @floatFromInt(count);
    const m: f32 = @floatFromInt(max);
    return @intFromFloat((c * 100) / m);
}

pub fn permille(max: usize, count: usize) usize {
    //assert(max > 0);
    if (max == 0) return 0;
    const c: f32 = @floatFromInt(count);
    const m: f32 = @floatFromInt(max);
    return @intFromFloat((c * 1000) / m);
}

pub fn compress_board(pos: *const Position) [32]u8 {
    var result: [32]u8 = @splat(0);
    for (pos.board, 0..) |piece, index| {
        const u: u8 = piece.u;
        const idx = index / 2;
        switch (index % 2) {
            0 => result[idx] |= u,
            1 => result[idx] |= (u << 4),
            else => unreachable,
        }
    }
    return result;
}

pub fn decompress_board(src: [32]u8) [64]Piece {
    var result: [64]Piece = @splat(Piece.NO_PIECE);

    var sq: u8 = 0;
    for (src) |u| {
        const a: u8 = u & 0b1111;
        const b: u8 = u >> 4;
        result[sq]     = .{ .u = @truncate(a) };
        result[sq + 1] = .{ .u = @truncate(b) };
        sq += 2;
    }
    return result;
}

/// DEBUG
pub fn print_bitboard(bb: u64) void {
    var y: u3 = 7;
    while (true) {
        var x: u3 = 0;
        while (true) {
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

/// DEBUG
pub fn print_bits(u: u8) void {
    const x: std.bit_set.IntegerBitSet(8) = .{.mask = u};
    for (0..8) |bit| {
        if (x.isSet(bit)) std.debug.print("1", .{}) else std.debug.print(".", .{});
    }
    std.debug.print("\n", .{});
}
