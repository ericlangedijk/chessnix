// zig fmt: off

//! Chess related utilities.

const std = @import("std");

const lib = @import("lib.zig");
const utils = @import("utils.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const attacks = @import("attacks.zig");
const position = @import("position.zig");

const Direction = types.Direction;
const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;
const CastleType = types.CastleType;
const Position = position.Position;

const assert = std.debug.assert;
const wtf = lib.wtf;

/// Enum for shifting pawn moves.
pub const PawnShift = enum(u2) { up, northwest, northeast };

/// Note that these are stored in SquarePair
pub inline fn square_distance(a: Square, b: Square) u3 {
    //lib.comptime_only();
    const ar: i32 = a.coord.rank;
    const br: i32 = b.coord.rank;
    const af: i32 = a.coord.file;
    const bf: i32 = b.coord.file;
    const d: u32 = @max(@abs(ar - br), @abs(af - bf));
    return @truncate(@abs(d));
    //return @intCast(@abs(d)); // TODO: truncate or intcast
}

/// Note that these are stored in SquarePair
pub inline fn manhattan_distance(a: Square, b: Square) u8 {
    //lib.comptime_only();
    const rank1: i32 = a.coord.rank;
    const rank2: i32 = b.coord.rank;
    const file1: i32 = a.coord.file;
    const file2: i32 = b.coord.file;
    const rank_distance = @abs (rank2 - rank1);
    const file_distance = @abs (file2 - file1);
    return @intCast(rank_distance + file_distance);
}


/// Relative rank 4,5,6
pub fn outpost(comptime us: Color) u64 {
    return if (us.e == .white) bitboards.bb_rank_4 | bitboards.bb_rank_5 | bitboards.bb_rank_6 else bitboards.bb_rank_3 | bitboards.bb_rank_4 | bitboards.bb_rank_5;
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

pub fn pawns_attacks(pawns: u64, comptime us: Color) u64 {
    return switch(us.e) {
        .white =>  ((pawns & ~bitboards.bb_file_a) << 7) | ((pawns & ~bitboards.bb_file_h) << 9),
        .black =>  ((pawns & ~bitboards.bb_file_h) >> 7) | ((pawns & ~bitboards.bb_file_a) >> 9),
    };
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

pub fn test_bit(u: comptime_int, bit: comptime_int) bool {
    const int_type = @TypeOf(u);
    const one: int_type = @as(int_type, 1) << bit;
    return u & one != 0;
}

/// Assumes movenr > 0. Only used in `Position.set()`.
pub fn movenumber_to_ply(movenr: u16, stm: Color) u16 {
    return @max(2 * (movenr - 1), 0) + (stm.u);
}

pub fn ply_to_movenumber(ply: u16, tomove: Color) u16 {
    return if (ply == 0) 1 else (ply - tomove.u) / 2 + 1;
}

/// Convert "mate in X moves" to an absolute "distance to mate".
/// * `mv` is always the matevalue from the perspective of white: negative -> white loses, positive -> white wins.
pub fn mate_to_dtm(mv: i32, stm: Color) i32 {
    if (mv == 0) return 0;
    const white_wins: bool = mv > 0;
    const white_to_move: bool = stm.e == .white;
    return (if (white_wins) mv * 2 else -mv * 2) - @intFromBool(white_wins == white_to_move);
}

pub fn eql(input: []const u8, comptime line: []const u8) bool {
    return std.mem.eql(u8, input, line);
}

/// Not used.
pub fn ptr_add(T: type, ptr: *const T, comptime delta: comptime_int) *T {
    return @ptrFromInt(@intFromPtr(ptr) + @sizeOf(T) * delta);
}

/// Not used.
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

/// Convert any int to f32.
pub fn float32(i: anytype) f32 {
    return @floatFromInt(i);
}

pub fn float64(i: anytype) f64 {
    return @floatFromInt(i);
}

pub fn percent(max: usize, count: usize) usize {
    if (max == 0) return 0;
    const c: f32 = @floatFromInt(count);
    const m: f32 = @floatFromInt(max);
    return @intFromFloat((c * 100) / m);
}

pub fn permille(max: usize, count: usize) usize {
    if (max == 0) return 0;
    const c: f32 = @floatFromInt(count);
    const m: f32 = @floatFromInt(max);
    return @intFromFloat((c * 1000) / m);
}

/// multiply any int with float.
pub fn fmul(i: anytype, f: f32) @TypeOf(i) {
    return @intFromFloat(float32(i) * f);
}

/// Debug only
pub fn print_bitboard(bb: u64) void {
    var y: u3 = 7;
    while (true) {
        var x: u3 = 0;
        while (true) {
            const square: Square = .from_rank_file(y, x);
            const b: u64 = square.to_bitboard();
            if (bb & b == 0) lib.io.debugprint(". ", .{}) else lib.io.debugprint("1 ", .{});
            if (x == 7) break;
            x += 1;
        }
        lib.io.debugprint("\n", .{});
        if (y == 0) break;
        y -= 1;
    }
    lib.io.debugprint("\n", .{});
}

/// Debug only
pub fn print_bits(u: u8) void {
    const x: std.bit_set.IntegerBitSet(8) = .{.mask = u};
    for (0..8) |bit| {
        if (x.isSet(bit)) lib.io.debugprint("1", .{}) else std.debug.print(".", .{});
    }
    lib.io.debugprint("\n", .{});
}




// fn float(x: anytype) f64 {
//     return switch (@typeInfo(@TypeOf(x))) {
//         .int, .comptime_int => @floatFromInt(x),
//         .float, .comptime_float => @floatCast(x),
//         else => @compileError(std.fmt.comptimePrint("unsupported type {}\n", .{@TypeOf(x)})),
//     };
// }

pub fn int(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .int, .comptime_int => @intCast(x),
        .float, .comptime_float => @intFromFloat(x),
        else => @compileError(std.fmt.comptimePrint("unsupported type {}\n", .{@TypeOf(x)})),
    };
}
