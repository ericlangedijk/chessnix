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

// fn abs_diff(a: usize, b: usize) usize {
//     return @max(a, b) - @min(a, b);
// }

/// Note that these are stored in SquarePair
pub inline fn square_distance(a: Square, b: Square) u3 {
    //lib.comptime_only();
    const ar: i32 = a.coord.rank;
    const br: i32 = b.coord.rank;
    const af: i32 = a.coord.file;
    const bf: i32 = b.coord.file;
    const d: u32 = @max(@abs(ar - br), @abs(af - bf));
    return @truncate(@abs(d));
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

pub fn is_relative_rank_456(us: Color, rank: u3) bool {
    return if (us.e == .white)
        rank >= bitboards.rank_4 and rank <= bitboards.rank_6
    else
        rank >= bitboards.rank_3 and rank <= bitboards.rank_5;
}

/// Relative rank 4,5,6
pub fn outpost(comptime us: Color) u64 {
    return if (us.e == .white) bitboards.bb_rank_4 | bitboards.bb_rank_5 | bitboards.bb_rank_6 else bitboards.bb_rank_3 | bitboards.bb_rank_4 | bitboards.bb_rank_5;
}

pub fn relative_side_bitboard(comptime us: Color) u64 {
    return if (us.e == .white) bitboards.bb_white_side else bitboards.bb_black_side;
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

pub fn shift_bitboard(u: u64, comptime dir: Direction) u64 {
    return switch (dir) {
        .north      => (u & ~bitboards.bb_rank_8) << 8,
        .east       => (u & ~bitboards.bb_file_h) << 1,
        .south      => (u & ~bitboards.bb_rank_1) >> 8,
        .west       => (u & ~bitboards.bb_file_a) >> 1,
        .north_west => (u & ~bitboards.bb_file_a) << 7,
        .north_east => (u & ~bitboards.bb_file_h) << 9,
        .south_east => (u & ~bitboards.bb_file_a) >> 7,
        .south_west => (u & ~bitboards.bb_file_a) >> 9,
    };
}

pub fn mirror_vertically(u: u64) u64 {
    return
        ( (u & 0x00000000000000ff) << 56) |
        ( (u & 0x000000000000ff00) << 40) |
        ( (u & 0x0000000000ff0000) << 24) |
        ( (u & 0x00000000ff000000) << 8 ) |
        ( (u & 0x000000ff00000000) >> 8 ) |
        ( (u & 0x0000ff0000000000) >> 24) |
        ( (u & 0x00ff000000000000) >> 40) |
        ( (u & 0xff00000000000000) >> 56);
}

pub fn popcnt(bitboard: u64) u7 {
    return @popCount(bitboard);
}

pub fn contains_square(bitboard: u64, sq: Square) bool {
    return test_bit_64(bitboard, sq.u);
}

/// Unsafe lsb. Assumes bitboard != 0.
pub fn first_square(bitboard: u64) Square {
    if (comptime lib.is_paranoid) {
        assert(bitboard != 0);
    }
    //const lsb: u6 = @truncate(@ctz(bitboard));
    const lsb: u6 = @intCast(@ctz(bitboard));
    return .{ .u = lsb };
}

/// I finally managed to make this even faster than manual popping (intCast is probably the trick instead of truncate).
pub inline fn bitloop(bitboard: *u64) ?Square {
    if (bitboard.* == 0) return null;
    defer bitboard.* &= (bitboard.* - 1);
    return .{ .u = @intCast(@ctz(bitboard.*)) };
}

/// Note: This requires x86-64 and the BMI2 instruction set.
pub fn get_nth_set_bit_or_null(bitboard: u64, n: u6) ?u6 {
    // Return null if we are asking for a bit that doesn't exist
    if (@popCount(bitboard) <= n) return null;

    const nth_bit = @as(u64, 1) << n;

    // PDEP (Parallel Bit Deposit) magic via inline assembly
    const isolated = asm (
        "pdep %[mask], %[val], %[out]"
        : [out] "=r" (-> u64),
        : [val] "r" (nth_bit),
          [mask] "r" (bitboard),
    );

    return @intCast(@ctz(isolated));
}

/// Note: This requires x86-64 and the BMI2 instruction set.
pub fn get_nth_set_bit(bitboard: u64, n: u6) u6 {
    const nth_bit = @as(u64, 1) << n;
    // PDEP (Parallel Bit Deposit) magic via inline assembly
    const isolated = asm (
        "pdep %[mask], %[val], %[out]"
        : [out] "=r" (-> u64),
        : [val] "r" (nth_bit),
          [mask] "r" (bitboard),
    );
    return @intCast(@ctz(isolated));
}

pub fn clear_square(bitboard: *u64, sq: Square) void {
    bitboard.* &= ~sq.to_bitboard();
}

pub fn test_bit_u8(u: u8, bit: u3) bool {
    const one: u8 = @as(u8, 1) << bit;
    return u & one != 0;
}

pub fn test_bit_64(u: u64, bit: u6) bool {
    const one: u64 = @as(u64, 1) << bit;
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

// fn int(comptime T: type, x: anytype) T {
//     return switch (@typeInfo(@TypeOf(x))) {
//         .int, .comptime_int => @intCast(x),
//         .float, .comptime_float => @intFromFloat(x),
//         else => @compileError(std.fmt.comptimePrint("unsupported type {}\n", .{@TypeOf(x)})),
//     };
// }
