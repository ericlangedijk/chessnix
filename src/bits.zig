const std = @import("std");
const lib = @import("lib");

const assert = std.debug.assert;

/// A bitmask for integer up to 64 bits.
/// * no use of usize (like in zig's std), but exact types.
pub fn BitSet(comptime size: u7) type
{
    comptime if (size == 0) @compileError("BitSet size 0 not supported");
    comptime if (size > 64)  @compileError("BitSet size > 64 not supported");

    return packed struct
    {

        const Self = @This();
        pub const zeroes: Self = .{};
        pub const ones: Self = .init(std.math.maxInt(MaskInt));

        /// The integer type used to represent a mask in this bit set. For BitSet(64) this will be a u64.
        pub const MaskInt = std.meta.Int(.unsigned, size);
        /// The integer type used to represent a bit number in this bit set. For BitSet(u64) this will be a u6.
        pub const BitInt = std.math.Log2Int(MaskInt);
        /// The integer type used to represent counting of bits. For BitSet(64) this will be a u7.
        pub const CountInt = std.math.Log2Int(std.meta.Int(.unsigned, size + 1));

        // /// The bit mask, as a single integer
        mask: MaskInt = 0,

        pub fn init(mask: MaskInt) Self
        {
            return .{ .mask = mask };
        }

        pub fn is_set(self: Self, bit: BitInt) bool
        {
            return (self.mask & mask_bit(bit)) != 0;
        }

        pub fn set(self: *Self, bit: BitInt) void
        {
            self.mask |= mask_bit(bit);
        }

        pub fn unset(self: *Self, bit: BitInt) void
        {
            self.mask &= ~mask_bit(bit);
        }

        pub fn count(self: Self) CountInt
        {
            return @popCount(self.mask);
        }

        pub fn lsb_or_null(self: Self) ?BitInt
        {
            return if (self.mask == 0) null else @truncate(@ctz(self.mask));
        }

        /// Unsafe.
        /// * Assumes mask > 0
        pub fn lsb(self: Self) BitInt
        {
            assert(self.mask != 0);
            return @truncate(@ctz(self.mask));
        }

        /// Unsafe
        pub fn clear_lsb(self: *Self) void
        {
            assert(self.mask != 0);
            self.mask &= (self.mask - 1);
        }

        pub fn mask_bit(bit: BitInt) MaskInt
        {
            return @as(MaskInt, 1) << bit;
        }

        pub fn iterator(self: Self) Iterator
        {
            return .{ .bits_remaining = self.mask };
        }

        const Iterator = struct
        {
            /// all bits which have not yet been iterated over.
            bits_remaining: MaskInt,

            /// Returns the next set bit.
            pub fn next(self: *Iterator) ?BitInt
            {
                if (self.bits_remaining == 0) return null;
                const next_index: BitInt = @truncate(@ctz(self.bits_remaining));
                self.bits_remaining &= (self.bits_remaining - 1);
                return next_index;
            }
        };
    };
}

// TODO: move to funcs.zig and rename this file to bitset.zig (if we ever use it)

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
