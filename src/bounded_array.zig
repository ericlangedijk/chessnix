// zig fmt: off

//! Because BoundedArray is gone from the stdlib

const std = @import("std");
const assert = std.debug.assert;

pub fn BoundedArray(comptime T: type, comptime buffer_capacity: usize) type {
    return struct {
        pub const empty: Self = .{};

        const Self = @This();
        buffer: [buffer_capacity]T = undefined,
        len: usize = 0,

        pub fn init(len: usize) error{Overflow}!Self {
            if (len > buffer_capacity) return error.Overflow;
            return Self{ .len = len };
        }

        pub fn slice(self: anytype) switch (@TypeOf(&self.buffer)) {
            *[buffer_capacity]T => []T,
            *const [buffer_capacity]T => []const T,
            else => unreachable,
        } {
            return self.buffer[0..self.len];
        }

        pub fn append(self: *Self, item: T) error{Overflow}!void {
            const new_item_ptr = try self.addOne();
            new_item_ptr.* = item;
        }

        pub fn append_assume_capacity(self: *Self, item: T) void {
            const new_item_ptr = self.add_one_assume_capacity();
            new_item_ptr.* = item;
        }

        pub fn append_slice(self: *Self, items: []const T) error{Overflow}!void {
            try self.ensure_unused_capacity(items.len);
            self.append_slice_assume_capacity(items);
        }

        pub fn append_slice_assume_capacity(self: *Self, items: []const T) void {
            const old_len = self.len;
            self.len += items.len;
            @memcpy(self.slice()[old_len..][0..items.len], items);
        }

        pub fn ensure_unused_capacity(self: Self, additional_count: usize) error{Overflow}!void {
            if (self.len + additional_count > buffer_capacity) {
                return error.Overflow;
            }
        }

        pub fn add_one(self: *Self) error{Overflow}!*T {
            try self.ensureUnusedCapacity(1);
            return self.add_one_assume_capacity();
        }

        pub fn add_one_assume_capacity(self: *Self) *T {
            assert(self.len < buffer_capacity);
            self.len += 1;
            return &self.slice()[self.len - 1];
        }


        pub fn unused_capacity_slice(self: *Self) []T {
            return self.buffer[self.len..];
        }

        pub fn print_assume_capacity(self: *Self, comptime fmt: []const u8, args: anytype) void {
            comptime assert(T == u8);
            assert(self.len < buffer_capacity);
            var w: std.io.Writer = .fixed(self.unused_capacity_slice());
            w.print(fmt, args) catch unreachable;
            self.len += w.end;
        }
    };
}
