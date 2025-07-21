//! No clue what is the best way to handle terminals.

const std = @import("std");
const lib = @import("lib.zig");

const wtf = lib.wtf;

pub fn check() void
{
    const writer = std.io.getStdOut().writer();
    print_str(@typeName(@TypeOf(writer)));
}

fn internal_print(comptime fmt: []const u8, args: anytype) void
{
    const writer = std.io.getStdOut().writer();
    writer.print(fmt, args) catch |err|
    {
        @panic(@errorName(err));
    };
}

/// The default print funcion.
pub fn print(comptime fmt: []const u8, args: anytype) void
{
    internal_print(fmt, args);
}

pub fn print_str(str: []const u8) void
{
    internal_print("{s}", .{str});
}

pub fn print_ln() void
{
    internal_print("\n", .{});
}
