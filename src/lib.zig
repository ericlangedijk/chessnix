// zig fmt: off

const std = @import("std");
const builtin = @import("builtin");

const assert = std.debug.assert;

pub const version= "0.1";

// Some app consts.
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release: bool = builtin.mode == .ReleaseFast;
pub const is_paranoid: bool = if (is_debug) true else false;

/// The global memory.
pub const ctx: *const MemoryContext = &private_context;

// IO.
pub var in: std.fs.File.Reader = undefined;
pub var out: std.fs.File.Writer = undefined;

// Internal.
var private_context: MemoryContext = undefined;
var private_is_tty: bool = false;

pub fn initialize() void
{
    private_context = .init();

    in = std.io.getStdIn().reader();
    out = std.io.getStdOut().writer();
    private_is_tty = std.io.getStdOut().isTty();

    @import("squarepairs.zig").initialize();
    @import("zobrist.zig").initialize();
    @import("data.zig").initialize();
    @import("masks.zig").initialize();
}

pub fn finalize() void
{
    private_context.deinit();
}

/// The global memory context of our exe
pub const MemoryContext = struct
{
    gpa: if (is_debug) std.heap.DebugAllocator(.{}) else void,
    galloc: std.mem.Allocator,

    fn init() MemoryContext
    {
        return MemoryContext
        {
            .gpa = if (is_debug) std.heap.DebugAllocator(.{}).init else {},
            .galloc = if (is_debug) private_context.gpa.allocator() else std.heap.smp_allocator,
        };
    }

    fn deinit(self: *MemoryContext) void
    {
        if (is_debug)
        {
            _ = self.gpa.deinit();
        }
    }
};

/// Are we in terminal mode?
pub fn is_tty() bool
{
    return private_is_tty;
}

/// Print without try :)
pub fn print(comptime fmt: []const u8, args: anytype) void
{
    out.print(fmt, args) catch wtf();
}

pub fn not_in_release() void
{
    if (is_release) @compileError("not in release!");
}

pub fn wtf() noreturn
{
    unreachable;
}