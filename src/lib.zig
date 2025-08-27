// zig fmt: off

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub fn initialize() void
{
    memory_context = .init();
    io_context = .init();

    // Then initialize chess.
    @import("squarepairs.zig").initialize();
    @import("zobrist.zig").initialize();
    @import("data.zig").initialize();
    @import("masks.zig").initialize();
}

pub fn finalize() void
{
    memory_context.deinit();
}

/// For now we put it here.
pub const BoundedArray = @import("bounded_array.zig").BoundedArray;

pub const version= "0.1";

// Some app consts.
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release: bool = builtin.mode == .ReleaseFast;
pub const is_paranoid: bool = if (is_debug) true else false;

/// The global memory.
pub const ctx: *const MemoryContext = &memory_context;
/// The global io.
pub const io: *const IoContext = &io_context;

/// Global memory.
var memory_context: MemoryContext = undefined;
/// Global Io.
var io_context: IoContext = undefined;

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
            .galloc = if (is_debug) memory_context.gpa.allocator() else std.heap.smp_allocator,
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

var in_buffer: [4096]u8 = undefined;
var out_buffer: [4096]u8 = undefined;
var stdin: std.fs.File.Reader = undefined;
var stdout: std.fs.File.Writer = undefined;

const IoContext = struct
{
    in: *std.Io.Reader,
    out: *std.Io.Writer,

    fn init() IoContext
    {
        stdin = std.fs.File.stdin().reader(&in_buffer);
        stdout = std.fs.File.stdout().writer(&out_buffer);

        return
        .{
            .in = &stdin.interface,
            .out = &stdout.interface,
        };
    }

    pub fn readline(self: *const IoContext) !?[]const u8
    {
        const line = try self.in.takeDelimiterInclusive('\n');
        const input: []const u8 = std.mem.trimEnd(u8, line, "\r\n");
        return if (input.len >= 0) input else null;
    }

    /// By default print and flush. We need that the most in uci.
    pub fn print(self: *const IoContext, comptime str: []const u8, args: anytype) !void
    {
        try self.out.print(str, args);
        try self.out.flush();
    }

    pub fn print_buffered(self: *const IoContext, comptime str: []const u8, args: anytype) !void
    {
        try self.out.print(str, args);
    }

    pub fn flush(self: *const IoContext) !void
    {
        try self.out.flush();
    }

    pub fn debugprint(self: *const IoContext, comptime str: []const u8, args: anytype) void
    {
        _ = self;
        std.debug.print(str, args);
    }
};

/// Are we in terminal mode?
pub fn is_tty() bool
{
    return std.fs.File.stdin().isTty();
}

pub fn not_in_release() void
{
    if (is_release) @compileError("not in release!");
}

pub fn wtf() noreturn
{
    unreachable;
}