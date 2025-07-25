const std = @import("std");
const builtin = @import("builtin");

const assert = std.debug.assert;

// Some app consts.
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release: bool = builtin.mode == .ReleaseFast;
pub const is_paranoid: bool = if (is_debug) true else false;

/// The global memory.
pub const ctx: *const MemoryContext = &private_context;

var private_context: MemoryContext = undefined;
//var private_console_output: UTF8ConsoleOutput = undefined;

pub fn initialize() void
{
    private_context = .init();
    //private_console_output = UTF8ConsoleOutput.init();

    @import("squarepairs.zig").initialize();
    @import("zobrist.zig").initialize();
    @import("data.zig").initialize();
    @import("masks.zig").initialize();
}

pub fn finalize() void
{
    //private_console_output.deinit();
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

pub const What = enum
{
    Memory,
};


pub fn wtf() noreturn
{
    unreachable;
    //@panic("WTF");
}

pub fn crash(comptime what: What) noreturn
{
    @panic(@tagName(what));
}

pub fn not_in_release() void
{
    if (is_release) @compileError("not in release!");
}

/// Temp solution.
const UTF8ConsoleOutput = struct
{
    original: ?c_uint = null,

    fn init() UTF8ConsoleOutput
    {

        // const windows = @cImport({
        //     @cInclude("windows.h");
        // });

        var self = UTF8ConsoleOutput{};
        if (builtin.os.tag == .windows)
        {
            const kernel32 = std.os.windows.kernel32;
            self.original = kernel32.GetConsoleOutputCP();
            _ = kernel32.SetConsoleOutputCP(65001);

            //const mode: u32 = kernel32.SetConsoleMode()
        }
        return self;
    }

    fn deinit(self: *UTF8ConsoleOutput) void
    {
        if (self.original) |org|
        {
            _ = std.os.windows.kernel32.SetConsoleOutputCP(org);
        }
    }
};
