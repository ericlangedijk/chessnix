// zig fmt: off

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub fn initialize() !void {
    if (lib_is_initialized) return;

    // If no timer available, the program is useless. In the rest of the code we use utils.Timer which assumes a timer is available.
    _ = try std.time.Timer.start();

    memory_context = .init();
    io_context = .init();

    lib_is_initialized = true;
}

pub fn finalize() void {
    lib_is_initialized = false;
    @import("position.zig").finalize();
    memory_context.deinit();
}

/// For now we put it here.
pub const BoundedArray = @import("utils.zig").BoundedArray;

////////////////////////////////////////////////////////////////
// Globals.
////////////////////////////////////////////////////////////////
pub const version = "1.3";
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release: bool = builtin.mode == .ReleaseFast;

/// Using this for time consuming checks.
pub const is_paranoid: bool = is_debug;

/// Using this for tricky bug hunting. Never in releasemode.
/// In ReleaseSafe mode we also create a logfile with the eror when calling crash().
pub const bughunt: bool = builtin.mode == .ReleaseSafe or builtin.mode == .Debug;

// Input output
pub const ctx: *const MemoryContext = &memory_context;
pub const io: *IoContext = &io_context;
var memory_context: MemoryContext = undefined;
var io_context: IoContext = undefined;
var lib_is_initialized: bool = false;

/// The global memory context of our exe
pub const MemoryContext = struct {
    gpa: if (is_debug) std.heap.DebugAllocator(.{}) else void,
    galloc: std.mem.Allocator,

    fn init() MemoryContext {
        return MemoryContext {
            .gpa = if (is_debug) std.heap.DebugAllocator(.{}).init else {},
            .galloc = if (builtin.is_test) std.testing.allocator else if (is_debug) memory_context.gpa.allocator() else std.heap.smp_allocator,
        };
    }

    fn deinit(self: *MemoryContext) void {
        if (is_debug) {
            _ = self.gpa.deinit();
        }
    }
};

// TODO: I cannot find a solution for these vars, which I would like to have inside IoContext.
var in_buffer: [8192]u8 = @splat(0);
var out_buffer: [4096]u8 = @splat(0);
var stdin: std.fs.File.Reader = undefined;
var stdout: std.fs.File.Writer = undefined;

const IoContext = struct {
    in: *std.Io.Reader,
    out: *std.Io.Writer,

    fn init() IoContext {
        stdin = std.fs.File.stdin().reader(&in_buffer);
        // stdin = std.fs.File.stdin().readerStreaming(&in_buffer); // NOTE: this seems to be safer
        stdout = std.fs.File.stdout().writer(&out_buffer);
        return .{
            .in = &stdin.interface,
            .out = &stdout.interface,
        };
    }

    /// Returns line without eol delimiters.
    pub fn readline(self: *const IoContext) !?[]const u8 {
        const line = try self.in.takeDelimiterInclusive('\n');
        const input: []const u8 = std.mem.trimEnd(u8, line, "\r\n");
        return if (input.len >= 0) input else null;
    }

    /// By default print and flush.
    pub fn print(self: *const IoContext, comptime str: []const u8, args: anytype) void {
        self.out.print(str, args) catch wtf();
        self.out.flush() catch wtf();
    }

    /// uci only.
    pub fn print_buffered(self: *const IoContext, comptime str: []const u8, args: anytype) void {
        self.out.print(str, args) catch wtf();
    }

    /// If `print_buffered` was used.
    pub fn flush(self: *const IoContext) void {
        self.out.flush() catch wtf();
    }

    /// Output depends on debug or releasesafe mode
    pub fn printerror(self: *const IoContext, comptime str: []const u8, args: anytype) void {
        not_in_release();
        if (is_debug) {
            std.debug.print(str, args);
        }
        else {
            self.print(str, args);
        }
    }

    /// Debug only.
    pub fn debugprint(_: *const IoContext, comptime str: []const u8, args: anytype) void {
        not_in_release();
        std.debug.print(str, args);
    }

    pub fn dpr(_: *const IoContext, comptime str: []const u8) void {
        not_in_release();
        std.debug.print("{s}\n", .{ str });
    }

    /// Allowed during UCI.
    pub fn errorprint(_: *const IoContext, comptime str: []const u8, args: anytype) void {
        not_in_release();
        std.debug.print(str, args);
    }
};

/// Are we in terminal mode?
pub fn is_tty() bool {
    return std.fs.File.stdin().isTty();
}

/// Call this anywhere where we do not want this in a release version.
pub fn not_in_release() void {
    if (is_release) @compileError("not in release!");
}

pub fn wtf() noreturn {
    @panic("wtf"); // TODO: finetune
}

/// Only in ReleaseSafe
pub fn verify(ok: bool, comptime str: []const u8, args: anytype) void {
    not_in_release();
    if (!ok) {
        crash(str, args);
    }
}

pub fn crash(comptime str: []const u8, args: anytype) noreturn {
    not_in_release();
    if (builtin.mode == .ReleaseSafe) {
        log_crash(str, args);
    }
    std.debug.panic(str, args);
}

/// Internal function
fn log_crash(comptime str: []const u8,  args: anytype) void {
    not_in_release();
    var writer = @import("utils.zig").TextFileWriter.init_cwd("chessnix.log", ctx.galloc, 256) catch return;
    defer writer.deinit();
    writer.writeline(str, args) catch return;
}