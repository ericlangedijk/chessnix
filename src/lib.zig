// zig fmt: off

const std = @import("std");
const builtin = @import("builtin");
const utils = @import("utils.zig");

pub fn initialize() !void {
    if (lib_is_initialized) {
        return;
    }
    memory_context = .init();
    io_context = .init();
    lib_is_initialized = true;
}

pub fn finalize() void {
    lib_is_initialized = false;
    @import("position.zig").finalize();
    memory_context.deinit();
}

////////////////////////////////////////////////////////////////
// Globals.
////////////////////////////////////////////////////////////////
pub const Program = enum {
    /// We are a chess engine.
    engine,
    /// We are convering the lichess json eval dataset.
    generating_lichess_dataset,
    /// We are running a hce tuning.
    tuning,

    nothing,
};

pub const program: Program = .engine;
//pub const program: Program = .nothing;


// Running tuner. Otherwise running uci engine.
pub const is_tuning: bool = program == .tuning;

// pub const time_logging: bool = true;

pub const version = "1.4";
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release: bool = builtin.mode == .ReleaseFast;

/// Only when debugging we use time consumming checks.
pub const is_paranoid: bool = is_debug;

/// Using this for tricky bug hunting. Never in releasemode.
/// In ReleaseSafe mode we also create a logfile when we crash.
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
        stdin = std.fs.File.stdin().readerStreaming(&in_buffer);
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
        self.out.print(str, args) catch wtf("print", .{});
        self.out.flush() catch wtf("flush", .{});
    }

    /// uci only.
    pub fn print_buffered(self: *const IoContext, comptime str: []const u8, args: anytype) void {
        self.out.print(str, args) catch wtf("print", .{});
    }

    /// If `print_buffered` was used.
    pub fn flush(self: *const IoContext) void {
        self.out.flush() catch wtf("flush", .{});
    }

    /// Debug only.
    pub fn debugprint(_: *const IoContext, comptime str: []const u8, args: anytype) void {
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

pub fn wtf(comptime str: []const u8, args: anytype) noreturn {
    if (bughunt) {
        log_wtf(str, args);
    }
    std.debug.panic(str, args);
}

/// Only in ReleaseSafe.
pub fn verify(ok: bool, comptime str: []const u8, args: anytype) void {
    not_in_release();
    if (!ok) {
        wtf(str, args);
    }
}

/// Only in ReleaseSafe.
fn log_wtf(comptime str: []const u8, args: anytype) void {
    not_in_release();
    var writer = utils.TextFileWriter.init_cwd("chessnix.log", ctx.galloc, 256) catch return;
    defer writer.deinit();
    writer.writeline(str, args) catch return;
}

