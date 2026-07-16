// zig fmt: off

//! Basics: memory, io, comptime config, error handling.

const std = @import("std");
const builtin = @import("builtin");
const utils = @import("utils.zig");

pub fn initialize() !void {
    comptime compilation_check();
    memory_context = .init();
    io_context = .init();
    init_console();
}

pub fn finalize() void {
    @import("position.zig").finalize();
    memory_context.deinit();
}

fn compilation_check() void {
    const v = "0.15.2";
    if (!std.mem.eql(u8, builtin.zig_version_string, v)) {
        @compileError("this chessnix version requires zig " ++ v);
    }
    if (@sizeOf(usize) != @sizeOf(u64)) {
        @compileError("target must be 64 bits");
    }
    if (is_release) {
        if (is_paranoid) @compileError("release is_paranoid");
    }
}

// --- Globals ---
pub const Program = enum {
    uci,
    hcetuner,
    lichess_dataset_conversion,
};

 pub const program: Program = .uci;
//pub const program: Program = .hcetuner;
//pub const program: Program = .lichess_dataset_conversion;

pub const version = "1.5";
pub const builddate = "2026-07-16";
pub const is_test: bool = builtin.is_test;
pub const is_debug: bool = builtin.mode == .Debug;
pub const is_release_safe: bool = builtin.mode == .ReleaseSafe;
pub const is_release: bool = builtin.mode == .ReleaseFast;
pub const is_tuning: bool = !is_test and program == .hcetuner;
pub const is_paranoid: bool = is_test or is_debug or is_release_safe;

// --- Io and memory ---
pub const ctx: *const MemoryContext = &memory_context;
pub const io: *IoContext = &io_context;
var memory_context: MemoryContext = undefined;
var io_context: IoContext = undefined;

/// The global memory context of our exe.
pub const MemoryContext = struct {
    gpa: std.mem.Allocator,

    fn init() MemoryContext {
        if (builtin.is_test) {
            return .{ .gpa = std.testing.allocator };
        }
        else if (is_debug) {
            return .{ .gpa = std.heap.DebugAllocator(.{}).init.backing_allocator };
        }
        else {
            return .{ .gpa = std.heap.smp_allocator };
        }
    }

    fn deinit(self: *MemoryContext) void {
        _ = self;
    }
};

var in_buffer: [8192]u8 = @splat(0);
var out_buffer: [4096]u8 = @splat(0);
var stdin: std.fs.File.Reader = undefined;
var stdout: std.fs.File.Writer = undefined;

const IoContext = struct {
    in: *std.Io.Reader,
    out: *std.Io.Writer,

    fn init() IoContext {
        // Thanks to Jonathan Hallström.
        stdin = std.fs.File.stdin().readerStreaming(&in_buffer);
        stdout = std.fs.File.stdout().writerStreaming(&out_buffer);
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

fn init_console() void {
    if (is_tty()) {
        _ = std.fs.File.stdout().getOrEnableAnsiEscapeSupport();
    }
}

/// Are we in terminal mode?
pub fn is_tty() bool {
    return std.fs.File.stdin().isTty();
}

pub inline fn not_in_release() void {
    if (is_release) @compileError("not in release!");
}

pub inline fn only_when_tuning() void {
    if (!is_tuning) @compileError("only when tuning!");
}

pub inline fn only_in_comptime() void {
    if (!@inComptime()) @compileError("only in comptime!");
}

/// Allowed in all built modes.
pub fn wtf(comptime str: []const u8, args: anytype) noreturn {
    std.debug.panic(str, args);
}

/// Only in ReleaseSafe or Debug.
pub fn verify(ok: bool, comptime str: []const u8, args: anytype) void {
    not_in_release();
    if (!ok) {
        std.debug.panic(str, args);
    }
}

/// Only in ReleaseSafe.
pub fn panic_function(msg: []const u8, returnaddress: ?usize) noreturn {
    not_in_release();
    if (is_debug) @compileError("not allowed in debug mode");

    // Dump the error + stacktrace to a file in the current working directory.
    var writer = utils.FileWriter.init_cwd("chessnix.log", 4096) catch std.process.exit(1);
    writer.wr.interface.print("{s}\n\n", .{ msg }) catch std.process.exit(1);
    dump_stack_trace(returnaddress, &writer.wr.interface) catch {};
    writer.deinit();

    // I don't know if the GUI's that drive chessnix are able to do something with it.
    std.debug.print("{s}", .{ msg });
    std.process.exit(1);
}

/// Prevent colored output.
pub fn dump_stack_trace(start_addr: ?usize, writer: *std.Io.Writer) !void {
    if (builtin.strip_debug_info) {
        try writer.writeAll("Unable to dump stack trace: debug info stripped\n");
        return;
    }
    const debug_info = std.debug.getSelfDebugInfo() catch |err| {
        try writer.print("Unable to dump stack trace: Unable to open debug info: {s}\n", .{ @errorName(err) });
        return;
    };
    std.debug.writeCurrentStackTrace(writer, debug_info, .no_color, start_addr) catch |err| {
        try writer.print("Unable to dump stack trace: {s}\n", .{ @errorName(err) });
        return;
    };
}
