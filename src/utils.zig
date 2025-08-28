// zig fmt: off

//! Non-chess related utilities.

const std = @import("std");
const lib = @import("lib.zig");

const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;

/// A little wrapper around the std times.
pub const Timer = struct
{
    std_timer: std.time.Timer,

    pub fn start() Timer
    {
        return
        .{
            .std_timer = std.time.Timer.start() catch wtf()
        };
    }

    pub fn read(self: *Timer) u64
    {
        return self.std_timer.read();
    }

    pub fn lap(self: *Timer) u64
    {
        return self.std_timer.lap();
    }

    pub fn reset(self: *Timer) void
    {
        self.std_timer.reset();
    }

    pub fn elapsed_ms(self: *Timer) u64
    {
        return self.std_timer.read() / std.time.ns_per_ms;
    }

    /// Handy routine for doing something after an interval.
    pub fn ticked(self: *Timer, milliseconds: u64) bool
    {
        if (self.elapsed_ms() < milliseconds) return false;
        self.reset();
        return true;
    }
};

/// A little convenient line reader.
/// * Max linesize must be known.
pub const TextFileReader = struct
{
    allocator: std.mem.Allocator,
    buffer: []u8,
    reader: std.fs.File.Reader,

    pub fn init(filename: []const u8, allocator: std.mem.Allocator, linebuffer_size: usize) !TextFileReader
    {
        const file = try std.fs.openFileAbsolute(filename, .{});
        const buf = try allocator.alloc(u8, linebuffer_size);
        return
        .{
            .allocator = allocator,
            .buffer = buf,
            .reader = file.reader(buf),
        };
    }

    pub fn deinit(self: *TextFileReader) void
    {
        self.reader.file.close();
        self.allocator.free(self.buffer);
    }

    pub fn readline(self: *TextFileReader) !?[]const u8
    {
        const line = self.reader.interface.takeDelimiterExclusive('\n') catch |err|
        {
            return if (err == std.io.Reader.DelimiterError.EndOfStream) null else err;
        };
        return std.mem.trimEnd(u8, line, "\r\n");
    }
};
