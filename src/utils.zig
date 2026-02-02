// zig fmt: off

//! Non-chess utilities.

const std = @import("std");
const lib = @import("lib.zig");

const ctx = lib.ctx;
const io = lib.io;
const assert = std.debug.assert;
const wtf = lib.wtf;

/// A little wrapper around the std timer.
pub const Timer = struct {
    pub const empty: Timer = std.mem.zeroes(Timer);

    std_timer: std.time.Timer,

    /// NOTE: at program startup we check once if timer is available, otherwise quit.
    /// From then on we assume it is there.
    pub fn start() Timer {
        return .{
            .std_timer = std.time.Timer.start() catch wtf()
        };
    }

    /// Elapsed nanoseconds.
    pub fn read(self: *Timer) u64 {
        return self.std_timer.read();
    }

    /// Elapsed nanoseconds.
    pub fn lap(self: *Timer) u64 {
        return self.std_timer.lap();
    }

    pub fn reset(self: *Timer) void {
        self.std_timer.reset();
    }

    pub fn elapsed_ms(self: *Timer) u64 {
        return self.std_timer.read() / std.time.ns_per_ms;
    }

    /// Handy routine for doing something after an interval.
    pub fn ticked(self: *Timer, milliseconds: u64) bool {
        if (self.elapsed_ms() < milliseconds) return false;
        self.reset();
        return true;
    }
};

/// My predictable random.
pub const Random = struct {
    pub const empty: Random = .{};

    const OFFSET: u64 = 0x7f7110ba7879ea6a;
    seed: u64 = 0,

    pub fn init(seed: u64) Random {
        return .{ .seed = OFFSET *% seed };
    }

    pub fn init_randomized() Random {
        return .{ .seed = OFFSET *% generate_timeseed() };
    }

    fn generate_timeseed() u64 {
        const t: i64 = std.time.microTimestamp();
        const u: u64 = @bitCast(t);
        return u;
    }

    pub fn reset_seed(self: *Random, seed: u64) void {
        self.seed = OFFSET *% seed;
    }

    pub fn next_u64(self: *Random) u64 {
        self.seed = self.seed +% 0x9e3779b97f4a7c15;
        var z: u64 = self.seed;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        z = (z ^ (z >> 31));
        return z;
    }
};

/// A little convenient line reader. Max linesize must be known.
pub const TextFileReader = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    reader: std.fs.File.Reader,

    pub fn init(filename: []const u8, allocator: std.mem.Allocator, linebuffer_size: usize) !TextFileReader {
        const file = try std.fs.openFileAbsolute(filename, .{});
        const buf = try allocator.alloc(u8, linebuffer_size);
        return .{
            .allocator = allocator,
            .buffer = buf,
            .reader = file.reader(buf),
        };
    }

    pub fn deinit(self: *TextFileReader) void {
        self.reader.file.close();
        self.allocator.free(self.buffer);
    }

    pub fn readline(self: *TextFileReader) !?[]const u8 {
        const line = self.reader.interface.takeDelimiterInclusive('\n') catch |err| {
            return if (err == std.io.Reader.DelimiterError.EndOfStream) null else err;
        };
        return std.mem.trimEnd(u8, line, "\r\n");
        // const line = self.reader.interface.takeDelimiterExclusive('\n') catch |err| {
        //     return if (err == std.io.Reader.DelimiterError.EndOfStream) null else err;
        // };
        // return std.mem.trimEnd(u8, line, "\r\n");
    }
};

/// A little convenient line writer.
pub const TextFileWriter = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    writer: std.fs.File.Writer,

    pub fn init(filename: []const u8, allocator: std.mem.Allocator, buffer_size: usize) !TextFileWriter {
        const file = try std.fs.createFileAbsolute(filename, .{});
        const buf = try allocator.alloc(u8, buffer_size);
        return .{
            .allocator = allocator,
            .buffer = buf,
            .writer = file.writer(buf),
        };
    }

    pub fn init_cwd(filename: []const u8, allocator: std.mem.Allocator, buffer_size: usize) !TextFileWriter {
        const file = try std.fs.cwd().createFile(filename, .{ .lock_nonblocking = true });
        const buf = try allocator.alloc(u8, buffer_size);
        return .{
            .allocator = allocator,
            .buffer = buf,
            .writer = file.writer(buf),
        };
    }

    pub fn deinit(self: *TextFileWriter) void {
        self.writer.interface.flush() catch {};
        self.writer.file.close();
        self.allocator.free(self.buffer);
    }

    pub fn write(self: *TextFileWriter, comptime fmt: []const u8, args: anytype) !void {
        try self.writer.interface.print(fmt, args);
    }

    pub fn writeline(self: *TextFileWriter, comptime fmt: []const u8, args: anytype) !void {
        try self.writer.interface.print(fmt, args);
        try self.writer.interface.writeByte(10);
    }

    pub fn flush(self: *TextFileWriter) !void {
        try self.writer.interface.flush();
    }
};

/// Because BoundedArray is gone from the std 0.15.
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
