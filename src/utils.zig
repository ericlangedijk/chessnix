// zig fmt: off

//! Non-chess utility structs.

const std = @import("std");
const lib = @import("lib.zig");

const ctx = lib.ctx;
const io = lib.io;

const assert = std.debug.assert;
const wtf = lib.wtf;

/// An easy fixed buffer short string.
pub fn String(comptime maxlen: u8) type {
    return BoundedArray(u8, maxlen);
}

pub const Timer = struct {
    const Instant = std.time.Instant;

    started: Instant,
    previous: Instant,

    pub const empty: Timer = std.mem.zeroes(Timer);

    pub fn start() Timer {
        const current = Instant.now() catch wtf("no timer", .{});
        return Timer{ .started = current, .previous = current };
    }

    pub fn reset(self: *Timer) void {
        const current = self.sample();
        self.started = current;
    }

    /// Elapsed nanoseconds.
    pub fn read(self: *Timer) u64 {
        const current: Instant = self.sample();
        return current.since(self.started);
    }

    /// Returns elapsed nanoseconds and resets.
    pub fn lap(self: *Timer) u64 {
        const current: Instant = self.sample();
        defer self.started = current;
        return current.since(self.started);
    }

    pub fn elapsed_ms(self: *Timer) u64 {
        return self.read() / std.time.ns_per_ms;
    }

    /// Handy routine for doing something after an interval.
    pub fn ticked(self: *Timer, milliseconds: u64) bool {
        if (self.elapsed_ms() < milliseconds) {
            return false;
        }
        self.reset();
        return true;
    }

    fn sample(self: *Timer) Instant {
        const current: Instant = Instant.now() catch unreachable;
        if (current.order(self.previous) == .gt) {
            self.previous = current;
        }
        return self.previous;
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

    pub fn next(self: *Random) u64 {
        self.seed = self.seed +% 0x9e3779b97f4a7c15;
        var z: u64 = self.seed;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        z = (z ^ (z >> 31));
        return z;
    }

    pub fn next_max(self: *Random, max: u64) u64 {
        const n = self.next() % max;
        return n;
    }
};

/// Using the global lib allocator.
pub const FileReader = struct {
    buf: []u8,
    rd: std.fs.File.Reader,

    pub fn init(filename: []const u8, buf_size: usize) !FileReader {
        const file = try std.fs.openFileAbsolute(filename, .{});
        const buf = try ctx.gpa.alloc(u8, buf_size);
        return .{
            .buf = buf,
            .rd = file.reader(buf),
        };
    }

    pub fn deinit(self: *FileReader) void {
        self.rd.file.close();
        ctx.gpa.free(self.buf);
    }

    pub fn readline(self: *FileReader) !?[]const u8 {
        const line = self.rd.interface.takeDelimiterInclusive('\n') catch |err| {
            return if (err == std.io.Reader.DelimiterError.EndOfStream) null else err;
        };
        return std.mem.trimEnd(u8, line, "\r\n");
    }

    pub fn reset(self: *FileReader) void {
        self.rd.seekTo(0);
    }
};

/// Using the global lib allocator.
pub const FileWriter = struct {
    buf: []u8,
    wr: std.fs.File.Writer,

    pub fn init(filename: []const u8, buffer_size: usize) !FileWriter {
        const file = try std.fs.createFileAbsolute(filename, .{});
        const buf = try ctx.gpa.alloc(u8, buffer_size);
        return .{
            .buf = buf,
            .wr = file.writer(buf),
        };
    }

    pub fn init_cwd(filename: []const u8, buffer_size: usize) !FileWriter {
        const file = try std.fs.cwd().createFile(filename, .{ .lock_nonblocking = true });
        const buf = try ctx.gpa.alloc(u8, buffer_size);
        return .{
            .buf = buf,
            .wr = file.writer(buf),
        };
    }

    /// Flushes before freeing.
    pub fn deinit(self: *FileWriter) void {
        self.wr.interface.flush() catch {};
        self.wr.file.close();
        ctx.gpa.free(self.buf);
    }

    pub fn write(self: *FileWriter, comptime fmt: []const u8, args: anytype) !void {
        try self.wr.interface.print(fmt, args);
    }

    pub fn writeline(self: *FileWriter, comptime fmt: []const u8, args: anytype) !void {
        try self.wr.interface.print(fmt, args);
        try self.wr.interface.writeByte(10);
    }

    pub fn flush(self: *FileWriter) !void {
        try self.wr.interface.flush();
    }
};

/// BoundedArray left the scene with Zig 0.15.
pub fn BoundedArray(comptime T: type, comptime buffer_capacity: usize) type {
    return struct {
        const Self = @This();

        pub const max_capacity: usize = buffer_capacity;

        buffer: [buffer_capacity]T = undefined,
        len: usize = 0,

        pub const empty: Self = .{};

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
            w.print(fmt, args) catch lib.wtf("print", .{});
            self.len += w.end;
        }
    };
}
