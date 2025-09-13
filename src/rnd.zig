// zig fmt: off

//! My predictable randoms.

const std = @import("std");

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

    pub fn next_u32(self: *Random) u32 {
        const n: u64 = self.next_u64();
        const a: u32 = @truncate(n);
        const b: u32 = @truncate(n >> 32);
        return a ^ b;
    }

    pub fn next_u64_max(self: *Random, max: u64) u64 {
        const n = self.next_u64() % max;
        return n;
    }
};

