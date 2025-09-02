// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");

const ctx = lib.ctx;
const wtf = lib.wtf();

const Value = types.Value;
const Move = types.Move;

/// Test phase only.
pub const using_tt: bool = true;

pub const Bound = enum(u2)
{
    None,
    Exact,
    Lower,
    Upper,
};

/// Must be 16 bytes.
pub const Entry = packed struct
{
    const empty: Entry =
    .{
        .bound = .None,
        .key = 0,
        .depth = 0,
        .move = .empty,
        .score = 0,
        .age = 0,
    };

    /// The kind (exact, alpha or beta) with which this entry was stored..
    bound: Bound,

    /// The position hash key. If key == 0 we assume this entry is empty.
    key: u64,

    /// The search depth when this entry was stored,
    depth: u8,

    /// The best move according to search.
    move: Move,

    /// The evaluation according to search.
    score: Value,

    /// The age
    age: u6,

    // fn to_result(self: *const Entry) Result
    // {
    //     return .{ .move = self.move, .score = self.score };
    // }

    // fn to_result_adjusted(self: *const Entry, adjusted_score: Value) Result
    // {
    //     return .{ .move = self.move, .score = adjusted_score };
    // }
};

// 64 MB = ok, 256 MB = very good.
pub const TranspositionTable = struct
{
    /// Array on heap.
    data: []Entry,

    /// The number of entries.
    len: u64,

    /// Used bytes
    mb: u64,

    /// Number of filled entries.
    filled: u64,

    /// Number of probe counts.
    probes: u64,

    /// Number of succesful probes.
    hits: u64,

    age: u6,

    pub fn init(size_in_megabytes: u64) !TranspositionTable
    {
        comptime if (@sizeOf(Entry) != 16) @compileError("TT Entry must be 16 bytes");
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;
        const len: u64 = (size_in_megabytes * 1024 * 1024) / 16;
        const data: []Entry = try ctx.galloc.alloc(Entry, len);
        @memset(data, Entry.empty);
        return .{ .data = data, .len = len, .mb = size_in_megabytes, .filled = 0, .probes = 0, .hits = 0, .age = 0 };
    }

    pub fn deinit(self: *TranspositionTable) void
    {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *TranspositionTable) void
    {
        @memset(self.data, Entry.empty);
        self.filled = 0;
        self.probes = 0;
        self.hits = 0;
    }

    pub fn increase_age(self: *TranspositionTable) void
    {
        self.age +%= 1;
    }

    pub fn permille(self: *const TranspositionTable) usize
    {
        return funcs.permille(self.len, self.filled);
    }

    pub fn store(self: *TranspositionTable, bound: Bound, key: u64, depth: u8, move: Move, score: Value) void
    {
        // TODO: atomic store.
        if (!using_tt) return;
        const entry: *Entry = self.get_mut(key);
        const was_empty = entry.key == 0;
        const overwrite: bool = was_empty or entry.age != self.age or entry.key != key or depth >= entry.depth;
        if (!overwrite) return;
        entry.* = .{ .bound = bound, .key = key, .depth = depth, .move = move, .score = score, .age = self.age };
        if (was_empty) self.filled += 1;
    }

    pub fn probe(self: *TranspositionTable, key: u64) ?Entry
    {
        if (!using_tt) return null;
        self.probes += 1;
        const entry: Entry = self.get_entry(key);
        if (entry.bound == .None or entry.key != key) return null;
        self.hits += 1;
        return entry;
    }

    fn index_of(self: *const TranspositionTable, key: u64) u64
    {
        return key & (self.len - 1);
    }

    fn get_mut(self: *TranspositionTable, key: u64) *Entry
    {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    pub fn get_entry(self: *TranspositionTable, key: u64) Entry
    {
        const idx = self.index_of(key);
        return self.data[idx];
    }

};

/// Adjust score for mate in X when storing.
pub fn get_adjusted_score_for_store(score: Value, ply: u16) Value
{
    if (@abs(score) >= types.mate_threshold)
    {
        const sign: i8 = if (score > 0) 1 else - 1;
        return ((score * sign) + ply) * sign;
    }
    return score;
}

/// Adjust score for mate in X when probing.
pub fn get_adjusted_score_for_probe(score: Value, ply: u16) Value
{
    if (@abs(score) >= types.mate_threshold)
    {
        const sign: i8 = if (score > 0) 1 else - 1;
        return ((score * sign) - ply) * sign;
    }
    return score;
}

// test "transpositiontable"
// {
//     const position = @import("position.zig");

//     try lib.initialize();

//     //lib.io.debugprint("testing tt---------------------\n", .{});

//     var tt: TranspositionTable = try .init(8);
//     defer tt.deinit();

//     try std.testing.expectEqual(8 * 1024 * 1024, tt.size);

//     var pos: position.Position = .empty;
//     var st: position.StateInfo = undefined;
//     try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
//     const move: types.Move = .create(types.Square.E2, types.Square.A6);

//     var e: ?Result = null;

//     // Raw get
//     {
//         const eq_entry: Entry = .{ .kind = .Exact, .key = pos.state.key, .depth = 1, .move = move, .eval = 42 };
//         tt.store(.Exact, pos.state.key, 1, 1, move, 42);
//         const retrieved: *const Entry = tt.get(pos.state.key);
//         try std.testing.expect(std.meta.eql(retrieved.*, eq_entry));
//     }

//     // Some probings
//     {
//         // Exact
//         tt.store(.Exact, pos.state.key, 1, 1, move, 600);

//         e = tt.probe(pos.state.key, 1, 1, -50, 50);
//         try std.testing.expect(e != null);

//         e = tt.probe(pos.state.key, 2, 1, -50, 50);
//         try std.testing.expect(e == null);

//         // // Alpha
//         // tt.store(.Alpha, pos.state.key, 1, 1, move, 42);
//         // e = tt.probe(pos.state.key, 1, 1, 43, -50);
//         // try std.testing.expect(e == null);

//         // // Beta
//         // tt.store(.Beta, pos.state.key, 1, 1, move, 42);

//         // e = tt.probe(pos.state.key, 1, 1, 50, -50);
//         // try std.testing.expect(e != null);

//         // e = tt.probe(pos.state.key, 1, 1, 50, 43);
//         // try std.testing.expect(e == null);


//     }

//     try std.testing.expectEqual(1, tt.filled);
// }


const Error = error
{
    TTSizeMustBeAPowerOfTwo,
};