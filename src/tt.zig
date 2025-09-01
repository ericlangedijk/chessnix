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

pub const Kind = enum(u2)
{
    Exact,
    Alpha,
    Beta,
};

pub const Entry = struct
{
    const empty: Entry = .{};

    /// The kind (exact, alpha or beta) with which this entry was stored..
    kind: Kind = .Exact,

    /// The position hash key. If key == 0 we assume this entry is empty.
    key: u64 = 0,

    /// The search depth when this entry was stored,
    depth: u8 = 0,

    /// The best move according to search.
    move: Move = .empty,

    /// The evaluation according to search.
    eval: Value = 0,

    fn to_result(self: *const Entry) Result
    {
        return .{ .move = self.move, .eval = self.eval };
    }

    fn to_result_adjusted(self: *const Entry, adjusted_eval: Value) Result
    {
        return .{ .move = self.move, .eval = adjusted_eval };
    }
};

pub const Result = struct
{
    move: Move,
    eval: Value,
};

// 64 MB = ok, 256 MB = very good.
pub const TranspositionTable = struct
{
    /// Array on heap.
    data: []Entry,

    /// The number of entries.
    size: u64,

    /// Number of filled entries.
    filled: u64,

    pub fn init(size_in_megabytes: u64) !TranspositionTable
    {
        if (!std.math.isPowerOfTwo(size_in_megabytes)) return Error.TTSizeMustBeAPowerOfTwo;

        const size: u64 = size_in_megabytes * 1024 * 1024;
        const data: []Entry = try ctx.galloc.alloc(Entry, size);
        @memset(data, Entry.empty);
        return .{ .data = data, .size = size, .filled = 0 };
    }

    pub fn deinit(self: *TranspositionTable) void
    {
        ctx.galloc.free(self.data);
    }

    pub fn clear(self: *TranspositionTable) void
    {
        @memset(self.data, Entry.empty);
        self.filled = 0;
    }

    /// TODO: atomic
    pub fn store(self: *TranspositionTable, kind: Kind, key: u64, depth: u8, ply: u16, move: Move, eval: Value) void
    {
        const entry: *Entry = self.get(key);
        const was_empty = entry.key == 0;
        const overwrite: bool = was_empty or entry.key != key or depth >= entry.depth;
        if (!overwrite) return;
        const corrected_eval: Value = get_corrected_eval_for_store(eval, ply);
        entry.kind = kind;
        entry.key = key;
        entry.depth = depth;
        entry.move = move;
        entry.eval = corrected_eval;
        if (was_empty) self.filled += 1;
    }

    pub fn probe(self: *TranspositionTable, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value) ?Result
    {
        const entry: *Entry = self.get(key);
        if (entry.key == key and entry.depth >= depth)
        {
            switch (entry.kind)
            {
                .Exact =>
                {
                    return entry.to_result();
                },
                .Beta =>
                {
                    const corr: Value = get_corrected_eval_for_probe(entry.eval, ply);
                    return if (corr >= beta) entry.to_result_adjusted(corr) else null;
                },
                .Alpha =>
                {
                    const corr: Value = get_corrected_eval_for_probe(entry.eval, ply);
                    return if (corr <= alpha) entry.to_result_adjusted(corr) else null;
                },
            }
        }
        return null;
    }

    pub fn permille(self: *const TranspositionTable) usize
    {
        return funcs.permille(self.size, self.filled);
    }

    fn index_of(self: *const TranspositionTable, key: u64) u64
    {
        return key & (self.size - 1);
    }

    fn get(self: *TranspositionTable, key: u64) *Entry
    {
        const idx = self.index_of(key);
        return &self.data[idx];
    }

    fn get_corrected_eval_for_store(eval: Value, ply: u16) Value
    {
        if (@abs(eval) >= types.mate_threshold)
        {
            const sign: i8 = if (eval >= 0) 1 else - 1;
            return ((eval * sign) + ply) * sign;
        }
        return eval;
    }

    fn get_corrected_eval_for_probe(eval: Value, ply: u16) Value
    {
        if (@abs(eval) >= types.mate_threshold)
        {
            const sign: i8 = if (eval >= 0) 1 else - 1;
            return ((eval * sign) - ply) * sign;
        }
        return eval;
    }
};

test "transpositiontable"
{
    const position = @import("position.zig");

    try lib.initialize();

    //lib.io.debugprint("testing tt---------------------\n", .{});

    var tt: TranspositionTable = try .init(8);
    defer tt.deinit();

    try std.testing.expectEqual(8 * 1024 * 1024, tt.size);

    var pos: position.Position = .empty;
    var st: position.StateInfo = undefined;
    try pos.set(&st, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const move: types.Move = .create(types.Square.E2, types.Square.A6);

    var e: ?Result = null;

    // Raw get
    {
        const eq_entry: Entry = .{ .kind = .Exact, .key = pos.state.key, .depth = 1, .move = move, .eval = 42 };
        tt.store(.Exact, pos.state.key, 1, 1, move, 42);
        const retrieved: *const Entry = tt.get(pos.state.key);
        try std.testing.expect(std.meta.eql(retrieved.*, eq_entry));
    }

    // Some probings
    {
        // Exact
        tt.store(.Exact, pos.state.key, 1, 1, move, 600);

        e = tt.probe(pos.state.key, 1, 1, -50, 50);
        try std.testing.expect(e != null);

        e = tt.probe(pos.state.key, 2, 1, -50, 50);
        try std.testing.expect(e == null);

        // // Alpha
        // tt.store(.Alpha, pos.state.key, 1, 1, move, 42);
        // e = tt.probe(pos.state.key, 1, 1, 43, -50);
        // try std.testing.expect(e == null);

        // // Beta
        // tt.store(.Beta, pos.state.key, 1, 1, move, 42);

        // e = tt.probe(pos.state.key, 1, 1, 50, -50);
        // try std.testing.expect(e != null);

        // e = tt.probe(pos.state.key, 1, 1, 50, 43);
        // try std.testing.expect(e == null);


    }

    try std.testing.expectEqual(1, tt.filled);
}


const Error = error
{
    TTSizeMustBeAPowerOfTwo,
};