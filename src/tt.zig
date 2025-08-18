// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");

const ctx = lib.ctx;
const wtf = lib.wtf();

const Value = types.Value;
const Move = types.Move;


pub const Kind = enum(u2)
{
    Empty,
    Exact,
    Alpha,
    Beta,
};

pub const Entry = struct
{
    const empty: Entry = .{};

    /// The kind (exact, alpha or beta) from which this entry was stored. Empty if unused slot.
    kind: Kind = .Empty,
    /// The position hash key.
    key: u64 = 0,
    /// The search depth when this entry was stored,
    depth: u8 = 0,
    /// The best move according to search.
    move: Move = .empty,
    /// The evaluation according to search.
    eval: Value = 0,
};

pub const TranspositionTable = struct
{
    data: std.ArrayListUnmanaged(Entry),
    size: u64,
    filled: u64,

    pub fn init(size_in_megabytes: u64) !TranspositionTable
    {
        const size: u64 = size_in_megabytes * 1024 * 1024;
        return
        .{
            .data = try create_data(size),
            .size = size,
            .filled = 0,
        };
    }

    pub fn deinit(self: *TranspositionTable) void
    {
        self.data.deinit(ctx.galloc);
    }

    fn create_data(size: u64) !std.ArrayListUnmanaged(Entry)
    {
        const list = try std.ArrayListUnmanaged(Entry).initCapacity(ctx.galloc, size);
        @memset(list.items, Entry.empty);
        return list;
    }

    pub fn clear(self: *TranspositionTable) void
    {
        @memset(self.data.items, Entry.empty);
        self.filled = 0;
    }

    pub fn store(self: *TranspositionTable, kind: Kind, key: u64, ply: u8, depth: u8, move: Move, eval: Value) void
    {
        const entry: *Entry = self.get(key);
        const was_empty = entry.kind == .empty;
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

    pub fn probe(self: *const TranspositionTable, key: u64, ply: u8, depth: u8, alpha: Value, beta: Value) ?*const Entry
    {
        const entry: *const Entry = self.get(key);
        if (entry.kind != .Empty and entry.key == key and entry.depth >= depth)
        {
            const corrected_eval: Value = get_corrected_eval_for_probe(entry.eval, ply);
            if
            (
                (entry.kind == .Exact) or
                (entry.kind == .Beta and corrected_eval >= beta) or
                (entry.kind == .Alpha and corrected_eval <= alpha)
            )
            return entry;
        }
        return null;
    }

    fn index_of(self: *const TranspositionTable, key: u64) u64
    {
        return key % (self.size - 1);
    }

    fn get(self: *TranspositionTable, key: u64) *Entry
    {
        const idx = self.index_of(key);
        return &self.data.items[idx];
    }

    fn get_corrected_eval_for_store(eval: Value, ply: u8) Value
    {
        if (@abs(eval) >= types.mate_threshold)
        {
            const sign: i8 = if (eval >= 0) 1 else - 1;
            return ((eval * sign) + ply) * sign;
        }
        return eval;
    }

    fn get_corrected_eval_for_probe(eval: Value, ply: u8) Value
    {
        if (@abs(eval) >= types.mate_threshold)
        {
            const sign: i8 = if (eval >= 0) 1 else - 1;
            return ((eval * sign) - ply) * sign;
        }
        return eval;
    }
};