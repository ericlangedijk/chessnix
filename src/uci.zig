// zig fmt: off
const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const typoes = @import("types.zig");
const position = @import("position.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Move = typoes.Move;
const Position = position.Position;

const in = lib.in;
const out = lib.out;

const ctx = lib.ctx;

pub fn run() !void
{
    //const format = comptime std.fmt.format;

    if (lib.is_tty())
    {
        try out.print("chessnix {s} by Eric Langedijk\n", .{lib.version});
    }

    var allocator = std.heap.ArenaAllocator.init(ctx.galloc);
    defer allocator.deinit();

    var pos: Position = .new();
    defer pos.deinit();

    while (true)
    {
        const line = try in.readUntilDelimiterOrEofAlloc(allocator.allocator(), '\n', 4096) orelse return;
        const input: []const u8 = std.mem.trim(u8, line, "\r");
        var tokenizer = std.mem.tokenizeScalar(u8, input, ' ');
        const cmd: []const u8 = tokenizer.next() orelse continue;

        if (eql(cmd, "uci"))
        {
            try out.print("id chessnix {s}\nauthor eric\nuciok\n", .{lib.version});
        }
        else if (eql(cmd, "isready"))
        {
            try out.print("readyok\n", .{});
        }
        else if (eql(cmd, "ucinewgame"))
        {
            pos.set_startpos();
        }
        else if (eql(cmd, "go"))
        {
            try out.print("TODO: WE SHOULD START THINKING NOW\n", .{});
        }
        else if (eql(cmd, "stop"))
        {
            try out.print("TODO: WE SHOULD STOP THINKING NOW\n", .{});
        }
        else if (eql(cmd, "position"))
        {
            const next = tokenizer.next() orelse continue;
            if (eql(next, "fen"))
            {
                const fen_str = tokenizer.next() orelse continue;
                try pos.set_fen(fen_str);
            }
            else if (eql(next, "startpos"))
            {
                pos.set_startpos();
            }

            const moves = tokenizer.next() orelse continue;
            if (eql(moves, "moves"))
            {
                while (tokenizer.next()) |move|
                {
                    const m: Move = pos.parse_move(move) catch continue;
                    pos.lazy_make_move(m);
                }
            }
        }
        else if (eql(cmd, "quit"))
        {
            return;
        }
        // custom
        else if (eql(cmd, "d"))
        {
            try position.print_pos(&pos);
        }
        // custom
        else if (eql(cmd, "bench"))
        {
            try tests.bench();
        }
        // custom
        if (eql(cmd, "perft"))
        {
            const next = tokenizer.next() orelse continue;
            const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue;
            perft.run(&pos, depth);
        }
        // custom
        if (eql(cmd, "qperft"))
        {
            const next = tokenizer.next() orelse continue;
            const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue;
            perft.qrun(&pos, depth);
        }
    }
}

fn eql(input: []const u8, comptime line: []const u8) bool
{
    return std.mem.eql(u8, input, line);
}


// info depth 8 seldepth 13 multipv 1 score cp 47 nodes 3436 nps 572666 hashfull 1 tbhits 0 time 6 pv e2e4 c7c5 b1c3 b8c6 g1f3 g8f6