// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const engine = @import("engine.zig");
const search = @import("search.zig");
const eval = @import("eval.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Position = position.Position;
const Tokenizer = std.mem.TokenIterator(u8, .scalar);

const ctx = lib.ctx;

pub fn run() void
{
    // TODO: what to do in non-terminal mode? Just crash?
    uci_loop() catch |err|
    {
        std.debug.print("ERROR: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
        _ = lib.in.readByte() catch {};
    };
}

fn uci_loop() !void
{
    const in = lib.in;
    const out = lib.out;
    const is_tty = lib.is_tty();

    try engine.initialize();
    defer engine.finalize();

    if (is_tty)
    {
        try out.print("chessnix {s} by eric\n", .{ lib.version });
    }

    var buffer: [4096]u8 = @splat(0);

    command_loop: while (true)
    {
        const line = try in.readUntilDelimiter(&buffer, '\n');
        const input: []const u8 = std.mem.trim(u8, line, "\r");
        if (input.len == 0) continue;
        var tokenizer: Tokenizer = std.mem.tokenizeScalar(u8, input, ' ');
        const cmd: []const u8 = tokenizer.next() orelse continue :command_loop;

        // Uci commands.
        if (eql(cmd, "uci"))
        {
            try out.print("id chessnix {s}\nauthor eric\nuciok\n", .{ lib.version });
        }
        else if (eql(cmd, "isready"))
        {
            try out.print("readyok\n", .{});
        }
        else if (eql(cmd, "ucinewgame"))
        {
            try engine.set_startpos();
        }
        else if (eql(cmd, "go"))
        {
            const go: Go = parse_go(&tokenizer) catch continue :command_loop;
            try lib.out.print("{any}\n", .{ go });
            try engine.start();
            // Each go command must be eventually responded to with bestmove, once the search is completed or interrupted with stop.
        }
        else if (eql(cmd, "stop"))
        {
            try engine.stop();
            // Return bestmove if we were thinking, otherwise ignore.
        }
        else if (eql(cmd, "position"))
        {
            const next = tokenizer.next() orelse continue :command_loop;
            if (eql(next, "fen"))
            {
                var fen_and_moves = std.mem.splitSequence(u8, tokenizer.rest(), "moves");
                try engine.set_position(fen_and_moves.next(), fen_and_moves.next());
            }
            else if (eql(next, "startpos"))
            {
                var moves = std.mem.splitSequence(u8, tokenizer.rest(), "moves");
                try engine.set_position(position.fen_classic_startpos, moves.next());
            }
        }
        else if (eql(cmd, "quit"))
        {
            return;
        }

        // Custom commands in terminal.
        else if (is_tty)
        {
            if (eql(cmd, "d"))
            {
                try engine.pos.print();
            }
            else if (eql(cmd, "bench"))
            {
                try tests.bench();
            }
            else if (eql(cmd, "perft"))
            {
                const next = tokenizer.next() orelse continue :command_loop;
                const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue :command_loop;
                perft.run(&engine.pos, depth);
            }
            else if (eql(cmd, "qperft"))
            {
                const next = tokenizer.next() orelse continue :command_loop;
                const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue :command_loop;
                perft.qrun(&engine.pos, depth);
            }
            // DEBUG TEMP
            else if (eql(cmd, "deb"))
            {
                try lib.out.print("size of Stack {}\n", .{@sizeOf(search.Stack)});
                try lib.out.print("size of SearchManager {}\n", .{@sizeOf(search.SearchManager)});
            }
            // DEBUG TEMP
            else if (eql(cmd, "m"))
            {
                try engine.pos.print_history();
            }
            // DEBUG TEMP
            else if (eql(cmd, "e"))
            {
                const e = eval.lazy_evaluate(&engine.pos);
                try lib.out.print("eval = {}\n", .{ e });
            }
        }
    }
}

fn parse_go(tokenizer: *Tokenizer) !Go
{
    var go: Go = .empty;

    while (tokenizer.next()) |next|
    {
        if (eql(next, "ponder"))
        {
            go.ponder = true;
        }
        else if (eql(next, "wtime"))
        {
            go.wtime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "btime"))
        {
            go.btime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "winc"))
        {
            go.winc = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "binc"))
        {
            go.binc = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "movestogo"))
        {
            go.movestogo = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "movetime"))
        {
            go.movetime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "infinite"))
        {
            go.infinite = true;
        }
   }
   return go;
}

fn parse_nr(str: ?[]const u8) ?u64
{
    if (str == null) return null;
    return std.fmt.parseInt(u64, str.?, 10) catch null;
}

fn eql(input: []const u8, comptime line: []const u8) bool
{
    return std.mem.eql(u8, input, line);
}

pub const Go = struct
{
    const empty: Go = .{};

    ponder: ?bool = null,
    wtime: ?u64 = null,
    btime: ?u64 = null,
    winc: ?u64 = null,
    binc: ?u64 = null,
    movestogo: ?u64 = null,
    depth: ?u64 = null,
    nodes: ?u64 = null,
    movetime: ?u64 = null,
    infinite: ?bool = null,
};

const Error = error
{
    ParsingError,
};