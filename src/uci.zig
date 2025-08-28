// zig fmt: off

// TODO: when encountering an illegal fen / moves always return a bestmove = '0000' indicating invalid state.

const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const bounded_array = @import("bounded_array.zig");
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
const io = lib.io;

const eql = funcs.eql;

pub fn run() void
{
    // TODO: what to do in non-terminal mode? Just crash?
    uci_loop() catch |err|
    {
        std.debug.print("ERROR: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
        //_ = lib.in.readByte() catch {}; TODO: repair 0.15.1
    };
}

fn uci_loop() !void
{
    try engine.initialize();
    defer engine.finalize();

    const is_tty = lib.is_tty();

    if (is_tty)
    {
        try io.print("chessnix {s} by eric\n", .{ lib.version });
    }

    command_loop: while (true)
    {
        const input = try io.readline() orelse break;
        var tokenizer: Tokenizer = std.mem.tokenizeScalar(u8, input, ' ');
        const cmd: []const u8 = tokenizer.next() orelse continue :command_loop;

        // Uci commands.
        if (eql(cmd, "uci"))
        {
            try io.print
            (
                \\id chessnix {s}
                \\nauthor eric
                \\uciok
                \\
                , .{ lib.version }
            );
        }
        else if (eql(cmd, "isready"))
        {
            try io.print("readyok\n", .{});
        }
        else if (eql(cmd, "ucinewgame"))
        {
            try engine.set_startpos(null);
        }
        else if (eql(cmd, "go"))
        {
            const go: Go = parse_go(&tokenizer) catch continue :command_loop;
            try io.print("{any}\n", .{ go });
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
            parse_position(&tokenizer) catch continue :command_loop;
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
                try perft.bench();
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
                try temp();
            }
            // DEBUG TEMP
            else if (eql(cmd, "r"))
            {
                _ = engine.pos.is_threefold_repetition();
            }
            // DEBUG TEMP
            else if (eql(cmd, "m"))
            {
                try engine.pos.print_history();
            }
            // DEBUG TEMP
            else if (eql(cmd, "e"))
            {
                const e = eval.lazy_evaluate(&engine.pos, true);
                try io.print("eval = {}\n", .{ e });
            }
            // DEBUG TEMP
            else if (eql(cmd, "f"))
            {
                var tempstate: position.StateInfo = undefined;
                try engine.pos.print();
                const e1 = eval.lazy_evaluate(&engine.pos, false);
                try io.print("eval = {}\n", .{ e1 });
                engine.pos.flip(&tempstate);
                try engine.pos.print();
                const e2 = eval.lazy_evaluate(&engine.pos, false);
                try io.print("eval = {}\n", .{ e2 });
            }
        }
    }
}

/// Parce uci command after "position"
fn parse_position(tokenizer: *Tokenizer) !void
{
    const next = tokenizer.next() orelse return;
    if (eql(next, "fen"))
    {
        var fen_and_moves = std.mem.splitSequence(u8, tokenizer.rest(), "moves");
        try engine.set_position(fen_and_moves.next(), fen_and_moves.next());
    }
    else if (eql(next, "startpos"))
    {
        if (tokenizer.next()) |n|
        {
            if (eql(n, "moves")) try engine.set_startpos(tokenizer.rest());
        }
        else
        {
            try engine.set_startpos(null);
        }
    }
}

/// Parse uci after "go"
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

/// Error parsing UCI string.
const Error = error
{
    ParsingError,
};

/// Just some stupid debug function.
fn temp() !void
{
}