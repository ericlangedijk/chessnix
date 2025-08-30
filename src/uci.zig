// zig fmt: off

// TODO: when encountering an illegal fen / moves always return a bestmove = '0000' indicating invalid state.
// TODO: report invalid input with: info string <str>
// TODO: log or report on crash?

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
        lib.io.debugprint("ERROR: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
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
            //try io.print("{any}\n", .{ go });
            try engine.start(&go);
            // Each go command must be eventually responded to with bestmove, once the search is completed or interrupted with stop.
        }
        else if (eql(cmd, "stop"))
        {
            try engine.stop();
            // Return bestmove if we were thinking, otherwise ignore.
        }
        else if (eql(cmd, "position"))
        {
            parse_position(&tokenizer) catch |err| { print_error(err); continue :command_loop; };
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
            else if (eql(cmd, "c"))
            {
                const next = tokenizer.next() orelse continue :command_loop;
                const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue :command_loop;
                perft.run_captures(&engine.pos, depth);
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
                const e = eval.evaluate_abs(&engine.pos, true);
                try io.print("eval abs = {}\n", .{ e });
                //eval.bench(&engine.pos);
            }
            // DEBUG TEMP
            else if (eql(cmd, "f"))
            {
                var tempstate: position.StateInfo = undefined;
                try engine.pos.print();
                const e1 = eval.evaluate(&engine.pos, false);
                try io.print("eval = {}\n", .{ e1 });
                engine.pos.flip(&tempstate);
                try engine.pos.print();
                const e2 = eval.evaluate(&engine.pos, false);
                try io.print("eval = {}\n", .{ e2 });
            }
        }
    }
}

/// TODO: Not sure yet how and what handling errors in ucimode / terminal mode.
pub fn print_error(err: anyerror) void
{
    lib.io.print("info string error: {t}\n", .{ err }) catch lib.wtf();
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
        else if (eql(next, "depth"))
        {
            go.depth = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
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

    /// Not used (yet).
    ponder: ?bool = null,
    /// Not used (yet).
    wtime: ?u64 = null,
    /// Not used (yet).
    btime: ?u64 = null,
    /// Not used (yet).
    winc: ?u64 = null,
    /// Not used (yet).
    binc: ?u64 = null,
    /// Not used (yet).
    movestogo: ?u64 = null,
    /// The maximum depth to search.
    depth: ?u64 = null,
    /// The maximum nodes to search.
    nodes: ?u64 = null,
    /// The maximum time to search in milliseconds.
    movetime: ?u64 = null,
    /// Infinite search. Overwrites the other limiting fields.
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
    //const fen = "8/kq6/2b5/3p4/4P3/5B2/6QK/8 w - - 0 1";
    const fen = "8/kb6/2p5/3p4/4Q3/5B2/6QK/8 w - - 0 1"; // BAD
    //const fen = "8/kb6/2p5/3q4/4B3/5B2/6QK/8 w - - 0 1";
    //const fen = "3r4/1q1r4/2b5/2Kpk3/4P3/5B2/3R2Q1/3R4 w - - 0 1";
    //const fen = "7k/1q6/2p5/3p4/4P3/8/6B1/K6B w - - 0 1"; // GOOD e4xd5
    //const fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    try engine.set_position(fen, null);
    const m = types.Move.create(.E4, .D5);

    // const occ = engine.pos.all() ^ m.to.to_bitboard() ^ m.from.to_bitboard();
    // funcs.print_bitboard(occ);

    // const bb = engine.pos.attackers_to_for_occupation(occ, .D5, .WHITE);

    //funcs.print_bitboard(bb);
    const good = eval.see(&engine.pos, m);
    const v = eval.see_score(&engine.pos, m);
    lib.io.debugprint("good capture = {}, see = {}\n", .{good, v});
}