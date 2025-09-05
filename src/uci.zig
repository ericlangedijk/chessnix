// zig fmt: off

// TODO: when encountering an illegal fen / moves always return a bestmove = '0000' indicating invalid state.
// TODO: report invalid input with: info string <str>
// TODO: log or report on crash?

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const bounded_array = @import("bounded_array.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const engine = @import("engine.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");
const eval = @import("eval.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Position = position.Position;
const Tokenizer = std.mem.TokenIterator(u8, .scalar);

const ctx = lib.ctx;
const io = lib.io;

const eql = funcs.eql;

pub fn run() void {
    // TODO: what to do in non-terminal mode? Just crash?
    uci_loop() catch |err| {
        lib.io.debugprint("ERROR: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
        //_ = lib.in.readByte() catch {}; TODO: repair 0.15.1
    };
}

fn uci_loop() !void
{
    try engine.initialize();
    defer engine.finalize();

    const is_tty = lib.is_tty();

    if (is_tty) {
        try io.print("chessnix {s} by eric\n", .{ lib.version });
    }

    command_loop: while (true) {
        const input = try io.readline() orelse break :command_loop;
        var tokenizer: Tokenizer = std.mem.tokenizeScalar(u8, input, ' ');
        const cmd: []const u8 = tokenizer.next() orelse continue :command_loop;

        // Uci commands.
        if (eql(cmd, "uci")) {
            try print_uciok();
        }
        else if (eql(cmd, "isready")) {
            try io.print("readyok\n", .{});
        }
        else if (eql(cmd, "ucinewgame")) {
            engine.set_startpos();
            engine.clear_for_new_game();
        }
        else if (eql(cmd, "go")) {
            const go: Go = parse_go(&tokenizer) catch continue :command_loop;
            try engine.start(&go); // Each go command must be eventually responded to with bestmove, once the search is completed or interrupted with stop.
        }
        else if (eql(cmd, "stop")) {
            try engine.stop(); // Return bestmove if we were thinking, otherwise ignore.
        }
        else if (eql(cmd, "setoption")) {
            try parse_and_set_option(&tokenizer);
        }
        else if (eql(cmd, "position")) {
            parse_and_set_position(&tokenizer) catch |err| { print_error(err); continue :command_loop; };
        }
        else if (eql(cmd, "quit")) {
            return;
        }
        // Custom commands in terminal.
        else if (is_tty) {
            if (eql(cmd, "d")) {
                try engine.pos.draw();
            }
            else if (eql(cmd, "bench")) {
                try perft.bench();
            }
            else if (eql(cmd, "perft")) {
                var depth: u8 = 1;
                if (tokenizer.next()) |next| {
                    depth = std.fmt.parseInt(u8, next, 10) catch 1;
                }
                perft.run(&engine.pos, depth);
            }
            else if (eql(cmd, "qperft")) {
                var depth: u8 = 1;
                if (tokenizer.next()) |next| {
                    depth = std.fmt.parseInt(u8, next, 10) catch 1;
                }
                perft.qrun(&engine.pos, depth);
            }
            // DEBUG TEMP
            else if (eql(cmd, "kiwi")) {
                try engine.set_position(tests.kiwi_fen, null);
            }
            else if (eql(cmd, "state")) {
                print_state();
            }
            else if (eql(cmd, "deb")) {
                try temp();
            }
            // DEBUG TEMP
            else if (eql(cmd, "c")) {
                const next = tokenizer.next() orelse continue :command_loop;
                const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue :command_loop;
                perft.run_captures(&engine.pos, depth);
            }
            // DEBUG TEMP
            else if (eql(cmd, "r")) {
                _ = engine.pos.is_threefold_repetition();
            }
            // DEBUG TEMP
            else if (eql(cmd, "m")) {
                try engine.pos.print_history();
            }
            // DEBUG TEMP
            else if (eql(cmd, "e")) {
                const e = eval.evaluate_abs(&engine.pos, true);
                try io.print("eval abs = {}\n", .{ e });
                // io.debugprint("{t}\n", .{ engine.pos.phase()});
                // io.debugprint("{}\n", .{ engine.pos.non_pawn_material()});
                // io.debugprint("{any}\n", .{ engine.pos.materials});
                //eval.bench(&engine.pos);
            }
        }
    }
}

/// TODO: Not sure yet how and what handling errors in ucimode / terminal mode.
pub fn print_error(err: anyerror) void {
    lib.io.print("info string error: {t}\n", .{ err }) catch lib.wtf();
}

fn print_uciok() !void {
    try io.print
    (
        \\id chessnix {s}
        \\id author eric
        \\option name Hash spin default {} min {} max {}
        \\uciok
        \\
        ,
        .{
            lib.version,
            engine.Options.default_hash_size, engine.Options.min_hash_size, engine.Options.max_hash_size,
        }
    );
}

/// Parse UCI command after "setoption"
fn parse_and_set_option(tokenizer: *Tokenizer) !void {
    // name Hash value 32
    const name_token: []const u8 = tokenizer.next() orelse return;
    if (!eql(name_token, "name")) return;

    const name: []const u8 = tokenizer.next() orelse return;

    const value_token: []const u8 = tokenizer.next() orelse return;
    if (!eql(value_token, "value")) return;

    const value: []const u8 = tokenizer.next() orelse return;
    if (eql(name, "Hash")) {
        const v: u64 = std.fmt.parseInt(u64, value, 10) catch return;
        engine.options.set_hash_size(v);
    }
    io.debugprint("{any}\n", .{ engine.options});
}

/// Parce UCI command after "position"
fn parse_and_set_position(tokenizer: *Tokenizer) !void {
    const next = tokenizer.next() orelse return;
    if (eql(next, "fen")) {
        var fen_and_moves = std.mem.splitSequence(u8, tokenizer.rest(), "moves");
        try engine.set_position(fen_and_moves.next(), fen_and_moves.next());
    }
    else if (eql(next, "startpos")) {
        if (tokenizer.next()) |n| {
            if (eql(n, "moves")) try engine.set_startpos_with_optional_moves(tokenizer.rest());
        }
        else {
            engine.set_startpos();
        }
    }
}

/// Parse UCI command after "go"
fn parse_go(tokenizer: *Tokenizer) !Go
{
    var go: Go = .empty;

    while (tokenizer.next()) |next| {
        if (eql(next, "ponder")) {
            go.ponder = true;
        }
        else if (eql(next, "wtime")) {
            go.wtime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "btime")) {
            go.btime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "winc")) {
            go.winc = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "binc")) {
            go.binc = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "movestogo")) {
            go.movestogo = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "movetime")) {
            go.movetime = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "depth")) {
            go.depth = parse_nr(tokenizer.next()) orelse return Error.ParsingError;
        }
        else if (eql(next, "infinite")) {
            go.infinite = true;
        }
   }
   return go;
}

fn parse_nr(str: ?[]const u8) ?u64 {
    if (str == null) return null;
    return std.fmt.parseInt(u64, str.?, 10) catch null;
}

pub const Go = struct {
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

    const empty: Go = .{};
};

/// Error parsing UCI string.
const Error = error {
    ParsingError,
};

/// Just some stupid debug function.
fn temp() !void {

    try io.print("{f}\n", .{&engine.pos});
    if (true) return;
    //const fen = "8/kq6/2b5/3p4/4P3/5B2/6QK/8 w - - 0 1";
    //const fen = "8/kb6/2p5/3p4/4Q3/5B2/6QK/8 w - - 0 1"; // BAD
    //const fen = "8/kb6/2p5/3q4/4B3/5B2/6QK/8 w - - 0 1";
    //const fen = "3r4/1q1r4/2b5/2Kpk3/4P3/5B2/3R2Q1/3R4 w - - 0 1";
    //const fen = "7k/1q6/2p5/3p4/4P3/8/6B1/K6B w - - 0 1"; // GOOD e4xd5
    const fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

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

/// Another stupid debug function.
fn print_state() void {
    const t: *const tt.TranspositionTable = &engine.searchmgr.transpositiontable;
    io.debugprint("tt slots {}\n", .{t.len});
    io.debugprint("tt filled {}\n", .{t.filled});
    io.debugprint("tt permille {}\n", .{t.permille()});
    // io.debugprint("\n", .{});
    // io.debugprint("\n", .{});
    // io.debugprint("\n", .{});
}
