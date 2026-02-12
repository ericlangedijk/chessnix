// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");
const Position = position.Position;
const Engine = search.Engine;
const hce = @import("hce.zig");

const Tokenizer = std.mem.TokenIterator(u8, .scalar);

const ctx = lib.ctx;
const io = lib.io;
const eql = funcs.eql;
const wtf = lib.wtf;

var engine: *Engine = undefined;

pub fn run() void {
    const is_tty: bool = lib.is_tty();
    uci_loop(is_tty) catch |err| {
        io.print("info error: {s}", .{ @errorName(err) });
    };
}

fn uci_loop(is_tty: bool) !void {
    if (is_tty) {
        // Enable cls and maybe later some fancy coloring.
        _ = std.fs.File.stdout().getOrEnableAnsiEscapeSupport();
        TTY.print_hello();
    }

    engine = try Engine.create(false);
    defer engine.destroy();

    command_loop: while (true) {
        const input = try io.readline() orelse continue :command_loop;
        if (input.len == 0) continue :command_loop;
        var tokenizer: Tokenizer = std.mem.tokenizeScalar(u8, input, ' ');
        const cmd: []const u8 = tokenizer.next() orelse continue :command_loop;

        // Uci commands.
        if (eql(cmd, "uci")) {
            UCI.respond_uciok();
        }
        else if (eql(cmd, "isready")) {
            UCI.respond_readyok();
        }
        else if (eql(cmd, "ucinewgame")) {
            try UCI.ucinewgame();
        }
        else if (eql(cmd, "go")) {
            try UCI.go(&tokenizer);
        }
        else if (eql(cmd, "stop")) {
            try UCI.stop();
        }
        else if (eql(cmd, "quit")) {
            try UCI.quit();
            return; // stop the program.
        }
        else if (eql(cmd, "setoption")) {
            try UCI.setoption(&tokenizer);
        }
        else if (eql(cmd, "position")) {
            try UCI.set_position(&tokenizer);
        }

        // Terminal only commands.
        else if (is_tty) {
            if (eql(cmd, "d")) {
                TTY.draw_position();
            }
            if (eql(cmd, "cls")) {
                TTY.cls();
            }
            else if (eql(cmd, "bench")) {
                try TTY.run_bench();
            }
            else if (eql(cmd, "perft")) {
                TTY.run_perft(&tokenizer, false);
            }
            else if (eql(cmd, "qperft")) {
                TTY.run_perft(&tokenizer, true);
            }
            else if (eql(cmd, "eval")) {
                TTY.eval();
            }
            else if (eql(cmd, "kiwi")) {
                try engine.set_position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", null);
            }
            else if (eql(cmd, "see")) {
                TTY.see(&tokenizer);
            }
            else if (eql(cmd, "state")) {
                TTY.print_state();
            }
        }
    }
}

/// Just a simple wrapper.
const UCI = struct {

    // option name Hash type spin default 16 min 1 max 33_554_432
    fn respond_uciok() void {
        io.print
        (
            \\id name chessnix {s}
            \\id author eric langedijk
            \\option name Hash type spin default {} min {} max {}
            \\option name UCI_Chess960 type check default false
            \\uciok
            \\
            ,
            .{
                lib.version,
                search.Options.default_hash_size, search.Options.min_hash_size, search.Options.max_hash_size,
            }
        );
    }

    fn respond_readyok() void {
        io.print("readyok\n", .{});
    }

    fn ucinewgame() !void {
        try engine.ucinewgame();
    }

    /// A go command must be eventually responded to with bestmove, once the search is completed or interrupted with stop.
    fn go(tokenizer: *Tokenizer) !void {
        const go_params: Go = try parse_go(tokenizer);
        try engine.go(&go_params);
    }

    // A stop command must return bestmove.
    fn stop() !void {
        try engine.stop();
    }

    fn quit() !void {
        try engine.quit();
    }

    /// Parse UCI command after "setoption".
    fn setoption(tokenizer: *Tokenizer) !void {
        // The tokens are setoption [name] [value] [requested-value]
        // name Hash value 32
        // name UCI_Chess960 value true

        const name_token: []const u8 = tokenizer.next() orelse return;
        if (!eql(name_token, "name")) return;

        const name: []const u8 = tokenizer.next() orelse return;

        const value_token: []const u8 = tokenizer.next() orelse return;
        if (!eql(value_token, "value")) return;

        const value: []const u8 = tokenizer.next() orelse return;

        if (eql(name, "Hash")) {
            const v: u64 = std.fmt.parseInt(u64, value, 10) catch return;
            engine.options.set_hash_size(v);
            try engine.apply_hash_size();
        }
        else if (eql(name, "UCI_Chess960")) {
            const v: bool = std.mem.eql(u8, value, "true");
            engine.options.set_chess_960(v);
        }
    }

    /// Parce UCI command after "position" (startpos or fen + optional moves).
    fn set_position(tokenizer: *Tokenizer) !void {
        const next = tokenizer.next() orelse return;
        if (eql(next, "fen")) {
            var fen_and_moves = std.mem.splitSequence(u8, tokenizer.rest(), "moves");
            try engine.set_position(fen_and_moves.next() orelse wtf(), fen_and_moves.next());
        }
        else if (eql(next, "startpos")) {
            if (tokenizer.next()) |n| {
                if (eql(n, "moves")) {
                    try engine.set_position(position.classic_startpos_fen, tokenizer.rest());
                }
            }
            else {
                try engine.set_position(position.classic_startpos_fen, null);
            }
        }
    }

    /// Parse UCI command after "go"
    fn parse_go(tokenizer: *Tokenizer) !Go {
        var result: Go = .empty;

        while (tokenizer.next()) |next| {
            if (eql(next, "wtime")) {
                result.time[0] = try parse_64(tokenizer.next());
            }
            else if (eql(next, "btime")) {
                result.time[1] = try parse_64(tokenizer.next());
            }
            else if (eql(next, "winc")) {
                result.increment[0] = try parse_64(tokenizer.next());
            }
            else if (eql(next, "binc")) {
                result.increment[1] = try parse_64(tokenizer.next());
            }
            else if (eql(next, "movestogo")) {
                result.movestogo = try parse_64(tokenizer.next());
            }
            else if (eql(next, "movetime")) {
                result.movetime = try parse_64(tokenizer.next());
            }
            else if (eql(next, "nodes")) {
                result.nodes = try parse_64(tokenizer.next());
            }
            else if (eql(next, "depth")) {
                result.depth = try parse_64(tokenizer.next());
            }
            else if (eql(next, "infinite")) {
                result.infinite = true;
            }
            else if (eql(next, "ponder")) {
                result.ponder = true;
            }
        }
        return result;
    }

    fn parse_64(str: ?[]const u8) Error!u64 {
        if (str == null) {
            return Error.ParsingError;
        }
        return std.fmt.parseInt(u64, str.?, 10) catch Error.ParsingError;
    }

    fn parse_32(str: ?[]const u8) Error!i32 {
        if (str == null) {
            return Error.ParsingError;
        }
        return std.fmt.parseInt(i32, str.?, 10) catch Error.ParsingError;
    }
};

/// Just a simple wrapper.
const TTY = struct {

    fn print_hello() void {
        io.print("chessnix {s} by eric langedijk\n", .{ lib.version });
    }

    fn cls() void {
        io.print("\x1B[2J\x1B[H", .{});
        print_hello();
    }

    fn draw_position() void {
        engine.pos.draw();
    }

    fn run_bench() !void {
        try perft.bench();
    }

    fn run_perft(tokenizer: *Tokenizer, fast: bool) void {
        var depth: u8 = 1;
        if (tokenizer.next()) |next| {
            depth = std.fmt.parseInt(u8, next, 10) catch 1;
        }
        if (!fast) {
            perft.run(&engine.pos, depth);
        }
        else {
            perft.qrun(&engine.pos, depth);
        }
    }

    fn eval() void {
        var ev: hce.Evaluator = .init();
        const e = ev.evaluate(&engine.pos);
        io.print("eval: {}\n", .{ e });
    }

    /// Input: "see move threshold"
    fn see(tokenizer: *Tokenizer) void {
        const move: []const u8 = tokenizer.next() orelse return;
        const threshold: []const u8 = tokenizer.next() orelse return;
        const t: i32 = std.fmt.parseInt(i32, threshold, 10) catch return;
        const result: bool = engine.see(move, t) catch return;
        io.print("{}\n", .{result});
    }

    fn print_state() void {
        io.print("engine busy: {}\n", .{ engine.is_busy()});
    }
};

/// The possible parameters for the go command. All fields not present in the go command are zero / false.
pub const Go = struct {
    /// The white and black time left.
    time: [2]u64,
    /// The white and black increment per move.
    increment: [2]u64,
    /// The number of moves until the next time control.
    movestogo: u64,
    /// The maximum depth to search.
    depth: u64,
    /// The maximum nodes to search.
    nodes: u64,
    /// The maximum time to search in milliseconds.
    movetime: u64,
    /// Infinite search. Overwrites the other limiting fields.
    infinite: bool = false,
    /// Not supported yet.
    ponder: bool = false,

    pub const empty: Go = .{
        .time = .{ 0, 0, },
        .increment = .{ 0, 0, },
        .movestogo = 0,
        .depth = 0,
        .nodes = 0,
        .movetime = 0,
        .infinite = false,
        .ponder = false,
    };
};

/// Error parsing UCI string.
const Error = error {
    ParsingError,
};
