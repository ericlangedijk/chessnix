// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const tt = @import("tt.zig");
const eval = @import("eval.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Position = position.Position;
const Engine = search.Engine;

const Tokenizer = std.mem.TokenIterator(u8, .scalar);

const ctx = lib.ctx;
const io = lib.io;
const eql = funcs.eql;
const wtf = lib.wtf;

var engine: *Engine = undefined;

pub fn run() void {
    // TODO: what to do in non-terminal mode? Just crash?
    uci_loop() catch |err| {
        //io.debugprint("error: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
        io.print("info error: {s}", .{ @errorName(err) });
        //_ = lib.in.readByte() catch {}; TODO: repair 0.15.1
    };
}

fn uci_loop() !void {
    engine = try Engine.create();
    defer engine.destroy();

    const is_tty = lib.is_tty();

    if (is_tty) {
        io.print("chessnix {s} by eric\n", .{ lib.version });
    }

    command_loop: while (true) {
        const input = try io.readline() orelse continue :command_loop; // continue, break?
        if (input.len == 0) continue :command_loop; // continue, break?
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
            try UCI.stop(); // TODO: engine still outputs bestmove.
            return;
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
                TTY.do_static_eval();
            }
            else if (eql(cmd, "state")) {
                TTY.print_state();
            }
        }
    }
}

/// Just a simple wrapper.
const UCI = struct {

    fn respond_uciok() void {
        // try io.print
        // (
        //     \\id chessnix {s}
        //     \\id author eric
        //     \\option name Hash spin default {} min {} max {}
        //     \\uciok
        //     \\
        //     ,
        //     .{
        //         lib.version,
        //         search.Options.default_hash_size, search.Options.min_hash_size, search.Options.max_hash_size,
        //     }
        // );
        io.print
        (
            \\id chessnix {s}
            \\id author eric
            \\uciok
            \\
            ,
            .{
                lib.version,
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
        try engine.start(&go_params);
    }

    // A stop command must return bestmove.
    fn stop() !void {
        try engine.stop();
    }

    /// Parse UCI command after "setoption"
    fn setoption(tokenizer: *Tokenizer) !void {
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
        // io.debugprint("{any}\n", .{ engine.options});
    }

    /// Parce UCI command after "position"
    fn set_position(tokenizer: *Tokenizer) !void {
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
    fn parse_go(tokenizer: *Tokenizer) !Go {
        var result: Go = .empty;

        while (tokenizer.next()) |next| {
            if (eql(next, "ponder")) {
                result.ponder = true;
            }
            else if (eql(next, "wtime")) {
                result.wtime = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "btime")) {
                result.btime = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "winc")) {
                result.winc = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "binc")) {
                result.binc = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "movestogo")) {
                result.movestogo = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "movetime")) {
                result.movetime = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "nodes")) {
                result.nodes = parse_64(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "depth")) {
                result.depth = parse_32(tokenizer.next()) orelse return Error.ParsingError;
            }
            else if (eql(next, "infinite")) {
                result.infinite = true;
            }
    }
    return result;
    }

    fn parse_64(str: ?[]const u8) ?u64 {
        if (str == null) return null;
        return std.fmt.parseInt(u64, str.?, 10) catch null;
    }

    fn parse_32(str: ?[]const u8) ?u32 {
        if (str == null) return null;
        return std.fmt.parseInt(u32, str.?, 10) catch null;
    }
};

/// Just a simple wrapper.
const TTY = struct {

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

    fn do_static_eval() void {
        const e = eval.evaluate_with_tracking_absolute(&engine.pos, &engine.evaltranspositiontable, &engine.pawntranspositiontable);
        io.print("eval abs = {} phase {}\n", .{ e, engine.pos.phase() });

        //const v = eval.tuned_eval(&engine.pos);
        //io.debugprint("e {} t {}\n", .{e, v});
        //eval.bench(&engine.pos, &engine.evaltranspositiontable);
        //io.debugprint("eval hits {}\n", .{ engine.evaltranspositiontable.hits });
    }

    fn print_state() void {
        //const t: *const tt.TranspositionTable = &engine.transpositiontable;

// position fen r5k1/pbN2rp1/4Q1Np/2pn1pB1/8/P7/1PP2PPP/6K1 b - - 0 25 moves d5c7 g6e7 g8f8 e7g6 f8g8 g6e7 g8f8 e7g6 f8g8 g6e7 g8f8 e7g6 f8g8

        //io.debugprint("upcoming rep {}\n", .{engine.pos.is_repetition()});
        //io.debugprint("threefold rep {}\n", .{engine.pos.is_threefold_repetition()});

        //_ = eval.eval2(&engine.pos, .Endgame);

        //io.debugprint("engine thinking {}\n", .{engine.is_busy()});
        //io.debugprint("flags {}\n", .{ engine.pos.gen_flags });
        io.print("draw by material {}\n", .{ eval.is_draw_by_insufficient_material(&engine.pos) });
        //io.print("3 rep {}\n", .{ engine.pos.is_threefold_repetition() });
        //io.print("1 rep {}\n", .{ engine.pos.is_upcoming_repetition() });
        //io.debugprint("engine controller thread active {}\n", .{engine.controller_thread != null});
        //io.debugprint("engine search thread active {}\n", .{engine.search_thread != null});

        //io.debugprint("tt MB {}\n", .{t.mb});
        //io.debugprint("tt slots {}\n", .{t.len});
        //io.debugprint("tt filled {}\n", .{t.filled});
        //io.debugprint("tt permille {}\n", .{t.permille()});
        //io.debugprint("tt age {}\n", .{t.age});

//        const s = &engine.searcher;

        io.print("eval hits {}\n", .{ engine.evaltranspositiontable.hits });

        //io.debugprint("{}\n", .{});
        // io.debugprint("tt probes {}\n", .{ s.transpositiontable.probes });
        // io.debugprint("tt hits   {}\n", .{ s.transpositiontable.hits });
        // io.debugprint("nodes     {}\n", .{ s.processed_nodes });
        // io.debugprint("quiets    {}\n", .{ s.processed_quiescence_nodes });
       // engine.pos.print_history();
    }
};

pub const Go = struct {
    /// Not used (yet).
    ponder: ?bool = null,
    /// Not used (yet).
    wtime: ?u32 = null,
    /// Not used (yet).
    btime: ?u32 = null,
    /// Not used (yet).
    winc: ?u32 = null,
    /// Not used (yet).
    binc: ?u32 = null,
    /// Not used (yet).
    movestogo: ?u32 = null,
    /// The maximum depth to search.
    depth: ?u32 = null,
    /// The maximum nodes to search.
    nodes: ?u64 = null,
    /// The maximum time to search in milliseconds.
    movetime: ?u32 = null,
    /// Infinite search. Overwrites the other limiting fields.
    infinite: ?bool = null,

    pub const empty: Go = .{};
};

/// Error parsing UCI string.
const Error = error {
    ParsingError,
};
