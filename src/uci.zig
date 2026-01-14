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
    // TODO: what to do in non-terminal mode? Just crash?
    uci_loop() catch |err| {
        //io.debugprint("error: {s}.\n\nPress any key to quit.\n", .{ @errorName(err) });
        io.print("info error: {s}", .{ @errorName(err) });
        //_ = lib.in.readByte() catch {}; TODO: repair 0.15.1
    };
}

fn uci_loop() !void {
    const is_tty = lib.is_tty();
    if (is_tty) {
        io.print("chessnix {s} by eric langedijk\n", .{ lib.version });
    }

    engine = try Engine.create(false);
    defer engine.destroy();

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
            // else  if (eql(cmd, "cls")) {
            //     io.print("{}", .{ "\x1B[2J\x1B[H" });
            // }
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
        try engine.start(&go_params);
    }

    // A stop command must return bestmove.
    fn stop() !void {
        try engine.stop();
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


    // fn parse_32(str: ?[]const u8) ?u32 {
    //     if (str == null) return null;
    //     return std.fmt.parseInt(u32, str.?, 10) catch null;
    // }
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

        //const e = eval.evaluate_with_tracking_absolute(&engine.pos, &engine.evaltranspositiontable, &engine.pawntranspositiontable);
        //io.print("eval abs = {} phase {}\n", .{ e, engine.pos.phase() });

        //const bb = engine.pos.all_except_pawns_and_kings();
        //funcs.print_bitboard(bb);

        var ev: hce.Evaluator = .init();
        const e = ev.evaluate(&engine.pos);
        io.print("eval: {}\n", .{ e });

        //const v = eval.tuned_eval(&engine.pos);
        //io.debugprint("e {} t {}\n", .{e, v});
        //eval.bench(&engine.pos, &engine.evaltranspositiontable);
        //io.debugprint("eval hits {}\n", .{ engine.evaltranspositiontable.hits });
    }

    fn print_state() void {

        // for (types.PieceType.all) |pt| {
        //     funcs.print_bitboard(engine.pos.threats[pt.u]);
        // }

        // io.print("evals: {}, evalhits: {}\n", .{ search.EVALS, search.EVAL_HITS });

        //io.print("{}\n", .{ engine.pos.threats});
        // @import("history.zig").debug_hist(&engine.searcher.history_heuristics, &engine.pos);
        // if (true) return;


        //if (true) return;

        // engine.set_position("1kr4r/1p1nnp2/1P1qb2p/p1pp4/P2P1NpN/2QBP1P1/5PP1/2R2RK1 w - - 0 24", null) catch return;
        // engine.set_position("1kr5/1p3p2/1P1qb2p/p1pp4/P2P2pN/2Q1P1P1/5PP1/2R3K1 w - - 0 24", null) catch return;

        // engine.set_position("5k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ -", null) catch return;

        // engine.set_position("r1b1kbnr/pppp1ppp/2n5/8/3PP2q/2N2N2/PPP1K1pP/R1BQ1B1R b kq - 1 7", null) catch return;

        // const m: types.Move = engine.pos.parse_move("g2h1q") catch return;

        // //const m: types.Move = .create(types.Square.C3, types.Square.A5, types.Move.capture);
        // const see = hce.see_score(&engine.pos, m);
        // //const see2 = hce.see_score_phased(&engine.pos, m);
        // //draw_position();

        // io.print("see score {}\n", .{ see });
        // //io.print("see score phased {}", .{ see2 });
        // if (true) return;



        //const t: *const tt.TranspositionTable = &engine.transpositiontable;

        // position fen r5k1/pbN2rp1/4Q1Np/2pn1pB1/8/P7/1PP2PPP/6K1 b - - 0 25 moves d5c7 g6e7 g8f8 e7g6 f8g8 g6e7 g8f8 e7g6 f8g8 g6e7 g8f8 e7g6 f8g8

        //io.debugprint("upcoming rep {}\n", .{engine.pos.is_repetition()});
        //io.debugprint("threefold rep {}\n", .{engine.pos.is_threefold_repetition()});

        //_ = eval.eval2(&engine.pos, .Endgame);

        //io.debugprint("engine thinking {}\n", .{engine.is_busy()});
        //io.debugprint("flags {}\n", .{ engine.pos.gen_flags });
        //io.print("draw by material {}\n", .{ eval.is_draw_by_insufficient_material(&engine.pos) });
        //io.print("3 rep {}\n", .{ engine.pos.is_threefold_repetition() });
        //io.print("1 rep {}\n", .{ engine.pos.is_upcoming_repetition() });
        //io.debugprint("engine controller thread active {}\n", .{engine.controller_thread != null});
        //io.debugprint("engine search thread active {}\n", .{engine.search_thread != null});

        //io.debugprint("tt MB {}\n", .{t.mb});
        //io.debugprint("tt slots {}\n", .{t.len});
        //io.debugprint("tt filled {}\n", .{t.filled});
        //io.debugprint("tt permille {}\n", .{t.permille()});
        //io.debugprint("tt age {}\n", .{t.age});

//       const s = engine;

        //io.print("eval hits {}\n", .{ engine.evaltranspositiontable.hits });
        //io.print("is draw {}\n", .{ engine.pos.is_draw_by_insufficient_material() });

        //io.print("sizeof ttentry {}\n", .{ @sizeOf(tt.Entry) });
        //io.print("sizeof evalentry {}\n", .{ @sizeOf(tt.EvalEntry) });
        //io.print("sizeof pawnentry {}\n", .{ @sizeOf(tt.PawnEntry) });

        // //io.debugprint("{}\n", .{});
        // io.print("tt len {}\n", .{ s.transpositiontable.hash.data.len });
        // io.print("eval tt len {}\n", .{ s.evaltranspositiontable.hash.data.len });
        // //io.print("pawneval len {}\n", .{ s.pawntranspositiontable.hash.data.len });

        // const a = s.transpositiontable.hash.percentage_filled();
        // const b = s.evaltranspositiontable.hash.percentage_filled();
        // //const c = s.pawntranspositiontable.hash.percentage_filled();

        // io.print("filled tt {}% eval {}%\n", .{ a, b });

        //s.searcher.history_heuristics.print_state();
        // io.debugprint("quiets    {}\n", .{ s.processed_quiescence_nodes });
        // engine.pos.print_history();

        // if (comptime lib.is_paranoid) {
        //     for (self.transpositiontable.hash.data) |e| {
        //         assert(e != tt.Entry.empty);
        //     }
        // }
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
