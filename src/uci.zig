// zig fmt: off
const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const typoes = @import("types.zig");
const position = @import("position.zig");
const engine = @import("engine.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Move = typoes.Move;
const Position = position.Position;

const ctx = lib.ctx;

pub fn run() !void
{
    const in = lib.in;
    const out = lib.out;

    try engine.initialize();

    if (lib.is_tty())
    {
        try out.print("chessnix {s} by eric\n", .{lib.version});
    }

    var buffer: [4096]u8 = @splat(0);

    uci_loop: while (true)
    {
        const line = try in.readUntilDelimiter(&buffer, '\n');
        const input: []const u8 = std.mem.trim(u8, line, "\r");
        if (input.len == 0) continue;
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
            //engine.set_startpos();
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
            const next = tokenizer.next() orelse continue :uci_loop;
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
        // custom
        else if (eql(cmd, "d"))
        {
            try position.print_pos(&engine.pos);
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
            perft.run(&engine.pos, depth);
        }
        // custom
        if (eql(cmd, "qperft"))
        {
            const next = tokenizer.next() orelse continue;
            const depth: u8 = std.fmt.parseInt(u8, next, 10) catch continue;
            perft.qrun(&engine.pos, depth);
        }
    }
}

fn set_position() void
{

}

fn eql(input: []const u8, comptime line: []const u8) bool
{
    return std.mem.eql(u8, input, line);
}
