// zig fmt: off

//! Converts my local lichess json eval dataset to multiple viri-files.
//!
//! Source: C:/Data/chess/lichess/lichess_db_eval.jsonl.
//! Dest: C:/Data/chess/lichess/datasets files (0000.epd...nnnn.vf)
//! Multiple .vf files are created with a fen + score + pv   :"8/p5b1/1p5p/4k1p1/2Pp1pP1/1P3K2/P2B3P/8 w - - 0 1 cp -254 pv ..."
//! Skipped: invalid and ridiculous positions.
//! Skipped: eval scores outside the range -10000...10000.
//! Skipped: positions that are in check.
//! Skipped: positions where the best move is a capture.

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const scoring = @import("scoring.zig");
const viri = @import("viri.zig");

const ctx = lib.ctx;

const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;

const src_file = "C:/Data/chess/lichess/lichess_db_eval.jsonl";
const dst_folder = "C:/Data/chess/lichess/datasets/";

pub const Settings = struct {
    // src file
    // dst folder
    // base name
    // positions per file
    // max to write positions.
};

/// Our final conversion result.
const Record = struct {
    fen: []const u8,
    cp: i32,
    pv: []const u8,
};

/// Json
const JsonPv = struct {
    cp: ?i32 = null,
    mate: ?i32 = null,
    line: []const u8,
};

/// Json
const JsonEval = struct {
    pvs: []JsonPv,
    knodes: u64,
    depth: u32,
};

/// Json
const JsonEntry = struct {
    fen: []const u8,
    evals: []JsonEval,

    fn extract_record(self: *const JsonEntry) ?Record {
        if (self.evals.len == 0) {
            return null;
        }
        // We take the first one
        const first_eval: JsonEval = self.evals[0];
        // And we take the first pvs.
        if (first_eval.pvs.len == 0) {
            return null;
        }
        const pv: JsonPv = first_eval.pvs[0];
        // Ensure one of the scores is filled.
        if (pv.cp == null and pv.mate == null) {
            return null;
        }
        const score: i32 = if (pv.mate != null) scoring.mate - pv.mate.? else pv.cp.?;
        return .{
            .fen = self.fen,
            .cp = score,
            .pv = pv.line,
        };
    }
};

const Stats = struct {
    source_filesize: u64,
    positions_processed: u64,
    positions_written: u64,
    mates_skipped: u64,
    pv_capture_skipped: u64,
    in_check_skipped: u64,
    eval_bounds_skipped: u64,
    nonsense_positions_skipped: u64,
    pv_empty: u64,
    pv_illegal: u64,
    max_eval: i32,
    min_eval: i32,
    by_stm_count: [2]u64,
    by_piece_count: [33]u64,
};

const PV = types.ExtMoveList(64);

var str_buf: [256]u8 = undefined;

fn to_str(comptime fmt: []const u8, args: anytype) []u8 {
    return std.fmt.bufPrint(&str_buf, fmt, args) catch lib.wtf("to_str", .{});
}

fn compute_viri_filename(file_nr: usize) []u8 {
    return to_str("{s}{:0>4}.vf", .{ dst_folder, file_nr });
}

pub fn convert_lichess_dataset_to_viri() !void {
    // In debug mode only generate a small amount.
    const gpa = ctx.gpa;
    const max: ?usize = if (lib.is_debug) 10000 else null;//100_000_000;
    const max_lines_per_file: usize = if (lib.is_debug) 1000 else 1_000_000;

    var timer: utils.Timer = .start();
    var file_nr: usize = 0;
    var file_lines_written: usize = 0;
    var sum_line_lengths: u64 = 0;
    var stats: Stats = std.mem.zeroes(Stats);

    // src file.
    var reader: utils.FileReader = try utils.FileReader.init(src_file, 8192);
    stats.source_filesize = try reader.rd.getSize();
    defer reader.deinit();

    var viri_filename = compute_viri_filename(file_nr);
    var viri_writer: viri.ViriFileWriter = try .init(viri_filename, 4096);
    defer viri_writer.deinit();

    const skipped_filename = to_str("{s}skipped.log", .{ dst_folder });
    var skipped_writer: utils.FileWriter = try utils.FileWriter.init(skipped_filename, 512);
    defer skipped_writer.deinit();

    const illegal_filename = to_str("{s}illegal.log", .{ dst_folder });
    var illegal_writer: utils.FileWriter = try utils.FileWriter.init(illegal_filename, 512);
    defer illegal_writer.deinit();

    var game: viri.ViriGame = .init();
    defer game.deinit();

    var pos: Position = .empty;
    var pvpos: Position = .empty;

    read_loop: while (true) {
        const line = reader.readline() catch break :read_loop;
        if (line == null) {
            break :read_loop;
        }
        stats.positions_processed += 1;
        const line_len: usize = line.?.len;
        sum_line_lengths += line_len;

        var parsed = try std.json.parseFromSlice(JsonEntry, gpa, line.?, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const entry: JsonEntry = parsed.value;

        if (entry.extract_record()) |rec| {
            // Skip mate scores.
            if (scoring.is_matescore(rec.cp)) {
                stats.mates_skipped += 1;
                continue :read_loop;
            }
            // Skip extreme scores.
            if (rec.cp > 10000 or rec.cp < -10000) {
                stats.eval_bounds_skipped += 1;
                continue :read_loop;
            }
            // Set the position.
            pos.setup(rec.fen, false) catch |err| {
                stats.nonsense_positions_skipped += 1;
                try skipped_writer.writeline("{s} (crashed: {s})", .{rec.fen, @errorName(err)});
                continue :read_loop;
            };
            // If nonsense skip it.
            validate_pos(&pos) catch |err| {
                stats.nonsense_positions_skipped += 1;
                try skipped_writer.writeline("{s} (discarded: {s})", .{rec.fen, @errorName(err)});
                continue :read_loop;
            };
            // If check skip it.
            if (pos.checkmask != 0) {
                stats.in_check_skipped += 1;
                continue :read_loop;
            }
            // Check the pv.
            pvpos = pos;
            const pv: PV = execute_pv(&pvpos, rec.pv, true) catch |err| {
                switch (err) {
                    PvError.empty => {
                        stats.pv_empty += 1;
                        continue :read_loop;
                    },
                    PvError.illegal => {
                        stats.pv_illegal += 1;
                        try illegal_writer.writeline("{s} pv {s} ({})", .{ rec.fen, rec.pv, err } );
                        continue :read_loop;
                    },
                }
            };
            // Now we have at least one non-capture move.
            const firstmove: ExtMove = pv.extmoves[0];
            if (firstmove.move.is_capture()) {
                stats.pv_capture_skipped += 1;
                continue :read_loop;
            }
            // Write this position + absoluate eval + first pv move.
            const eval: i16 = @intCast(rec.cp);
            // We do not allow to skip a crash here. If we fail, we fail completely.
            try game.reset_with_position(&pos, eval, types.WDL.draw.u);
            try game.append_move(firstmove.move, eval);
            try viri_writer.write_game(&game);
            // Keep track of things.
            file_lines_written += 1;
            stats.positions_written += 1;
            // Update statistics.
            const piececount: u7 = funcs.popcnt(pos.all());
            stats.by_piece_count[piececount] += 1;
            stats.max_eval = @max(stats.max_eval, rec.cp);
            stats.min_eval = @min(stats.min_eval, rec.cp);
            stats.by_stm_count[pos.stm.u] += 1;
            // Ready?
            if (max) |m| {
                if (stats.positions_written >= m) {
                    break;
                }
            }
            // Create next file.
            if (file_lines_written >= max_lines_per_file) {
                viri_writer.deinit();
                file_lines_written = 0;
                file_nr += 1;
                viri_filename = compute_viri_filename(file_nr);
                viri_writer = try .init(viri_filename, 4096);
            }
        }

        // Some visual feedback of the progress.
        if (timer.elapsed_ms() > 1000) {
            const perc = funcs.percent(stats.source_filesize, sum_line_lengths);
            lib.io.print(
                "processed {} M, written {} M, progress {}% ({}m / {}m)\n",
                .{ stats.positions_processed / types.million, stats.positions_written / types.million, perc, sum_line_lengths / types.million, stats.source_filesize / types.million }
            );
            timer.reset();
        }
    }

    // We are done with processing.
    try viri_writer.flush();
    try skipped_writer.flush();

    // Write statistics.
    const stats_filename = to_str("{s}stats.log", .{ dst_folder });
    var stats_writer: utils.FileWriter = try utils.FileWriter.init(stats_filename, 512);
    defer stats_writer.deinit();

    try stats_writer.writeline(
        \\filesize {}
        \\processed {}
        \\written {}
        \\mate_skipped {}
        \\bounds_skipped {}
        \\in_check_skipped {}
        \\pv_capture_skipped {}
        \\nonsense_positions_skipped {}
        \\pv_empty {}
        \\pv_illegal {}
        \\max_score {}
        \\min_eval {}
        ,
        .{
            stats.source_filesize,
            stats.positions_processed,
            stats.positions_written,
            stats.mates_skipped,
            stats.in_check_skipped,
            stats.pv_capture_skipped,
            stats.eval_bounds_skipped,
            stats.nonsense_positions_skipped,
            stats.pv_empty,
            stats.pv_illegal,
            stats.max_eval,
            stats.min_eval
        }
    );

    try stats_writer.writeline("wtm: {}", .{ stats.by_stm_count[0] });
    try stats_writer.writeline("btm: {}", .{ stats.by_stm_count[1] });

    for (2..32 + 1) |c| {
        try stats_writer.writeline("piececount [{}]: {}", .{ c, stats.by_piece_count[c] });
    }

    try stats_writer.flush();

    // wait before close terminal.
    lib.io.print("lichess json conversion ready ({} position written)\n", .{stats.positions_written});
    _ = lib.io.readline() catch return;
}

fn validate_pos(pos: *const Position) Discarded!void {
    const all: u7 = funcs.popcnt(pos.all());
    if (all == 2) return Discarded.kings_only;
    if (all > 32) return Discarded.too_many_pieces;
    if (pos.pawn_count(.white) > 8 or pos.pawn_count(.black) > 8) return Discarded.too_many_pawns;
    if (pos.all_pawns() & (bitboards.bb_rank_1 | bitboards.bb_rank_8) != 0) return Discarded.pawns_on_backrank;
    if (pos.knight_count(.white) > 4 or pos.knight_count(.black) > 4) return Discarded.more_than_four_knights;
    if (pos.bishop_count(.white) > 4 or pos.bishop_count(.black) > 4) return Discarded.more_than_four_bishops;
    if (pos.rook_count(.white) > 4 or pos.rook_count(.black) > 4) return Discarded.more_than_four_rooks;
    if (pos.queen_count(.white) > 4 or pos.queen_count(.black) > 4) return Discarded.more_than_four_queens;
    if (pos.knight_count(.white) + pos.pawn_count(.white) > 10 or pos.knight_count(.black) + pos.pawn_count(.black) > 10) return Discarded.too_many_knights;
    if (pos.bishop_count(.white) + pos.pawn_count(.white) > 10 or pos.bishop_count(.black) + pos.pawn_count(.black) > 10) return Discarded.too_many_bishops;
    if (pos.rook_count(.white) + pos.pawn_count(.white) > 10 or pos.rook_count(.black) + pos.pawn_count(.black) > 10) return Discarded.too_many_rooks;
    if (pos.queen_count(.white) + pos.pawn_count(.white) > 9 or pos.queen_count(.black) + pos.pawn_count(.black) > 9) return Discarded.too_many_queens;
}

/// `stop_if_first_move_is_capture` prevents useless processing, because we do not store 'best move is capture' positions.
fn execute_pv(pos: *Position, pv_str: []const u8, stop_if_first_move_is_capture: bool) PvError!PV {
    if (pv_str.len == 0) {
        return PvError.empty;
    }
    var pv: types.ExtMoveList(64) = .init();
    var tokenizer = std.mem.tokenizeScalar(u8, pv_str, ' ');
    var idx: usize = 0;
    while (tokenizer.next()) |m| {
        const ex: ExtMove = pos.parse_move(m) catch return PvError.illegal;
        pos.lazy_do_move(ex);
        pv.add(ex);
        if (idx == 0 and stop_if_first_move_is_capture) {
            break;
        }
        idx += 1;
        if (idx >= PV.max_capacity) {
            break;
        }
    }
    return pv;
}

const Discarded = error {
    kings_only,
    too_many_pieces,
    too_many_pawns,
    pawns_on_backrank,
    more_than_four_knights,
    more_than_four_bishops,
    more_than_four_rooks,
    more_than_four_queens,
    too_many_knights,
    too_many_bishops,
    too_many_rooks,
    too_many_queens,
};

const PvError = error {
    empty,
    illegal,
};


// --- The lichess dataset json format (formatted from 1 line) ---

// {
//     "fen":"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
//     "evals":[
//         {
//             "pvs":[
//             {
//                 "cp":48,
//                 "line":"f7g7 e6e2 h8d8 e2d2 b7b5 c4b3 a6a5 a2a3 g7f6 b3a2"
//             }
//             ],
//             "knodes":644403,
//             "depth":55
//         },
//         {
//             "pvs":[
//             {
//                 "cp":69,
//                 "line":"f7g7 e6e2 h8d8 e2d2 b7b5 c4b3 g7f6 d1e1 a6a5 a2a3"
//             },
//             {
//                 "cp":163,
//                 "line":"h8d8 d1e1 a6a5 a2a3 c6d7 e6e7 f7f6 e1f2 b7b5 c4b3"
//             },
//             {
//                 "cp":229,
//                 "line":"h8a8 d1e1 a6a5 e6h6 f7g7 h6h4 a8d8 c4d3 c6g2 d3f5"
//             },
//             {
//                 "cp":231,
//                 "line":"h8f8 d1e1 b7b5 c4b3 a6a5 e6h6 f7g7 h6h4 f8e8 e1f2"
//             },
//             {
//                 "cp":237,
//                 "line":"h8b8 d1e1 a6a5 e6h6 f7g7 h6h4 b8d8 c4d3 c6g2 d3f5"
//             }
//             ],
//             "knodes":4189972,
//             "depth":46
//         }
//     ]
// }
