// zig fmt: off

//! Converts my local lichess json dataset
//! Source: C:/Data/chess/lichess/lichess_db_eval.jsonl.
//! Dest: C:/Data/chess/lichess/datasets files (0000.epd...nnnn.epd)
//! Multiple epd files are created with a fen + score + pv   :"8/p5b1/1p5p/4k1p1/2Pp1pP1/1P3K2/P2B3P/8 w - - 0 1 cp -254 pv ..."
//! Skipped: invalid and ridiculous positions.
//! Skipped: eval scores outside the range -10000...10000.
//! Skipped: positions that are in check.
//! Skipped: positions where the best move is a captures.


const std = @import("std");
const lib = @import("../lib.zig");
const utils = @import("../utils.zig");
const types = @import("../types.zig");
const bitboards = @import("../bitboards.zig");
const funcs = @import("../funcs.zig");
const position = @import("../position.zig");
const scoring = @import("../scoring.zig");

const ctx = lib.ctx;

const Color = types.Color;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;

const src_file = "C:/Data/chess/lichess/lichess_db_eval.jsonl";
const dst_folder = "C:/Data/chess/lichess/datasets/";

/// Our final conversion result.
const TuningRecord = struct {
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

    fn extract_tuning_record(self: *const JsonEntry) ?TuningRecord {
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
    /// The size of the source file.
    filesize: u64,
    /// Total processed positiions read.
    processed: u64,
    /// Total number of positions converted.
    written: u64,
    /// Mate evaluation skipped.
    mates_skipped: u64,
    /// Out of bounds evaluations skipped.
    pv_capture_skipped: u64,
    /// Out of bounds evaluations skipped.
    is_check_positions_skipped: u64,
    /// Out of bounds evaluations skipped.
    bounds_skipped: u64,
    /// Nonsensical positions skipped.
    nonsense_positions_skipped: u64,
    /// Empty pv's found.
    pvs_empty: u64,
    /// Illegal pv's skipped.
    pvs_illegal: u64,
    // The maximum used score.
    max_score: i32,
    /// The minimum used score.
    min_score: i32,
    /// Counts per stm
    by_stm_count: [2]u64,
    /// Counts per piececount.
    by_piece_count: [33]u64,
    /// Counts per kingbucket.
    by_king_bucket: [12][12]u64,
};

const PvResult = enum {
    ok,
    empty,
    illegal,
};

var str_buf: [256]u8 = undefined;

pub fn to_str(comptime fmt: []const u8, args: anytype)[]u8 {
    return std.fmt.bufPrint(&str_buf, fmt, args) catch lib.wtf("to_str", {});
}

pub fn convert_lichess_dataset() !void {

    const max: usize = if (lib.is_debug) 10000 else 100_000_000;
    const max_lines_per_file: usize = if (lib.is_debug) 1000 else 1_000_000;

    var timer: utils.Timer = .start();
    var file_nr: usize = 0;
    var file_lines_written: usize = 0;
    var sum_line_lengths: u64 = 0;
    var stats: Stats = std.mem.zeroes(Stats);

    // src file.
    var reader: utils.TextFileReader = try utils.TextFileReader.init(src_file, ctx.galloc, 8192);
    stats.filesize = try reader.reader.getSize();
    defer reader.deinit();

    // dst files.
    var epd_filename = to_str("{s}{:0>4}.epd", .{ dst_folder, file_nr });
    var epd_writer: utils.TextFileWriter = try utils.TextFileWriter.init(to_str(), ctx.galloc, 512);
    defer epd_writer.deinit();

    const skipped_filename = to_str("{s}skipped.log", .{ dst_folder });
    var skipped_writer: utils.TextFileWriter = try utils.TextFileWriter.init(skipped_filename, ctx.galloc, 512);
    defer skipped_writer.deinit();

    const illegal_filename = to_str("{s}illegal.log", .{ dst_folder });
    var illegal_writer: utils.TextFileWriter = try utils.TextFileWriter.init(illegal_filename, ctx.galloc, 512);
    defer illegal_writer.deinit();

    var pos: Position = .empty;
    var pvpos: Position = .empty;

    read_loop: while (true) {
        const line = reader.readline() catch break :read_loop;
        if (line == null) {
            continue;
        }
        stats.processed += 1;
        const line_len: usize = line.?.len;
        sum_line_lengths += line_len;

        var parsed = try std.json.parseFromSlice(JsonEntry, ctx.galloc, line.?, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const entry: JsonEntry = parsed.value;

        if (entry.extract_tuning_record()) |rec| { // TODO: log misses.

            // Skip mate scores.
            if (scoring.is_matescore(rec.cp)) {
                stats.mates_skipped += 1;
                continue :read_loop;
            }
            // Skip extreme scores.
            if (rec.cp > 10000 or rec.cp < -10000) {
                stats.bounds_skipped += 1;
                continue :read_loop;
            }
            // Set the position.
            pos.set(rec.fen, false) catch {
                stats.nonsense_positions_skipped += 1;
                try skipped_writer.writeline("{s} (crashed)", .{ rec.fen } );
                continue :read_loop;
            };
            // If nonsense skip it.
            if (!pos_ok(&pos)) {
                stats.nonsense_positions_skipped += 1;
                try skipped_writer.writeline("{s} (discarded)", .{ rec.fen } );
                continue :read_loop;
            }
            // If check skip it.
            if (pos.checkmask != 0) {
                stats.is_check_positions_skipped += 1;
                continue :read_loop;
            }
            // Check the pv.
            pvpos = pos;
            var first_move_is_capture: bool = false;
            const pv_result: PvResult = execute_pv(&pvpos, rec.pv, &first_move_is_capture);
            if (pv_result == .illegal) {
                stats.pvs_illegal += 1;
                try illegal_writer.writeline("{s} pv {s}", .{ rec.fen, rec.pv } );
                continue :read_loop;
            }
            if (first_move_is_capture) {
                stats.pv_capture_skipped += 1;
                continue :read_loop;
            }
            // Write this position + eval + pv.
            try write_pos(&epd_writer, &pos, rec.cp, rec.pv);
            // Keep track of lines.
            file_lines_written += 1;
            stats.written += 1;
            // Update statistics.
            const piececount1: u7 = bitboards.popcnt(pos.all());
            if (pv_result == .empty) {
                stats.pvs_empty += 1;
            }
            const kbw: u8 = pos.king_bucket(Color.white);
            const kbb: u8 = pos.king_bucket(Color.black);
            stats.by_king_bucket[kbw][kbb] += 1;
            stats.by_piece_count[piececount1] += 1;
            stats.max_score = @max(stats.max_score, rec.cp);
            stats.min_score = @min(stats.min_score, rec.cp);
            stats.by_stm_count[pos.stm.u] += 1;

            if (file_lines_written >= max_lines_per_file) {
                epd_writer.deinit();
                file_lines_written = 0;
                file_nr += 1;
                epd_filename = to_str("{s}{:0>4}.epd", .{ dst_folder, file_nr });
                epd_writer = try utils.TextFileWriter.init(epd_filename, ctx.galloc, 512);
            }
        }

        // Some visual feedback of the progress.
        if (timer.elapsed_ms() > 1000) {
            const perc = funcs.percent(stats.filesize, sum_line_lengths);
            lib.io.print("processed {} M, written {} M, progress {}% ({}m / {}m)\n", .{ stats.processed / types.million, stats.written / types.million, perc, sum_line_lengths / types.million, stats.filesize / types.million });
            timer.reset();
        }
        if (stats.processed > max) {
            break;
        }
    }

    try epd_writer.flush();
    try skipped_writer.flush();

    // Write statistics.
    const stats_filename = to_str("{s}stats.log", .{ dst_folder });
    var stats_writer: utils.TextFileWriter = try utils.TextFileWriter.init(stats_filename, ctx.galloc, 512);
    defer stats_writer.deinit();

    try stats_writer.writeline(
        "filesize {}\nprocessed {}\nwritten {}\nmate_skipped {}\nbounds_skipped {}\nis_check_skipped {}\npv_capture_skipped {}\nnonsense_positions_skipped {}\npv_empty {}\npv_illegal {}\nmax_score {}\nmin_score {}",
        .{ stats.filesize, stats.processed, stats.written, stats.mates_skipped, stats.is_check_positions_skipped, stats.pv_capture_skipped, stats.bounds_skipped, stats.nonsense_positions_skipped, stats.pvs_empty, stats.pvs_illegal, stats.max_score, stats.min_score }
    );

    try stats_writer.writeline("wtm: {}", .{ stats.by_stm_count[0] });
    try stats_writer.writeline("btm: {}", .{ stats.by_stm_count[1] });

    for (2..33) |c| {
        try stats_writer.writeline("piececount [{}]: {}", .{ c, stats.by_piece_count[c] });
    }

    // for (0..12) |kb1| {
    //     for (0..12) |kb2| {
    //         try stats_writer.writeline("kingbucket [{}][{}]: {}", .{ kb1, kb2, stats.by_king_bucket[kb1][kb2] });
    //     }
    // }
    try stats_writer.flush();

    // wait before close terminal.
    lib.io.print("lichess json conversion ready\n", .{});
    if (lib.is_release) {
        _ = try lib.io.readline();
    }
}

/// Skip ridiculous positions.
fn pos_ok(pos: *const Position) bool {
    const cnt: u7 = bitboards.popcnt(pos.all());
    const ok: bool =
        cnt > 2 and cnt <= 32 and
        pos.pawn_count(.white) <= 8 and pos.pawn_count(.black) <= 8 and
        pos.knight_count(.white) <= 4 and pos.knight_count(.black) <= 4 and
        pos.bishop_count(.white) <= 4 and pos.bishop_count(.black) <= 4 and
        pos.rook_count(.white) <= 4 and pos.rook_count(.black) <= 4 and
        pos.queen_count(.white) <= 4 and pos.queen_count(.black) <= 4 and
        pos.knight_count(.white) + pos.pawn_count(.white) <= 9 and pos.knight_count(.black) + pos.pawn_count(.black) <= 9 and
        pos.bishop_count(.white) + pos.pawn_count(.white) <= 9 and pos.bishop_count(.black) + pos.pawn_count(.black) <= 9 and
        pos.rook_count(.white) + pos.pawn_count(.white) <= 9 and pos.rook_count(.black) + pos.pawn_count(.black) <= 9 and
        pos.queen_count(.white) + pos.pawn_count(.white) <= 9 and pos.queen_count(.black) + pos.pawn_count(.black) <= 9;
    return ok;
}

fn execute_pv(pos: *Position, pv: []const u8, first_move_is_capture: *bool) PvResult {
    first_move_is_capture.* = false;
    if (pv.len == 0) {
        return .empty;
    }
    var tokenizer = std.mem.tokenizeScalar(u8, pv, ' ');
    var idx: usize = 0;
    while (tokenizer.next()) |m| {
        const ex: ExtMove = pos.parse_move(m) catch return .illegal;
        if (idx == 0 and ex.move.is_capture()) {
            first_move_is_capture.* = true;
        }
        pos.lazy_do_move(ex);
        idx += 1;
    }
    return .ok;
}

/// Flip if needed.
fn write_pos(textfilewriter: *utils.TextFileWriter, pos: *const Position, cp: i32, pv: []const u8) !void {
    try textfilewriter.writeline("{f} cp {} pv {s}", .{ pos, cp, pv });
}

// --- The lichess dataset json format (formatted) ---
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
