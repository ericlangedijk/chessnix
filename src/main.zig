// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

    // TEMP
    const types = @import("types.zig");
    const hcetables = @import("hcetables.zig");
    const funcs = @import("funcs.zig");
    const bitboards = @import("bitboards.zig");
    const li = @import("tools/lichess.zig");
    const mp = @import("movepick.zig");
    const tt = @import("tt.zig");

pub fn main() !void {
    @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        //try @import("tests.zig").run_silent_debugmode_tests();
    }

    // const current_age: i32 = 4;
    // const age: i32 = 16;
    // const age_penalty: i32 = (current_age - age) & 31;
    // lib.io.debugprint("{} {} -> {}\n", .{ current_age, age, age_penalty});

    //try li.analyze_lichess_eval_dataset_moves("C:/Data/Download Programmeren/chess/lichess eval datasets/lichess_db_eval.jsonl");
    // lib.io.print("{} {}\n\n", .{ @sizeOf(tt.Entry), @bitSizeOf(tt.Entry) });
    // lib.io.print("{} {}\n\n", .{ @sizeOf(tt.Bucket), @bitSizeOf(tt.Bucket) });

    // lib.io.print("{} {}\n\n", .{ @sizeOf(Entry), @sizeOf(Bucket) });

    // const e0_age_penalty: i32 = (@as(i32, self.age) - bucket.e0.age) & 31;
    // var a: u5 = 0;
    // var e: u5 = 0;

    // for (0..45) |_| {
    //     for (0..45) |_| {
    //         const penalty: i32 = (@as(i32, a) - e) & 31;
    //         lib.io.debugprint("a {} e {} p {}\n", .{ a, e, penalty });
    //         e +%= 1;
    //     }
    //     a +%= 1;
    // }



    switch (lib.program) {
        .engine => {
            uci.run();
        },
        .generating_lichess_dataset => {
            //try li.convert_lichess_dataset("C:/Data/Download Programmeren/chess/lichess eval datasets/short.jsonl");
            //try li.convert_lichess_dataset("C:/Data/Download Programmeren/chess/lichess eval datasets/lichess_db_eval.jsonl");
            _ = try lib.io.readline();
        },
        .tuning => {
            const tune = @import("tuner.zig");
            try tune.tune("C:/Data/Download Programmeren/chess/lichess eval datasets/dateset.epd", null);
            _ = try lib.io.readline();
        },
        .nothing => {
            // do nothing :)
        },
    }
}


// pub const Square = enum(u6) {
//     a1, b1, c1, d1, e1, f1, g1, h1,
//     a2, b2, c2, d2, e2, f2, g2, h2,
//     a3, b3, c3, d3, e3, f3, g3, h3,
//     a4, b4, c4, d4, e4, f4, g4, h4,
//     a5, b5, c5, d5, e5, f5, g5, h5,
//     a6, b6, c6, d6, e6, f6, g6, h6,
//     a7, b7, c7, d7, e7, f7, g7, h7,
//     a8, b8, c8, d8, e8, f8, g8, h8,
// };

// pub const Move = packed struct {
//     from: Square,
//     to: Square,
//     flags: u4,
// };



// pub const Bound = enum(u2) {
//     /// No bound. Empty or static eval only.
//     none,
//     /// Upper bound.
//     alpha,
//     /// Lower bound.
//     beta,
//     /// Exact score.
//     exact,
// };


// pub const Entry = struct {

//     pub const Flags = packed struct {
//         bound: Bound,
//         age: u5,
//     };

//     key: u16,
//     score: i16,
//     raw_static_eval: i16,
//     move: u16,
//     depth: u8,
//     flags: Flags,
// };

// pub const Bucket = struct {
//     e0: Entry,
//     e1: Entry,
// };