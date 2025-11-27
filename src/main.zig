// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // const q: Square = .a1;
    // std.debug.print("{t}\n", .{ q });
    // const n: Square = q.add(1);
    // std.debug.print("{t}\n", .{ n });


    uci.run();
}

// const Square = enum(u6) {
//     a1, b1, c1, d1, e1, f1, g1, h1,
//     a2, b2, c2, d2, e2, f2, g2, h2,
//     a3, b3, c3, d3, e3, f3, g3, h3,
//     a4, b4, c4, d4, e4, f4, g4, h4,
//     a5, b5, c5, d5, e5, f5, g5, h5,
//     a6, b6, c6, d6, e6, f6, g6, h6,
//     a7, b7, c7, d7, e7, f7, g7, h7,
//     a8, b8, c8, d8, e8, f8, g8, h8,

//     fn from(u: u6) Square {
//         return @enumFromInt(u);
//     }

//     fn add(self: Square, delta: u6) Square {
//         return @enumFromInt(@intFromEnum(self) + delta);
//     }
// };

// const SquareUnion = packed union {
//     pub const Enum = enum(u6) {
//         a1, b1, c1, d1, e1, f1, g1, h1,
//         a2, b2, c2, d2, e2, f2, g2, h2,
//         a3, b3, c3, d3, e3, f3, g3, h3,
//         a4, b4, c4, d4, e4, f4, g4, h4,
//         a5, b5, c5, d5, e5, f5, g5, h5,
//         a6, b6, c6, d6, e6, f6, g6, h6,
//         a7, b7, c7, d7, e7, f7, g7, h7,
//         a8, b8, c8, d8, e8, f8, g8, h8,
//     };
//     /// The enum value.
//     e: Enum,
//     /// The numeric value
//     u: u6,

//     fn from(u: u6) SquareUnion {
//         return .{ .u = u };
//     }

//     fn add(self: SquareUnion, delta: u6) SquareUnion {
//         return .{ .u = self.u + delta};
//     }
// };