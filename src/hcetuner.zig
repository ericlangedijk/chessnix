// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");

const ScorePair = types.ScorePair;
const Position = position.Position;
const Evaluator = hce.Evaluator;

const scorepair_count: usize = @sizeOf(hceterms.Terms) / @sizeOf(ScorePair);
/// A mutable flat view on the terms.
const view: *[scorepair_count]ScorePair = @constCast(@ptrCast(hceterms.terms));

const ScorePairStats = struct {
    this_usage: u64,
    global_usage: u64,
};

/// evaluation triggers this function for each used scorepair when tuning.
pub fn register_scorepair_usage(sp: *ScorePair, times: usize) void {
    if (!lib.is_tuning){
        @compileError("only when tuning");
    }
    register(sp, times);
}

pub fn run() !void {
    if (!lib.is_tuning){
        @compileError("only when tuning");
    }
    //names();
    //if (true) return;
    const pos: Position = try .init("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", false);
    var evaluator: Evaluator = .init();
    _ = evaluator.evaluate(&pos);
}

fn register(sp: *ScorePair, times: usize) void {
    lib.io.debugprint("used {any} times {} index {} name {s}\n", .{ sp.*, times, index_of_scorepair(sp), name_of_scorepair(sp) });
}

fn index_of_scorepair(sp: *const ScorePair) usize {
    const base: usize = @intFromPtr(view);
    const addr: usize = @intFromPtr(sp);
    return (addr - base) / @sizeOf(ScorePair);
}

fn name_of_scorepair(sp: *const ScorePair) []const u8 {
//fn names() void {
    const idx: usize = index_of_scorepair(sp) * @sizeOf(ScorePair);
    //const t = @typeInfo(hceterms.Terms);

    //const T: type = @TypeOf(hceterms.Terms);

    const info = @typeInfo(hceterms.Terms).@"struct";

    inline for (info.fields) |field| {
        //if (@typeInfo(field.type) == .array) {
            const offset = @offsetOf(hceterms.Terms, field.name);
            const size = @sizeOf(field.type);
            //lib.io.debugprint("    searching {} field {s} offset {} leng {}\n", .{ idx, field.name, offset, size });
            if (idx >= offset and idx <= size + offset) return field.name;
        //}
    }


    return "?";

    // for (@typeInfo(@TypeOf(hceterms.Terms)).@"struct".fields) |field| {
    //     if (@typeInfo(field.type) == .array) {
    //         lib.io.debugprint("field", .{  });
    //     }
    //     //_ = field;
    //     //lib.io.debugprint("field", .{  });
    // }

    // inline for (t.fields) |field| {
    //     const offset = @offsetOf(S, field.name);
    //     const size = @sizeOf(field.type);

    //     std.debug.print(
    //         "{s}: offset={}, size={}\n",
    //         .{ field.name, offset, size },
    //     );
    // }

    // const base: usize = @intFromPtr(view);
    // const addr: usize = @intFromPtr(sp);
    // return (addr - base) / @sizeOf(ScorePair);
}

const ScorePairInfo = struct {
    array_index: usize,
    name: utils.BoundedArray(u8, 64),
};

const MetaData = struct {
    info: [scorepair_count]ScorePairInfo,
};

pub const king_bucket_table: [64]u4 = .{
    0,  0,  0,  1,  1,  2,  2,  2,
    0,  0,  0,  1,  1,  2,  2,  2,
    3,  3,  3,  4,  4,  5,  5,  5,
    3,  3,  3,  4,  4,  5,  5,  5,
    6,  6,  6,  7,  7,  8,  8,  8,
    6,  6,  6,  7,  7,  8,  8,  8,
    9,  9,  9, 10, 10, 11, 11, 11,
    9,  9,  9, 10, 10, 11, 11, 11,
};