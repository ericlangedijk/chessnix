// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const bitboards = @import("bitboards.zig");
const data = @import("data.zig");

const ctx = lib.ctx;
const wtf = lib.wtf;

const Color = types.Color;
const Direction = types.Direction;
const Square = types.Square;
const Piece = types.Piece;
const Move = types.Move;
const Position = position.Position;

pub const FenResult  = struct
{
    pieces: [64]Piece,
    to_move: Color,
    ep: ?Square,
    castling_rights: u4,
    draw_count: u16,
    game_ply: u16,

    fn init() FenResult
    {
        return FenResult
        {
            .pieces =  @splat(Piece.NO_PIECE),
            .to_move = Color.WHITE,
            .ep = null,
            .castling_rights = 0,
            .draw_count = 0,
            .game_ply = 0,
        };
    }
};

pub const FenError = error
{
    BoardChar, Color, DrawCount, Castle, MoveNumber,
    NoSemiColon, NoDepthChar, ExpectedNodes, TooManyPartsInDepth
};

/// happy only. we should catch the biggest nonsense still.
// "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
// * If there follow 'custom' information with semicolons, we need a space after the normal fen.
pub fn decode(fen: []const u8) FenError!FenResult
{
    // parse consts
    const STATE_BOARD: i32 = 0;
    const STATE_COLOR: i32 = 1;
    const STATE_CASTLE: i32 = 2;
    const STATE_EP: i32 = 3;
    const DRAW_COUNT: i32 = 4;
    const MOVENUMBER: i32 = 5;

    var result: FenResult = .init();

    var state: i32 = STATE_BOARD;
    var rank: u3 = bitboards.rank_8;
    var file: u3 = bitboards.file_a;

    //const semicolon = std.mem.indexOfScalar(u8)

    //const stripped: []const u8 = fen;

    //const semicolon = std.mem.indexOfScalar(u8, fen, ';'); // TODO: get rid of this!!
    //var iter = if (semicolon) |p| std.mem.tokenizeScalar(u8, fen[0..p], ' ') else std.mem.tokenizeScalar(u8, fen, ' ');

    var iter = std.mem.tokenizeScalar(u8, fen, ' ');

    //var iter = std.mem.tokenizeAny(u8, fen, &.{' ', ';'} );
    //var iter = std.mem.tokenizeScalar(u8, fen, ' ');
    outer: while (iter.next()) |slice|
    {
        // std.debug.print("WTF {s}\n", .{slice});
        switch (state)
        {
            STATE_BOARD =>
            {
                for (slice) |c|
                {
                    switch (c)
                    {
                        '1'...'8' =>
                        {
                            const delta: u3 = @truncate(c - '0');
                            //console.print("xfile {}, ", .{file});
                            file = file +| delta;
                        },
                        '/' =>
                        {
                            rank -= 1;
                            file = bitboards.file_a;
                        },
                        'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K' =>
                        {
                            const pc: Piece = get_piece(c);
                            const sq: Square = .from_rank_file(rank, file);
                            result.pieces[sq.u] = pc;
                            //console.print("pfile {}, ", .{file});
                            file = file +| 1;
                        },
                        else =>
                        {
                           return FenError.BoardChar;
                        }
                    }
                }
            },
            STATE_COLOR =>
            {
                if (slice.len != 1) return FenError.Color;
                const c = slice[0];
                result.to_move = switch(c)
                {
                    'w', 'W' => Color.WHITE,
                    'b', 'B' => Color.BLACK,
                    else => return FenError.Color,
                };
            },
            STATE_CASTLE =>
            {
                // nothing = '-'

                for (slice) |c|
                {
                    switch (c)
                    {
                        'K' => result.castling_rights |= position.cf_white_short,
                        'Q' => result.castling_rights |= position.cf_white_long,
                        'k' => result.castling_rights |= position.cf_black_short,
                        'q' => result.castling_rights |= position.cf_black_long,
                        else => {}
                    }
                }

                // TODO: chess960 shredder
            },
            STATE_EP =>
            {
                if (slice.len == 2)
                {
                    result.ep = parse_square(slice[0..2]);
                }
            },
            DRAW_COUNT =>
            {
                const v: u16 = std.fmt.parseInt(u16, slice, 10) catch break :outer;
                result.draw_count = v;
            },
            MOVENUMBER =>
            {
                const v: u16 = std.fmt.parseInt(u16, slice, 10)  catch break :outer;
                result.game_ply = funcs.movenumber_to_ply(v, result.to_move);
            },
            else =>
            {
                //console.print("{s}\n", . { slice } );
                break :outer;
            }
        }
        state += 1;
    }
    return result;
}

pub const FenDepths = std.BoundedArray(u64, 16);

/// "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ;D1 20 ;D2 400 ;D3 8902 ;D4 197281 ;D5 4865609 ;D6 119060324"
/// "1B6/1r3Bk1/8/Pp6/4KN1p/8/5b1R/8 b - -;23;728;14764;461899;9440955;292742932"
pub fn decode_depths(fen: []const u8) FenError!FenDepths
{
    var depths: std.BoundedArray(u64, 16) = .{};
    depths.appendAssumeCapacity(0);

    const first_semicolon: usize = index_of(fen, ';') orelse return FenError.NoSemiColon;
    const last_part = fen[first_semicolon + 1..];

    var iter = std.mem.tokenizeScalar(u8, last_part, ';');

    var depth: usize = 1;
    while (iter.next()) |slice| // D1 20
    {

        if (std.mem.indexOfScalar(u8, slice, ' ') == null)
        {
            const nodes: u64 = std.fmt.parseInt(u64, slice, 10) catch return FenError.ExpectedNodes;
            depths.appendAssumeCapacity(nodes);
        }
        else
        {
            if (slice[0] != 'D') return FenError.NoDepthChar;
            var sub_iter = std.mem.tokenizeScalar(u8, slice, ' ');
            var i: usize = 0;
            while (sub_iter.next()) |part|
            {
                if (i == 0) // D1 (depth)
                {
                    //const nr = part[1..];
                    // TODO: skip for now. lazy me. we should parse the depth here.
                }
                else if (i == 1) // 20 (node)
                {
                    //std.debug.print("PART [{s}], ", .{part});
                    const nodes: u64 = std.fmt.parseInt(u64, part, 10) catch return FenError.ExpectedNodes;
                    depths.appendAssumeCapacity(nodes);
                }
                else return FenError.TooManyPartsInDepth;
                i += 1;
            }
        }
        depth += 1;
    }
    return depths;
}

fn get_piece(char: u8) Piece
{
    return switch(char)
    {
        'P' => Piece.W_PAWN,
        'N' => Piece.W_KNIGHT,
        'B' => Piece.W_BISHOP,
        'R' => Piece.W_ROOK,
        'Q' => Piece.W_QUEEN,
        'K' => Piece.W_KING,
        'p' => Piece.B_PAWN,
        'n' => Piece.B_KNIGHT,
        'b' => Piece.B_BISHOP,
        'r' => Piece.B_ROOK,
        'q' => Piece.B_QUEEN,
        'k' => Piece.B_KING,
        else => wtf(),
    };
}

fn contains(slice: []const u8, value:u8) bool
{
    for (slice) |c|
    {
        if (c == value) return true;
    }
    return false;
}

fn index_of(slice: []const u8, value:u8) ?usize
{
    for (slice, 0..) |c, i|
    {
        if (c == value) return i;
    }
    return null;
}


fn parse_square(s: []const u8) Square
{

    const file: u8 = switch (s[0])
    {
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'  => (s[0] - 'a'),
       else => wtf(),
    };
    const rank: u8 = s[1] - '1';

    return Square.from_rank_file(@truncate(rank), @truncate(file));
}

