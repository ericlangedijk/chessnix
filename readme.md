Engine plays terrible chess, mainly because I am not interested at all in writing an evaluation function.
To be continued...

### UCI Chess Engine. Work in progress version
- Needs Zig 0.15.1
- On Windows run chessnix in administrator mode for better performance.

- 2025-09-13: First UCI version working. Can be connected with a GUI like CuteChess. TT bug still, so is cleared every move.
- 2025-08-30: Engine basics kinda working (blocks IO, single-threaded, no move ordering, no TT, quiescence still exploding).

### Terminal Commands:
- position startpos [optional_moves] -> sets the classic startposition and makes the moves, if there are any.
- position fen [fen] [optional_moves] -> set the position from the fen and makes the moves, if there are any.
- go depth [depth] -> search position to depth. NOTE: total tree explosion when many captures.
- go movetime [milliseconds] -> search position for milliseconds.
- perft [depth] -> performs a perft test with subtotals per move.
- qperft [depth] -> performs a perft without subtotals per move.
- bench -> performs perft speedtests on 4 positions.
- d -> draw the current position.
- quit -> stop the program.
