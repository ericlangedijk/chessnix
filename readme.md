### Current Code Swamp 1.x
Currently finetuning the algorithms, speed, chess960 support. ELO target for the 1.x version is 3200.

I am a github noob and only use the main branch as a backup of the current state on my PC.
With every release an extra zip of the source code at that moment is provided.

### Version 1.0
Engine plays reasonable chess. No numbers yet but guessing between 2500 and 2900.
Many many things need still to be changed, but it is time for a first version.

### Version 0.1
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

### Thanks to:
Aron Petkovski. I used most of the tuned evaluation of an old version of Integral (3).
More thanks to Janez and Colin on Discord. The people on TalkChess. The people on the Zig Forum.
And the authors of all the engines I used for testing: Bbc, Chessplusplus, Colossus, Cwtch, Infrared, Integral, Lambergar, Lishex, Mess, Monty, Seawall, Shallow, Stockfish, Supernova, Teki.
I also learned a lot from the source code of all these engines.