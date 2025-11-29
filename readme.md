A testversion named chessnix_bot can be played against on lichess.org.

### Current Work in Progress Code Swamp 1.2
Currently finetuning the algorithms, speed, chess960 support. ELO target for the 1.x version is 3100.

- A little stronger: probably somewhere around 2890 ELO.
- Added: Chess960 support.
- Added: Razoring.
- Added: Continuation History.
- Added: Correction History.
- Added: Internal Iterative Reduction.
- Added: CutNode reduction.
- Added: Node inherits killers from 2 ply earlier.
- Changed: Transposition Table with 2 entries per bucket.
- Changed: Aspiration Window strategy.
- Changed: Late Move Reduction. Using a precomputed table now.
- Changed: Quiescence search. Does not store bestmove anymore.
- Changed: Node counting. Only incremented after doing a move.
- Bug solved: invalid principal variation output.
- Bug solved: bishop pair evaluation bug. The mask for white / black squares was wrong.

### Version 1.1
Does not exist. This was a local bugfix for pv output.

### Version 1.0
Engine plays reasonable chess. Entered CCRL with 2842 ELO.
Many many things need still to be changed, but it is time for a first version.

### Version 0.1
Engine plays terrible chess, mainly because I am not interested at all in writing an evaluation function.
To be continued...

### UCI Chess Engine. Work in progress version
- Needs Zig 0.15.1
- On Windows run chessnix in administrator mode for better performance.

- 2025-09-13: First UCI version working. Can be connected with a GUI like CuteChess. TT bug still, so is cleared every move.
- 2025-08-30: Engine basics kinda working (blocks IO, single-threaded, no move ordering, no TT, quiescence still exploding).

### Terminal Commands Examples:
- setoption name UCI_Chess960 value true.
- position startpos moves e2e4 e7e5. "moves" is optional.
- position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4. "moves is optional"
- go depth 7 -> search position to depth 7.
- go movetime 5000 -> search position for 5000 milliseconds.
- perft 5 -> performs a perft test with subtotals per move.
- qperft 5 -> performs a perft without subtotals per move.
- bench -> performs perft speedtests on 4 positions.
- d -> draw the current position and some info.
- quit -> stop the program.

### Thanks to
Aron Petkovski. I used most of the tuned evaluation of an old version of Integral (3).
More thanks to Janez and Colin on Discord. The people on TalkChess. The people on the Zig Forum.
And the authors of all the engines I used for testing: Bbc, Chessplusplus, Colossus, Cwtch, Infrared, Integral, Lambergar, Lishex, Mess, Monty, Seawall, Shallow, Stockfish, Supernova, Teki.
I also learned a lot from the source code of all these engines.

### Noob
I am a github noob and only use the main branch as a backup of the current state on my PC.
With every release an extra zip of the source code at that moment is provided.
