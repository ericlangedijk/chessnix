A testversion named chessnix_bot can be played against on lichess.org.

### UCI Chess Engine
- Needs Zig 0.15.2.
- On Windows run chessnix terminal in administrator mode for better performance.
- chessnix is a windows 64 bits exe for modern computers.

### Version 1.3 Work In Progress...
- Target 3200+ ELO. Working on time management, search finetuning, speed, structure. Currently at ~3150 ELO.
- Working on another terrible bug: the correction history is completely wrong.

### Version 1.2
- Stronger: ~3015 ELO on CCRL.
- Added: Chess960 support.
- Added: Staged movepicking.
- Added: Razoring.
- Added: Continuation History.
- Added: Correction History.
- Added: Internal Iterative Reduction.
- Added: CutNode reduction.
- Added: SEE pruning.
- Changed: Transposition Table with 2 entries per bucket.
- Changed: Aspiration Window strategy.
- Changed: Late Move Pruning. Using an improvement rate + depth formula.
- Changed: Late Move Reduction. Using a precomputed table now.
- Changed: Quiescence search. Does not store bestmove anymore.
- Changed: Node counting. Only incremented after doing a move.
- Changed: Finetuned search in general.
- Move generation refactor.
- Position always storing pins for both sides. For future perfect SEE.
- Bug solved: Invalid principal variation output.
- Bug solved: Bishop pair evaluation bug. The mask for white / black squares was wrong.
- Bug solved: A terrible 'get all attacks' bug affecting evaluation, SEE and move ordering.

### Version 1.1
Does not exist. This was a local bugfix for pv output.

### Version 1.0
Engine plays reasonable chess. Entered CCRL with 2842 ELO.
Many many things need still to be changed, but it is time for a first version.

### Version 0.1
Engine plays terrible chess, mainly because I am not interested at all in writing an evaluation function.
To be continued...

### Versioning
- Major number changes when the evaluation function changes.
- Minor number changes for improvements and bugfixes.

### Planning
- Version 1.x 3200+ ELO.
- Version 2.x 3400+ ELO (extended evaluation tuning).
- Version 3.x 3600+ ELO (use neural network training).

### Some terminal command examples
- setoption name UCI_Chess960 value true -> enables chess960.
- position startpos moves e2e4 e7e5. (moves" is optional) -> sets the startposition, does the moves.
- position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4. (moves is optional) -> sets the fen, does the moves.
- eval -> give the static evaluation of the position.
- go depth 7 -> search position to depth 7.
- go movetime 5000 -> search position for 5000 milliseconds.
- perft 5 -> performs a perft test with subtotals per move.
- qperft 5 -> performs a perft without subtotals per move.
- bench -> performs perft speedtests on 4 positions.
- d -> draw the current position and some info.
- quit -> stop the program.

### Thanks
Aron Petkovski. I used most of the tuned evaluation of an old version of Integral (3) to get me started (from chessnix 2.0 onwards I will do my own tuning).
Janez and Colin on Discord. The people of TalkChess, Zig Forum, CCRL.
The authors of the engines I used for testing: Bbc, Cheese, Chessplusplus, Colossus, Cwtch, Infrared, Integral, Lambergar, Lishex, Mess, Monty, OpenCritter, PlentyChess, Seawall, Shallow, Shallowguess, Stash, Stockfish, Supernova, Teki.
Some extra thanks to these people who made their engine support chess960.
I also learned a lot from the source code of all these engines.
Chessnix contains, besides my own inventions, a wild mix of ideas from the chess programming wiki as well as several other chess engines (Cwtch, Lambergar, Integral, PlentyChess, Stockfish and probably others).

### Noob
I am a github noob and only use a main branch as a backup of the current state on my PC.
With every release an extra zip of the source code at that moment is provided.
