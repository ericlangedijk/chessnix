# chessnix

<br/>
<p align="center">
<img src="logo.png" alt="logo" width=104 height=174/>
</p>
<br/>

Chessnix is a UCI chess engine.<br>
It uses 'HCE' (hand crafted evaluation) to guide the search.
I decided I don't understand anything of chess programming.

- Needs Zig to compile.
- chessnix is a windows 64 bits exe for modern computers.
- chessnix_bot can be played against on lichess.org.

## Version 1.4 Work in Progress...
- Compiler: Zig 0.15.2.
- Stronger: Optimistic estimation is ~3300 ELO (blitz ~3350). ELO will be updated here after release and CCRL results.
- Added: Evaluation: Knight outpost refinement, bad bishop, bishop on long diagonal.
- Changed: Correction History (better math, 5 tables for 5 position hashkeys).
- Changed: Alpha Raises Reduction LMR (instead of depth reduction).
- Changed: Only update pv when pv-node. Removed update pv in quiescence search.
- Changed: Time management.
- Changed: TT structure and entry replacement strategy.
- Changed: Lots of minor search algorithm details.
- Changed: Prefetch TT for a slight speed boost.
- Changed: Mate scores (in whole moves) in uci output.
- Removed: Reward or punish captures in quiescence search.
- Removed: Code that we do not need (yet).
- Bug solved: Ridiculous mate scores. The source of the problem is using tt score as node evaluation.
- Bug solved: Another terrible one: qsearch comparing a zero score with TT score instead of the eval.
- Bug solved: Parsing negative time.
- Refactor: Attempt to centralize all scoring logic in scoring.zig, hce terms in 1 struct, most search vars in 1 struct.

## Version 1.3
- Compiler: Zig 0.15.2.
- Stronger: ~3168 ELO on CCRL (blitz ~3271).
- Added: Slight center bias in quiet move ordering for shallow depths (experimental).
- Added: Search history pruning, history reduction.
- Added: Store raw static evaluation in TT.
- Added: Slight capture history bonus and malus in quiescence search.
- Removed: Killer moves.
- Removed: Maintaining pins of both sides. I will not code a 'perfect' SEE in the near future.
- Removed: Unused see_score function.
- Changed: Lots of tweaks in the search algorithm (history pruning + reduction, different LMR table).
- Changed: Correction history.
- Changed: History structure and calculations, especially continuation history.
- Changed: Rescaled history values.
- Changed: Moved 'ply' from position to search.
- Changed: Included a slight score correction during search using rule50.
- Changed: Bound logic when storing to TT. I think it is correct now.
- Changed: Move generation creates ExtMoves (64 bits) instead of the raw moves (16 bits), avoiding copying stuff.
- Bug solved: Terrible LMR table accessing out of bounds value resulting in god knows what.
- Bug solved: Node clearing (before enter search and on making nullmoves).
- Bug solved: Corrected 'id name' uci output.

## Version 1.2
- Compiler: Zig 0.15.2.
- Stronger: ~3015 ELO on CCRL (blitz ~3037).
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

## Version 1.1
Does not exist. This was a local bugfix for pv output.

## Version 1.0
Engine plays reasonable chess. Entered CCRL with ~2842 ELO (blitz ~2820).
Many many things need still to be changed, but it is time for a first version.

## Version 0.1
Engine plays terrible chess, mainly because I am not interested at all in writing an evaluation function.
To be continued...

## Versioning
- Major number changes when the evaluation data changes.
- Minor number changes for improvements and bugfixes.

## Planning
- Version 1.x 3200+ ELO.
- Version 2.x 3400+ ELO (extended HCE tuning).
- Version 3.x 3600+ ELO (proceed from HCE to neural network).

## Some terminal command examples
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
- cls -> clear the terminal.

## Thanks
Aron Petkovski. I used most of the tuned evaluation of an old version of Integral (3) to get me started (from chessnix 2.0 onwards I will do my own tuning).
Jonathan, Janez and Colin on Discord. The people of TalkChess, Zig Forum, CCRL.<br>
The authors of the engines I used for testing: Bbc, Cadie, Cheese, Chessplusplus, Colossus, Cwtch, Infrared, Integral, Lambergar, Linx, Lishex, Mess, Monty, OpenCritter, PlentyChess, Pounce, Priessnitz, Seawall, Seredina, Shallow, Shallowguess, Simbelmine, Stash, Supernova, Teki, Yakka.<br>
Some extra thanks to these people who made their engine support chess960.<br>
I also learned a lot from the source code of all these engines.<br>
Chessnix contains, besides my own inventions, a wild mix of ideas from the chess programming wiki as well as several other chess engines (Alexandria, Cwtch, Integral, PlentyChess, Pawnocchio, Sirius, Stockfish).

## The name
I spent many years on Lemmix, the DOS Lemmings clone. So in my feeling the name had to end with "ix" as well.
Later I found out a nix is a kind of water spirit. So then I added the queen water spirit logo.

## Noob
I am a github noob and only use a main branch as a backup of the current state on my PC.
With every release an extra zip of the source code at that moment is provided.
