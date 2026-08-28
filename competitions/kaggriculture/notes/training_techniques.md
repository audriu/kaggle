# How we trained a farming-game AI — the techniques, explained simply

This project builds an AI player for *Kaggriculture*, an online competition
where two virtual farmers share one market for 30 game-days and the winner is
whoever ends with more money. Here are the techniques we used, in the order we
used them, and why.

## 1. Start with a hand-written player ("heuristic")

Before any machine learning, we wrote a simple rule-based farmer by hand:
plant crops, water them, sell the harvest, buy some geese. This gave us a
working player, a score to beat, and — most importantly — something for the
learning system to start from, so learning began from "sensible" instead of
"random".

## 2. Make the game fast (the unglamorous step that enables everything)

The official game runner played one match in ~2.7 seconds — too slow to learn
from. We built a lean runner that skips the bookkeeping and calls the game's
own rules directly: ~16× faster, and we *proved* it plays identically to the
official one (same seeds → bit-for-bit identical games). All training stands
on this: if the fast copy drifted from the real game, everything learned on it
would be wrong.

## 3. Turn strategy into numbers ("parameterisation")

We rewrote the hand-made player so that every strategic choice — how many
workers to hire, which day to buy cows, at what price to sell milk — is a
number ("dial"). The full strategy became a list of 47 numbers (later grown to
64, then 74). With default numbers it behaves exactly like the hand-made
player; changing the numbers changes the strategy. Now "find a better
strategy" means "find better numbers" — a problem computers are good at.

## 4. Evolution instead of deep learning (CEM)

We used the **Cross-Entropy Method**, a simple evolutionary algorithm:

1. Create ~32 slightly-mutated copies of the current strategy.
2. Let each play many matches; measure average money earned.
3. Keep the best quarter ("elites"), average their numbers, add exploration
   noise, repeat — hundreds of generations, thousands of matches per
   generation across 16 CPU cores.

Why not a neural network with reinforcement learning? The game gives only ONE
reward per 720 turns, luck swings results by ±40%, and strategy fits in a few
dozen numbers. Evolution handles exactly this: it only compares "who did
better", so noise averages out.

## 5. Self-play leagues and a "bully" opponent

Training against a fixed opponent overfits to it. So the population trained
against a **league**: the basic bot, snapshots of its own past best versions
(self-play), and a hand-built **exploiter** — a "market bully" that floods the
shared market to crash prices. Facing the bully every generation forced the
learned strategies to stay robust to price wars instead of only prospering in
polite company.

## 6. Crash-proof training ("checkpointing")

Every generation, the full training state is saved atomically — even pulling
the plug mid-write loses nothing (we tested this literally, with kill -9). Any
run can be stopped and resumed with the same command. Overnight runs became
routine instead of risky.

## 7. Measure everything, trust nothing (evaluation discipline)

Luck is huge in this game, so every comparison used many random maps, with the
players swapping sides, and reported statistical error bars. A new bot was only
submitted after beating the previous champion under those rules. We also ran
honest **A/B tests** — one famous internal result: a sophisticated
"look-ahead search" feature we spent a day building measurably *lost* money
(the value estimates were noisier than the differences it was trying to
detect), so we shelved it and wrote down exactly why. Negative results were
kept, not hidden — they saved weeks.

## 8. A value network on the GPU (deep learning has a cameo)

We did train one neural network: given a snapshot of the farm mid-game, it
predicts the final money. Trained on 360,000 snapshots from 12,000 simulated
games, it reached 95% accuracy (R²) for late-game positions. It powered the
look-ahead experiment above; when that was shelved, the network stayed as a
diagnostic tool.

## 9. Scouting the opposition (the turning point)

After four training runs our bot kept improving *locally* but its ladder
rating flat-lined around 500 while the leaders sat at ~3,000. The reason: we
had only ever trained against ourselves. So we **downloaded and analysed 32
real replays of the top-8 teams** (public data), like a sports team studying
game film. The discovery: every top team plays the same "ranch" strategy —
buy cows and sheep on day one, hire a large daily workforce, grow strawberry
fields, and sell milk/wool in a careful trickle matched to the town's demand so
prices never crash. They earn about **double** what our best bot earned. Our
own market measurements had been misleading, because we had measured prices in
a world without the town's continuous buying.

## 10. Imitate, then optimise

We translated the ranch strategy into our strategy dials (imitation by
parameter-fitting), fixed the parts that stalled (one wrong sell-price floor
silently froze the whole early economy — found by tracing a single game day by
day), added new dials the ranch play needs (feed purchasing, per-item sales
pacing, reacting to which shops open), and then **handed the tuned starting
point back to evolution** — the algorithm now explores around the imitated
strategy instead of rediscovering farming from scratch.

## The one-line summary

Make the game fast and provably faithful → express strategy as numbers →
evolve them against increasingly hostile opponents → measure with error bars
and keep the negative results → and when self-play plateaus, study the real
champions and aim evolution at what actually wins.
