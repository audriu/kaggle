# Kaggriculture — training progress log

Competition: Kaggle "Kaggriculture" (farming simulation, 2 players, 30 in-game
days, final money = score). Deadline 2026-09-30. ~6,700 teams, $50,000 prize.
Leaderboard score is an Elo-style rating earned by playing ladder matches.

| date | milestone | local income* | Kaggle score | rank |
|---|---|---|---|---|
| Aug 5 | hand-written heuristic agent (carrot+goose), 2 submissions | ~13k | 296.6 / 300.7 | ~5,800 |
| Aug 26 | migrated to CUDA box; pipeline verified (bit-identical sim, 15.5× fast runner); **cem1** trained (300 gens); first trained θ submitted | 28k | 439–463 | 5,799 |
| Aug 27 (night) | **cem2**: 300 gens vs exploiter league ("crasher" price-flooder); melons unlocked, cows moved to day ~5 | 44.7k | 506.5 | 4,933 |
| Aug 27 | **cem3** (wider-sigma restart, 400 gens) submitted | 45.6k | 480.9 | — |
| Aug 27 | rollout-search v1 built + verified safe, but A/B showed it LOSES money → parked with diagnosis | — | — | — |
| Aug 27 (eve) | **cem4** (restart, 400 gens) submitted; **θ-v2**: policy grew 47→64 dials (season phases, sell caps, scarcity holds) | 52.4k | 509.5 | — |
| Aug 28 (night) | **cem5**: first 64-dial run (500 gens) | 51.0k | held back | — |
| Aug 28 | **Reassessment**: top of ladder ≈ 2,900–3,300 vs our ~500; local gains stopped translating → pivot to opponent intelligence | — | — | 4,770 |
| Aug 28 | **Replay mining**: crawled the matchmaking graph to the top-8 teams, downloaded 32 full replays; wrote notes/meta_report.md | — | — | — |
| Aug 28 | **Discovery**: entire ladder top plays a RANCH+GARDEN meta (9 cows + 4–6 sheep from day 0, zero geese, 12–14 rehired hands/day, 30–40 strawberries, town-drain-paced selling) earning **90–100k/episode**; our price model had measured the wrong variable | — | — | — |
| Aug 28 | hand-fit ranch θ: 5k → 24k after fixing the wheat-floor choke; **cem6** launched from that seed (climbing: gen 91 best 42.5k) | 24k→ | — | — |
| Aug 28 | **θ-v3**: +10 dials the meta needs (milk/wool/strawberry sell pacing, routine feed buying, feed cutoff, yarn-shop-adaptive sheep, strawberry fertilization, hire ramp); all tests green | — | — | — |

*local income = mean final money per episode in our own arena (opponents differ
per row; numbers are not directly comparable across rows, but the trend is).

## Where the result stands (Aug 28)

- **Best submitted bot**: cem4 — public score 509.5 and climbing, rank 4,770 of
  ~6,700. Every submission so far beat its predecessor locally by 90–100% win
  rates before upload.
- **Best local bot**: cem6 (in training) — learning the ranch meta discovered
  from top-team replays; already at 42.5k against a harder opponent league.
- **Key negative results** (documented, reproducible): inference-time rollout
  search loses ≈$0.4–2k/episode because value-network noise ($2.6–3.1k MAE)
  exceeds real plan differences ($0.5–0.9k) — parked with exact reopen
  conditions; and self-play-only training plateaus at ~500 Elo because it never
  observes the real meta.
- **Next**: cem6/7 to convergence on the ranch basin with θ-v3 dials, gate
  against cem4, submit, then freeze the best bot ~Sep 7–10 so its Elo can climb
  uninterrupted (top teams' scores include 1–2 weeks of undisturbed climbing).

## Verification discipline (what "done" meant at every step)

Every stage was gated: the fast simulator is proven bit-identical to the
official engine (difftest, 12/12 episodes); the parameterised policy reproduces
the hand-written agent decision-for-decision at default settings (episode-reward
equality); every schema extension (47→64→74 dials) is append-only with old
checkpoints padding forward; training checkpoints survive kill -9 mid-write;
every candidate submission passed a seat-swapped many-seed arena gate against
the previous champion before upload; and every claim in the meta report cites
specific replays.
