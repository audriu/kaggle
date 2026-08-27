# Kaggriculture — self-learning agent: research log & development plan

*2026-08-26 (evening). Status: migrated to the CUDA box (RTX 3060 Ti, torch 2.13.0+cu130,
CUDA verified); all §6 gates passed (difftest 12/12 bit-identical, 15.5×; policy equivalence;
checkpoint suite 6/6). First trained θ (cem1 gen 38) SUBMITTED to Kaggle. Exploiter league,
value-net groundwork and run-report tool built + adversarially verified — see §4b/§8.*

Goal: win the Kaggle **Kaggriculture** simulation competition (deadline ~30 Sep 2026) with a
**self-learning, fully resumable** training system. Original constraints: one dev + AI assistant,
16-core CPU box (WSL2), RTX 3060 8 GB (currently unusable — see §6), ~35 days.

---

## 1. Measured facts (everything below was measured, not assumed)

### Game shape
- 2 players, 720 steps (30 days × 24 turns). Reward = own final money. Start $3000.
- **Farms are fully independent.** The only player coupling is the shared market
  (prices/inventory) plus an RNG subtlety: the per-day weed RNG stream is consumed by farm 0
  then farm 1 before the town-shop-unlock draw, so the opponent shifts your weeds *and* which
  shops unlock. Same seed ≠ same world across opponents ⇒ common-random-number pairing barely
  works (paired-seed correlation measured at only +0.10..+0.52).
- Branching: 8–22 meaningful actions/unit, up to 11 units ⇒ 1e24–1e28 joint/turn (StarCraft
  scale, not Go). But the space factorises per unit + market list.
- Eval noise: mean 13035, sd 4781 over 40 seeds (sd/mean ≈ 37%; up to 56% in self-play).
- Inference budget on Kaggle: `actTimeout=1 s`/turn, `remainingOverageTime=60 s`/episode,
  `runTimeout=1200 s` for the whole 2-agent episode (⇒ budget ~0.5 s/turn sustained).

### Simulation cost
- `env.run()` ≈ 2.7 s/episode, ~96–98% framework overhead (deepcopy 39%, structify ~25%,
  jsonschema re-validation 21%); the rules interpreter is only 0.09–0.25 ms/step.
- **FastEnv** (same interpreter, no framework): 0.168 s/episode — 16×; 47 eps/s on 16 procs.
- Audit estimates (measured probes): flat pure-Python fork 25–40k steps/s/core;
  numpy-batched N=4096 ≈ 150–250k steps/s/core (~3000–4900 eps/s on 14 cores);
  GPU only 2–5× over 14 cores (bandwidth-bound workload) and needs N≥16k batches.
- CPython MT19937 is reproducible in numpy (`RandomState`) bit-exactly; weed RNG uses a fresh
  `Random((seed*1_000_003)^day)` each day ⇒ no RNG state carries across days (resume-friendly).

### Economics (the crux)
Price after selling N units (base at inventory 10 000):

| item | 0 | 50 | 100 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|---|
| WHEAT | 25 | 22 | 21 | 21 | 20 | 20 | 19 |
| EGG | 50 | 43 | 42 | 41 | 40 | 39 | 38 |
| CARROT | 35 | 27 | 23 | 19 | 15 | 9 | 1 |
| TOMATO | 60 | 42 | 35 | 24 | 16 | 3 | 1 |
| MELON | 250 | 225 | 150 | 1 | 1 | 1 | 1 |
| STRAWBERRY | 120 | 24 | 1 | 1 | 1 | 1 | 1 |
| MILK / WOOL | 160/200 | 55/55 | 1 | 1 | 1 | 1 | 1 |

Only WHEAT and EGG are liquid at volume. Town demand drains inventory and pushes prices *up*
(e.g. STRAWBERRY 120→308 at −500), so scarcity is rewarded. Key experiments:

- Baseline agent crashes CARROT 35→$14 by day 11 and holds ~$30–900 cash for 20 days.
- **Sell-restraint sweep (3840 eps): every hold/dribble variant loses** (up to −7000). The fix
  is the production mix, not the sales schedule — holding produce starves working capital.
- **Capital sweep**: start-$1k→4237, $3k→7467, $12k→17060, $50k→56000. Marginal capital converts
  at 2.18× around $12k and exactly 1.00× above $25k ⇒ sharply capital-constrained early, and the
  strategy engine itself plateaus at ~$6k/season earnings — big headroom for a learner.
- Opponent sensitivity: vs pass 12809 / vs starter 13382 / vs itself 8385 (−35%, pure price
  competition) ⇒ league self-play is mandatory, mirror-only would overfit.
  **CORRECTION (2026-08-26, twice independently re-measured on fresh seed blocks):** the
  mirror number is far worse than first recorded — baseline-vs-itself is **~3.1–3.7k**
  (3097±342 on seed_block(0,40); 3164±279 on seeds 200000+; 3677±415 on seeds 300000+),
  and baseline-vs-starter on those same blocks is 8.6–9.7k, not 13k (block-to-block drift
  is huge; always re-measure references on the candidate's own seeds). Price competition
  costs the baseline ~2/3 of its income, not 35%.
- Raising baseline hand target 8→10 collapses it 13035→1416 (cash starvation, not shed
  overflow — measured discards ≈ 0).

## 2. Decision: CEM over a structured policy + self-play league

- **Not deep RL**: one reward per 720 steps, 37–56% noise, CPU-only training, 1e24 branching.
- **Not GPU-sim-first**: CUDA absent on this box; 1–2 weeks of fidelity-risky work; 47 eps/s
  already trains a generation in ~40 s.
- **Yes CEM**: strategy is ~46 numbers (portfolio, schedules, thresholds); rank-based CEM is
  robust to exactly this noise; embarrassingly parallel; trivially resumable.
- AlphaGo/AlphaStar reference: what transfers is *self-play + league* (AlphaStar) and the idea
  of a *learned value fn over macro-decisions* (AlphaGo, optional later stage). MCTS over raw
  actions does not transfer (branching + 1 s/turn).

## 3. What is built (all tested)

```
competitions/kaggriculture/
  sim/fastenv.py        lean runner: same interpreter, no framework; 16× faster
  sim/difftest.py       proves bit-identical trajectories vs reference (20 eps × 720 steps)
  sim/arena.py          parallel league eval; seat-swapped; SE-reporting; 47 eps/s
  agent/policy.py       46-param policy; default θ == baseline EXACTLY (10/10 episode
                        rewards equal); melon/strawberry/sheep/fertilizer unlockable
  train/checkpoint.py   atomic ckpt (tmp+fsync+rename+manifest, CRC, RNG capture)
  train/test_checkpoint.py  6/6 incl. real kill -9 mid-write
  train/test_policy.py  equivalence test
  train/cem.py          CEM trainer + league; resumable (verified live);
                        --exploiters/--no-exploiters (default ON) adds fixed exploiters
  train/exploiters.py   "crasher" price-flood league member (see §4b); make_theta
                        raises on out-of-bounds/unknown overrides
  train/test_exploiters.py  bounds/name/50-step sanity
  sim/features.py       112-dim act-time-legal feature vector from a live obs
                        (audited vs env source: own private + public opponent only)
  train/collect.py      parallel npz dataset collector; idempotent, bit-reproducible
                        shards; snapshot loop proven reward-identical to plain runs
  train/value_net.py    torch MLP on CUDA; delta-residual head V=money+net(x);
                        split by episode id; atomic model saves
  train/test_features.py    finiteness / no-obs-mutation sanity
  scripts/report_run.py read-only live-run reporter: fitness curve + sparkline,
                        θ drift in z-units, NEW-param [ON]/[off] verdicts, --json;
                        race-safe direct manifest reads (strace/audit-hook verified)
```

### Commands
```bash
source ../../.venv/bin/activate           # from competitions/kaggriculture
python sim/difftest.py 720 1,2,3          # sim fidelity vs reference
python train/test_policy.py               # θ-default == baseline
python train/test_checkpoint.py           # crash-safety suite
python sim/arena.py agent/main.py --seeds 100   # evaluate any agent file
python train/cem.py --gens 60 --pop 24 --seeds-per 12 --outdir train/runs/cem1
# interrupt any time (Ctrl-C, reboot, kill -9); SAME command resumes:
python train/cem.py --gens 200 --pop 24 --seeds-per 12 --outdir train/runs/cem1
python scripts/export_policy.py --run train/runs/cem1   # → dist/main.py
python scripts/submit.py -m "cem1 genNN"
```

### Resumability design (hard requirement)
Checkpoint per generation: `{gen, mu, sigma, best_theta, best_fitness, league, history, rng}`.
Write = tmp file → fsync → atomic rename → manifest rewrite (manifest is the commit point) →
dir fsync; CRC32 verified on load with fallback to older checkpoints; `keep=5` pruned only
after commit. Seed blocks derive from gen index, so a resumed run evaluates on exactly the
seeds it would have used. Survives `kill -9` mid-write (tested with a real SIGKILL).

## 4. First results

### 4a. Run `cem1` (WSL2 box, then re-run fresh on the CUDA box — no exploiters in league)
Old box, gen 0→11: population mean 12k→21k, best ~29k. New box (fresh run, 2026-08-26):
gen 38 best_fitness 31.7k. **Gate eval of exported gen-38 θ** (fresh seeds 400000+, both
seats, n=100/opponent): **vs baseline 28,051 (96% win); vs crasher 27,501 (94%); vs starter
33,450 (100%)** — reference baseline on the same seeds: 6,148 mean overall, 4,854 (22%) vs
crasher. The trained θ is ~4.5× the baseline's income and, untrained against it, already
crasher-robust. **Submitted 2026-08-26** ("CEM-trained theta (cem1 gen38)"). Optimizer
direction at gen ~19 (scripts/report_run.py): wheat portfolio up (+3.1z), hiring down,
cows earlier, land buffer down, premium sell floors up; no NEW capability activated yet
(melon/straw/sheep/fert day-gates still unreachable); sigma at floor on ~10 params.

### 4b. Exploiter "crasher" (train/exploiters.py; 6 experiment rounds, ~1000 eps)
Kamikaze floods (14 hires, carrot floods) bankrupt themselves without depressing the
baseline further. The efficient attack is a season-long **tomato flood** (ongoing crop,
one $50 seed flowers all season; 60→$3 at 500 sold) + full staple dumping, with income
parked in flood-proof goods (eggs early, milk early, strawberries mid) and late spending
throttled. Result (pooled n=200): baseline held to **3,660 (≈ mirror level, −62% vs its
starter income; ~15% winrate)** while the crasher banks **~8,975**. Independently
reproduced by a verifier on fresh seeds (3,041±317 / 8,974±527).

### 4c. Value-net groundwork (GPU live)
112 features, 480 episodes / 28,800 rows collected (idempotent shards). MLP on the
RTX 3060 Ti (63 s / 300 epochs): val R² 0.384 overall vs −0.02 for predict-mean;
per-day-bucket R² rises 0.09 (day 0–4, mostly irreducible) → 0.79 (day 25–29).
Bottleneck is dataset size (best epoch 15 of 300 = memorization); collector extends
in place at ~7 eps/s. Next: 5–10k episodes + day-conditioned head before wiring into
rollout truncation.

## 5. Roadmap

| Week | Deliverable (each ends submittable) | Cut order |
|---|---|---|
| 1 (now) | pipeline ✅, first trained θ ✅, submit ✅ (2026-08-26) | — |
| 2 | overnight runs ✅ (cem1 300 gens → cem2 300 gens w/ exploiter, auto-chained), exploiter league member ✅ (crasher), sensitivity report ✅ (scripts/report_run.py) | — |
| 3 | flat fast sim (25–40k steps/s/core) → inference-time rollout search (~8 season rollouts/turn) | cut 2nd |
| 4 | value-net V(state)→final money on GPU; macro-plan search truncated by V | cut 1st |
| 5 | freeze, wide-seed robustness eval, final 2 submissions | — |

Tripwires: if CEM fitness plateaus < baseline+30% by week 2 → widen sigma_floor / add params
before adding ML; if trained θ beats league but loses seat-swapped vs baseline → league too
narrow, add exploiters; always trust `sim/arena.py` numbers only (SE-reported, both seats).

## 6. Continuing on a CUDA device

Nothing in the pipeline needs a GPU; it makes optional stages (batched sim, value net) viable.

1. **Transfer**: clone the repo; `python3.12 -m venv .venv && pip install -r requirements.txt`.
   To continue a run, also copy `competitions/kaggriculture/train/runs/<name>/` (gitignored;
   plain pickles, portable) — then the *same* `train/cem.py --outdir` command resumes it.
   Without the runs dir, training restarts from the baseline θ (still fine, ~1 h to re-pass it).
2. **Verify the box**: `python sim/difftest.py 720 1,2,3` must print all-OK before trusting
   any training on a new machine/python/kaggle-environments version.
3. **CUDA check**: `nvidia-smi` inside WSL/host; `pip install torch --index-url
   https://download.pytorch.org/whl/cu124`; `python -c "import torch;
   print(torch.cuda.is_available())"`. (This dev box had no CUDA userspace — that was the
   blocker, not hardware.)
4. **What the GPU unlocks** (in order of value):
   a. *Value net*: regress (state features, day) → final money on arena episode dumps;
      use as rollout-truncation for inference-time search (week-4 stage).
   b. *Batched numpy/torch sim* for ~100× training throughput — follow the audit: 1-D linear
      tile indexing, prefix-sum market (closed form verified), MT19937 via `RandomState`,
      per-unit-slot loop must stay sequential; keep the CPU FastEnv as the fidelity arbiter
      (differential test, random action streams).
5. **More cores** scale linearly today: `--procs N` (arena saturates ~0.34 s/ep/core).

## 7. Submission mechanics (unchanged)

5 submissions/day, latest 2 scored. `dist/main.py` is self-contained (policy.py + baked θ);
`scripts/submit.py` validates before upload. Keep `agent/main.py` (heuristic baseline) as the
fallback submission at all times. Auth on this box is the new-style `~/.kaggle/access_token`
(kaggle CLI 2.2.4), not kaggle.json.

## 8. Overnight results (2026-08-26 → 27) & current state

The chained pipeline ran clean. **cem1** (no exploiters) finished gen 300 at best 42.6k.
**cem2** (crasher in league) finished gen 300 at best_fitness 42.2k *on the harder league*;
gate eval on fresh seeds 500000+ (both seats, n=100/opp): **vs cem1-g38 44,580 (91% win);
vs baseline 45,461 (100%); vs starter 45,202 (100%); vs crasher 43,367 (100%)** — cem1-g38
reference mean on the same seeds was 31.3k. **Submitted 2026-08-27** ("cem2 g300").
Leaderboard after ~6 h of the first submission: public 462.9 (old heuristic: 300.7),
rank 5799 → 4933 and climbing.

What cem2 learned beyond cem1 (scripts/report_run.py): **melons ACTIVATED** (first NEW
capability: melon_day 31→28.3, melon_seed_max 0→2.2), cows radically earlier (day 14→4.6,
−3.1z), fewer geese (target 6→1.5), higher feed reserves, and sell floors raised across the
board — a scarcity-respecting posture learned from facing the crasher. Still off: sheep,
strawberries (seed_max drifted +2.9z but the day gate never crossed), fertilizer.

Next (in flight / queued):
- value-net dataset at scale: `train/collect.py --center-run train/runs/cem2` →
  `train/data_cem2/` (6k episodes; strong-play states, since the net will truncate
  rollouts of the TRAINED agent) — then retrain train/value_net.py, expect the
  memorization bottleneck (best epoch 15/300 on 480 eps) to lift.
- week-3 stage: inference-time rollout search truncated by V(state); needs the flat
  fast sim OR the value net to be good enough that shallow rollouts suffice.
- cem3 candidates: wider sigma restart from cem2 μ (sigma floor was pinning ~30 params),
  a second exploiter targeting melons/eggs (cem2's new income), longer league memory.
