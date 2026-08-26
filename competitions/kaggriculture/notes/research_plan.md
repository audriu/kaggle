# Kaggriculture — self-learning agent: research log & development plan

*2026-08-26. Status: training pipeline live; first CEM run in progress.*

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
  train/cem.py          CEM trainer + league; resumable (verified live)
  scripts/export_policy.py  best θ → single dist/main.py, Kaggle-style validated
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

## 4. First results (run `cem1`, in progress)

Baseline (600 eps): mean 10368 ± 238, win 78.2% (vs starter 93.5%, vs self 48.5%).
CEM gen 0→11: population mean 12k→21k, μ-control ~25k, best-so-far ~29k on mixed league
opposition. Watch: `feed_res_*`, `goose_target`, `hire_*`, and whether melon/sheep/fertilizer
params switch on.

## 5. Roadmap

| Week | Deliverable (each ends submittable) | Cut order |
|---|---|---|
| 1 (now) | pipeline ✅, first trained θ, submit | — |
| 2 | overnight runs (100s of gens), exploiter league member (price-crasher), sensitivity report | — |
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
fallback submission at all times.
