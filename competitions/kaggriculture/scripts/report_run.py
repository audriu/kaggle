"""Read-only run report / sensitivity tool for CEM training runs.

Answers, for any run dir under train/runs/, "how is training going and what has the
optimizer learned": fitness curve, theta drift away from the baseline defaults in
init-sigma units, exploration collapse (sigma vs init sigma), whether the NEW
capability params (melon/strawberry/sheep/fertilizer/plant-stop) have switched on,
league composition, and where best_theta disagrees with mu. This is the week-2
sensitivity report from notes/research_plan.md and the tool for watching live runs.

Safety design -- the trainer may be LIVE-writing the run dir while this runs:

  * We deliberately do NOT use train.checkpoint.CheckpointManager. Its __init__
    does mkdir(parents=True, exist_ok=True) (a write side effect that would also
    silently create a directory for a typo'd path), and its save path reaps tmp
    files and prunes checkpoints. load_latest() itself happens to be read-only,
    but bypassing the class entirely keeps this tool provably write-free.
  * Instead we re-read the manifest ourselves (json), walk its checkpoint entries
    newest-first, verify each blob's CRC32 exactly like the manager does, and
    unpickle the first one that verifies. The manifest is the commit point, and
    every checkpoint file is written tmp+fsync+atomic-rename, so any file we can
    open is internally consistent -- a torn read is impossible, only a *missing*
    file is (pruned between manifest read and open). That race falls back to the
    next-older entry; if every entry fails we re-read the manifest once (the
    trainer has certainly committed a newer one by then) and try again.
  * Nothing here opens any path inside the run dir for writing; --json refuses
    to write inside the run dir.

Episode counts are NOT stored in checkpoints (cem.py doesn't persist pop /
seeds_per), so the header reconstructs them from cem.py's schedule:
eps(gen) = (pop+2 candidates) * (2 fixed opponents + min(2, gen//league_every)
league members) * seeds_per * 2 seats. The pop/seeds/league-every flags default
to cem.py's defaults; override them if the run used different ones.

Usage:
  python scripts/report_run.py [train/runs/cem1] [--gens-rows 40] [--json out.json]
"""

import argparse
import json
import math
import pickle
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402

MANIFEST = "manifest.json"
SPARK = "▁▂▃▄▅▆▇█"
# Params marked NEW in agent/policy.py: capabilities the baseline never uses.
NEW_PREFIXES = ("melon_", "straw_", "sheep_", "fert_")
NEW_EXACT = ("plant_stop_day",)


def is_new(name):
    return name in NEW_EXACT or name.startswith(NEW_PREFIXES)


# ---------- read-only checkpoint loading ----------

def read_manifest(run_dir):
    p = run_dir / MANIFEST
    try:
        text = p.read_text()
    except FileNotFoundError:
        return None
    try:
        m = json.loads(text)
    except json.JSONDecodeError:
        # The manifest is replaced by atomic rename, so a torn read should be
        # impossible; treat a bad parse as "try again once" rather than a crash.
        time.sleep(0.2)
        try:
            m = json.loads(p.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    m.setdefault("checkpoints", [])
    return m


def try_load_entry(run_dir, entry):
    """CRC-verified unpickle of one manifest entry, or None. Never writes."""
    try:
        blob = (run_dir / entry["name"]).read_bytes()
    except (FileNotFoundError, OSError):
        return None  # pruned between manifest read and open -- fall back
    if zlib.crc32(blob) != entry.get("crc32"):
        return None
    try:
        return pickle.loads(blob)
    except Exception:
        return None


def load_latest(run_dir):
    """Newest checkpoint that verifies. Returns (entry, payload) or exits cleanly."""
    if not run_dir.is_dir():
        sys.exit(f"error: run dir {run_dir} does not exist (nothing was created)")
    for attempt in range(2):
        m = read_manifest(run_dir)
        if m is None:
            sys.exit(f"error: no readable {MANIFEST} in {run_dir} -- not a run dir?")
        for entry in reversed(m["checkpoints"]):
            payload = try_load_entry(run_dir, entry)
            if payload is not None:
                return entry, payload
        # Every listed file vanished or failed CRC: the trainer must have
        # committed a newer manifest meanwhile. Re-read once and retry.
        if attempt == 0:
            time.sleep(0.3)
    sys.exit(f"error: no checkpoint listed in {run_dir}/{MANIFEST} could be loaded")


# ---------- report pieces ----------

def sparkline(vals, width=60):
    if not vals:
        return ""
    if len(vals) > width:  # bucket-mean down to `width` characters
        k = len(vals) / width
        vals = [sum(vals[int(i * k):max(int(i * k) + 1, int((i + 1) * k))])
                / max(1, int((i + 1) * k) - int(i * k)) for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / span * 8))] for v in vals)


def downsample_rows(history, max_rows):
    """Every k-th gen plus always the last."""
    n = len(history)
    if n <= max_rows:
        return list(history)
    k = -(-n // max_rows)  # ceil
    rows = history[::k]
    if rows[-1] is not history[-1]:
        rows.append(history[-1])
    return rows


def activation_verdicts(mu_by_name):
    """[ON]/[off] for the NEW param groups, from how policy.py actually uses them.

    Rules read off agent/policy.py. Day is an INTEGER 0..29, so a day-gate like
    `day >= melon_day` only ever fires when ceil(melon_day) <= 29 -- e.g. a mu of
    29.6 is still fully off, not "almost on".
      melon:  BUY_SEED gate is `seeds < melon_seed_max` (any value > 0 buys) and
              `day >= melon_day`; PLANT additionally needs `day < plant_stop_day`
              (melon is a one-time crop), so some integer day must satisfy both.
      straw:  strawberry is ongoing, so planting is NOT gated by plant_stop_day;
              needs a reachable straw_day and straw_seed_max > 0.
      sheep:  BUY_ANIMAL gate is `day >= sheep_day and sheep_total < sheep_target`.
      plant_stop_day: one-time planting runs while `day < plant_stop_day`, so it
              only bites if some day 0..29 has day >= plant_stop_day.
      fert:   the code literally tests `fert_buy_max >= 1`.
    Cash-threshold params (sheep_cash, fert_cash) inherit their group's verdict.
    """
    p = mu_by_name
    on = {}
    melon_d0 = math.ceil(p["melon_day"])  # first integer day the gates can pass
    melon = melon_d0 <= 29 and melon_d0 < p["plant_stop_day"] and p["melon_seed_max"] > 0
    straw = math.ceil(p["straw_day"]) <= 29 and p["straw_seed_max"] > 0
    sheep = math.ceil(p["sheep_day"]) <= 29 and p["sheep_target"] > 0
    fert = p["fert_buy_max"] >= 1.0
    stop = p["plant_stop_day"] <= 29.0
    for name in p:
        if name.startswith("melon_"):
            on[name] = melon
        elif name.startswith("straw_"):
            on[name] = straw
        elif name.startswith("sheep_"):
            on[name] = sheep
        elif name.startswith("fert_"):
            on[name] = fert
        elif name == "plant_stop_day":
            on[name] = stop
    return on


def episodes_estimate(history, pop, seeds_per, league_every):
    return sum((pop + 2) * (2 + min(2, h["gen"] // league_every)) * seeds_per * 2
               for h in history)


def print_report(run_dir, entry, state, args):
    names = policy.theta_names()
    defaults = policy.default_theta()
    init_sig = policy.sigmas()
    mu, sigma = state["mu"], state["sigma"]
    best = state["best_theta"]
    history = state["history"]
    gens = state["gen"]

    # --- header ---
    walls = [h["elapsed"] for h in history]
    wall_all = sum(walls) / len(walls) if walls else 0.0
    wall_recent = sum(walls[-10:]) / len(walls[-10:]) if walls else 0.0
    eps = episodes_estimate(history, args.pop, args.seeds_per, args.league_every)
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("time", 0)))
    bf = state.get("best_fitness")
    print(f"run {run_dir}")
    print(f"  checkpoint     {entry['name']} (step {entry['step']}, committed {when}) "
          f"[direct manifest read; CheckpointManager bypassed]")
    print(f"  gens completed {gens}   wall/gen {wall_all:.1f}s (last10 {wall_recent:.1f}s)   "
          f"total wall {sum(walls) / 60:.1f}min")
    print(f"  episodes       ~{eps} (reconstructed: pop={args.pop}+2 ctrl, "
          f"seeds/gen={args.seeds_per}, both seats, league grows /{args.league_every} gens)")
    print(f"  best_fitness   {bf:.0f}" if bf is not None else "  best_fitness   (none yet)")

    # --- fitness curve ---
    print(f"\nfitness curve ({len(history)} gens, mu_fit sparkline):")
    print(f"  {sparkline([h['mu_fit'] for h in history])}")
    print(f"  {'gen':>5s} {'pop_mean':>9s} {'pop_best':>9s} {'mu_fit':>9s} "
          f"{'best_fit':>9s} {'sec':>5s}")
    for h in downsample_rows(history, args.gens_rows):
        print(f"  {h['gen']:5d} {h['mean_pop']:9.0f} {h['best_pop']:9.0f} "
              f"{h['mu_fit']:9.0f} {h['best_fit']:9.0f} {h['elapsed']:5.0f}")

    # --- theta drift ---
    mu_by_name = dict(zip(names, mu))
    verdict = activation_verdicts(mu_by_name)
    rows = []
    for i, name in enumerate(names):
        z = (mu[i] - defaults[i]) / init_sig[i]
        if abs(z) > 0.2 or is_new(name):
            rows.append((abs(z), i, name, z))
    rows.sort(key=lambda r: -r[0])
    print(f"\ntheta drift (|z| = |mu-default|/init_sigma > 0.2, plus all NEW params; "
          f"sig/init < ~{args.sigma_floor_note:.2f} = at the sigma floor):")
    print(f"  {'param':<18s} {'default':>8s} {'mu':>8s} {'z':>6s} "
          f"{'sigma':>7s} {'init_s':>7s} {'sig/init':>8s}  flags")
    for _, i, name, z in rows:
        flags = ""
        if is_new(name):
            flags = "NEW " + ("[ON]" if verdict.get(name) else "[off]")
        print(f"  {name:<18s} {defaults[i]:8.2f} {mu[i]:8.2f} {z:+6.2f} "
              f"{sigma[i]:7.2f} {init_sig[i]:7.2f} {sigma[i] / init_sig[i]:8.2f}  {flags}")

    # --- league ---
    league = state.get("league") or []
    lg = ", ".join(f"g{g}" for g, _ in league) or "(empty)"
    print(f"\nleague: {len(league)} member(s) snapshotted at {lg}")

    # --- best_theta vs mu ---
    diffs = sorted(
        ((abs(best[i] - mu[i]) / init_sig[i], i) for i in range(len(names))),
        key=lambda r: -r[0])
    print("\nbest_theta vs mu, top-10 |diff|/init_sigma:")
    print(f"  {'param':<18s} {'mu':>8s} {'best':>8s} {'dz':>6s}")
    for dz, i in diffs[:10]:
        if dz <= 0:
            break
        print(f"  {names[i]:<18s} {mu[i]:8.2f} {best[i]:8.2f} "
              f"{(best[i] - mu[i]) / init_sig[i]:+6.2f}")


def dump_json(path, run_dir, entry, state):
    out = Path(path).resolve()
    rd = run_dir.resolve()
    if out == rd or str(out).startswith(str(rd) + "/") or out.is_dir():
        sys.exit("error: refusing to write --json inside the run dir (read-only tool)")
    doc = {
        "run_dir": str(run_dir),
        "checkpoint": entry["name"],
        "step": entry["step"],
        "gen": state["gen"],
        "names": policy.theta_names(),
        "defaults": policy.default_theta(),
        "init_sigma": policy.sigmas(),
        "mu": list(map(float, state["mu"])),
        "sigma": list(map(float, state["sigma"])),
        "best_theta": list(map(float, state["best_theta"])),
        "best_fitness": state.get("best_fitness"),
        "league_gens": [g for g, _ in (state.get("league") or [])],
        "history": state["history"],
    }
    out.write_text(json.dumps(doc, indent=1))
    print(f"\njson -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Read-only report on a CEM run dir.")
    ap.add_argument("run_dir", nargs="?", default=str(ROOT / "train" / "runs" / "cem1"))
    ap.add_argument("--gens-rows", type=int, default=40,
                    help="max rows in the fitness table (downsampled, last gen kept)")
    ap.add_argument("--json", default=None, help="also dump history+theta as JSON here")
    # episode-count reconstruction (cem.py does not persist these; defaults match cem.py)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seeds-per", type=int, default=12)
    ap.add_argument("--league-every", type=int, default=5)
    ap.add_argument("--sigma-floor-note", type=float, default=0.15,
                    help="the trainer's --sigma-floor, only annotates the drift table")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    entry, state = load_latest(run_dir)
    if not isinstance(state, dict) or "mu" not in state or "history" not in state:
        sys.exit(f"error: {entry['name']} does not look like a cem.py state dict")
    if len(state["mu"]) != len(policy.PARAMS):
        sys.exit(f"error: checkpoint has {len(state['mu'])} params but agent/policy.py "
                 f"has {len(policy.PARAMS)} -- policy changed since this run?")
    print_report(run_dir, entry, state, args)
    if args.json:
        dump_json(args.json, run_dir, entry, state)


if __name__ == "__main__":
    main()
