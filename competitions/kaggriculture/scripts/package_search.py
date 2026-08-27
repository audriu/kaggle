"""Assemble the single-file search agent: dist/main_search.py.

Like scripts/export_policy.py but for the week-3 search stack: one flat file
containing agent/policy.py + sim/features.py's extract + agent/vnet_infer.py +
agent/vnet_weights.py (VNET_B64) + agent/reconstruct.py + agent/search.py + a
baked THETA from a CEM run, ending in

    value_fn = ValueFn.from_embedded(VNET_B64)
    agent = build_search_agent(THETA, value_fn)

Assembly rules (each one exists because of a measured/observed hazard):

* Top-level import lines are hoisted out of every included module and emitted
  once, deduplicated, with `from __future__ import annotations` first — a
  __future__ import in mid-file is a SyntaxError, and policy.py + vnet_infer.py
  both carry one.
* sim/features.py is NOT inlined flat: its CROPS/ANIMALS are LISTS and would
  shadow policy.py's CROPS/ANIMALS DICTS (policy._scan does CROPS.get(...) —
  a list there crashes every turn). It is embedded as a raw-string literal and
  exec'd into a private namespace; only `extract` and FEATURE_NAMES are pulled
  out. The packager asserts the source contains no ''' and no backslash (would
  corrupt the literal) and validates the inlined extract row-for-row against
  the dev sim.features.extract on a live mid-season observation.
* agent/policy.py's tail `agent = build(default_theta())` is stripped so the
  only module-level `agent` in the artifact is the search closure (load_agent
  resolves the name "agent" first).
* agent/search.py's dev-import block (`if "make_sim" not in globals():`) stays
  in the file but is skipped at exec time because reconstruct.py is inlined
  above it — that is what makes search.py dual-mode without any rewriting.

Validation (all on the written artifact, exec'd with no __file__ exactly like
Kaggle's loader):
  1. inlined-extract parity + FEATURE_NAMES equality vs sim/features.py,
  2. one FastEnv episode vs starter must beat the $3000 passive floor,
  3. one more episode with per-turn wall-clock capture: reports max/p99 turn
     time, total overage consumed (sum of (t - 1 s)+ — Kaggle's actTimeout is
     1 s with 60 s episode overage), and the search telemetry (candidates
     scored per turn, switches). Numbers printed on this box are 2-4x
     pessimistic while a 16-proc CEM run owns the cores.

Usage:
  python scripts/package_search.py                       # cem2 best -> dist/main_search.py
  python scripts/package_search.py --run train/runs/cem3 --out dist/main_search_cem3.py
"""

import argparse
import re
import statistics as stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import features as dev_features  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402
from train.checkpoint import CheckpointManager  # noqa: E402
from agent import policy  # noqa: E402

POLICY_TAIL = "agent = build(default_theta())"
IMPORT_RE = re.compile(r"^(?:from \S+ import .+|import \S+(?: as \S+)?)$")
FUTURE = "from __future__ import annotations"


def split_imports(src):
    """(hoisted top-level import lines, body without them). Only column-0
    single-line imports move — indented ones (search.py's dev-mode block) must
    stay where they are."""
    imports, body = [], []
    for line in src.splitlines():
        (imports if IMPORT_RE.match(line) else body).append(line)
    return imports, "\n".join(body)


def section(title, body):
    bar = "=" * (70 - len(title))
    return f"\n\n# ==== {title} {bar}\n\n{body.strip()}\n"


def assemble(theta, provenance):
    src_policy = (ROOT / "agent" / "policy.py").read_text()
    assert POLICY_TAIL in src_policy, "policy.py tail changed; update package_search.py"
    src_policy = src_policy.replace(POLICY_TAIL, "")

    src_features = (ROOT / "sim" / "features.py").read_text()
    assert "'''" not in src_features and "\\" not in src_features, \
        "features.py grew ''' or backslashes; the raw-string embedding would corrupt"
    assert not re.search(r"^(?:from|import) sim\b", src_features, re.M), \
        "features.py grew a sim-package import; strip it before embedding"

    src_vnet = (ROOT / "agent" / "vnet_infer.py").read_text()
    src_weights = (ROOT / "agent" / "vnet_weights.py").read_text()
    src_recon = (ROOT / "agent" / "reconstruct.py").read_text()
    src_search = (ROOT / "agent" / "search.py").read_text()

    imports, bodies = [], {}
    for name, src in [("agent/policy.py", src_policy),
                      ("agent/vnet_infer.py", src_vnet),
                      ("agent/reconstruct.py", src_recon),
                      ("agent/search.py", src_search)]:
        imp, body = split_imports(src)
        imports.extend(imp)
        bodies[name] = body
    imports = [i for i in dict.fromkeys(imports) if i != FUTURE]

    theta_lines = ",\n    ".join(
        f"{v!r},  # {n}" for v, n in zip(theta, policy.theta_names()))

    features_block = (
        "# Exec'd in a private namespace: features.py's CROPS/ANIMALS are lists\n"
        "# and would shadow policy.py's dicts if inlined flat (see packager doc).\n"
        f"_FEATURES_SRC = r'''\n{src_features}'''\n"
        "_FEATURES_NS = {}\n"
        'exec(compile(_FEATURES_SRC, "sim/features.py", "exec"), _FEATURES_NS)\n'
        'extract = _FEATURES_NS["extract"]\n'
        'FEATURE_NAMES = _FEATURES_NS["FEATURE_NAMES"]\n'
    )

    tail = (
        f"# {provenance}\n"
        f"THETA = [\n    {theta_lines}\n]\n"
        "value_fn = ValueFn.from_embedded(VNET_B64)\n"
        "agent = build_search_agent(THETA, value_fn)\n"
    )

    return (
        '"""kaggriculture search agent — GENERATED by scripts/package_search.py;'
        ' do not edit.\n\n'
        f"{provenance}\n"
        'Layers: policy (theta heuristic) -> features/value net -> obs->sim\n'
        'reconstruction -> once-a-day theta-candidate rollout search.\n'
        '"""\n\n'
        + FUTURE + "\n\n"
        + "\n".join(imports) + "\n"
        + section("agent/policy.py", bodies["agent/policy.py"])
        + section("sim/features.py", features_block)
        + section("agent/vnet_infer.py", bodies["agent/vnet_infer.py"])
        + section("agent/vnet_weights.py", src_weights)
        + section("agent/reconstruct.py", bodies["agent/reconstruct.py"])
        + section("agent/search.py", bodies["agent/search.py"])
        + section("baked theta + entrypoint", tail)
    )


def live_obs(day=10):
    """A real mid-season observation for the extract parity check."""
    env = FastEnv(4321)
    starter = load_agent("starter")
    for _ in range(day * 24):
        env.step([policy.agent(env.observation(0)), starter(env.observation(1))])
    return env.observation(0)


def main():
    ap = argparse.ArgumentParser(description="Package the rollout-search agent")
    ap.add_argument("--run", default=str(ROOT / "train" / "runs" / "cem2"))
    ap.add_argument("--out", default=str(ROOT / "dist" / "main_search.py"))
    ap.add_argument("--use-mu", action="store_true", help="bake mu instead of best_theta")
    ap.add_argument("--skip-timing", action="store_true",
                    help="skip the per-turn timing episode (dev iteration only)")
    args = ap.parse_args()

    cm = CheckpointManager(args.run)
    step, state = cm.load_best()
    if state is None:
        step, state = cm.load_latest()
    if state is None:
        sys.exit(f"no checkpoint in {args.run}")
    theta = state["mu"] if args.use_mu else state["best_theta"]
    provenance = (f"theta: {Path(args.run).name} gen {state['gen']} "
                  f"(fitness {state.get('best_fitness')}; "
                  f"{'mu' if args.use_mu else 'best_theta'})")

    src = assemble(theta, provenance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")

    # ---- 1. exec with no __file__ + inlined-extract parity -----------------
    ns = {}
    exec(compile(src, out.name, "exec"), ns)  # noqa: S102 - our own artifact
    assert "__file__" not in ns
    assert list(ns["FEATURE_NAMES"]) == list(dev_features.FEATURE_NAMES)
    obs = live_obs()
    row_pkg, row_dev = ns["extract"](obs), dev_features.extract(obs)
    assert row_pkg == row_dev, "inlined extract diverged from sim/features.py"
    v_pkg = ns["value_fn"].v(row_dev)
    print(f"extract parity OK on a live day-10 obs ({len(row_dev)} features); "
          f"embedded V(sample) = ${v_pkg:,.0f}")

    # ---- 2. smoke episode --------------------------------------------------
    fn = load_agent(str(out))
    starter = load_agent("starter")
    t0 = time.time()
    r = FastEnv(9999).run([fn, starter])
    print(f"smoke episode vs starter (seed 9999): {r}  [{time.time() - t0:.1f}s]")
    assert r[0] > 3000, "packaged agent scored below the passive baseline"
    tele = fn._search_state["telemetry"]
    assert tele and not any(e["overrun"] for e in tele), "search never ran or overran"

    # ---- 3. per-turn wall-clock episode ------------------------------------
    if args.skip_timing:
        return
    fn2 = load_agent(str(out))  # fresh exec -> fresh closure/state
    times = []

    def timed(obs):
        t = time.perf_counter()
        a = fn2(obs)
        times.append(time.perf_counter() - t)
        return a

    r2 = FastEnv(31337).run([timed, starter])
    assert r2[0] > 3000, r2
    times.sort()
    p99 = times[int(0.99 * (len(times) - 1))]
    overage = sum(t - 1.0 for t in times if t > 1.0)
    st = fn2._search_state
    scored = [e["n_scored"] for e in st["telemetry"]]
    switches = sum(e["switched"] for e in st["telemetry"])
    elapsed = [e["elapsed_s"] for e in st["telemetry"]]
    print(f"timing episode (seed 31337, reward {r2[0]:.0f}), {len(times)} turns:")
    print(f"  turn wall-clock  max {max(times) * 1e3:7.1f} ms   "
          f"p99 {p99 * 1e3:7.1f} ms   median {stats.median(times) * 1e3:7.3f} ms")
    print(f"  overage consumed (sum of (t-1s)+): {overage:.2f} s of 60 s")
    print(f"  searches {st['searches']}  candidates/turn "
          f"min/mean/max {min(scored)}/{stats.mean(scored):.1f}/{max(scored)}  "
          f"switches {switches}")
    print(f"  search-turn elapsed mean {stats.mean(elapsed) * 1e3:.0f} ms  "
          f"max {max(elapsed) * 1e3:.0f} ms (budget "
          f"{fn2._search_cfg['turn_budget_s'] * 1e3:.0f} ms)")
    print("  NOTE: measured on the loaded box (16-proc CEM in flight) — "
          "latencies are 2-4x pessimistic vs a dedicated Kaggle worker.")


if __name__ == "__main__":
    main()
