"""Export a trained theta into a single submittable main.py.

Takes the best checkpoint from a training run, embeds the theta into a copy of
agent/policy.py, validates it Kaggle-style (exec with no __file__), and runs one
episode as a smoke test. Output: dist/main.py.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.fastenv import FastEnv, load_agent  # noqa: E402
from train.checkpoint import CheckpointManager  # noqa: E402

TAIL = "agent = build(default_theta())"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "train" / "runs" / "cem1"))
    ap.add_argument("--out", default=str(ROOT / "dist" / "main.py"))
    ap.add_argument("--use-mu", action="store_true", help="export mu instead of best_theta")
    args = ap.parse_args()

    cm = CheckpointManager(args.run)
    step, state = cm.load_best()
    if state is None:
        step, state = cm.load_latest()
    if state is None:
        sys.exit(f"no checkpoint in {args.run}")
    theta = state["mu"] if args.use_mu else state["best_theta"]
    fit = state.get("best_fitness")

    src = (ROOT / "agent" / "policy.py").read_text()
    assert TAIL in src, "policy.py tail changed; update export_policy.py"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    theta_lines = ",\n    ".join(
        f"{v!r},  # {n}" for v, n in zip(theta, __import__("agent.policy", fromlist=["x"]).theta_names())
    )
    body = src.replace(TAIL, (
        f"# Exported {stamp} from {Path(args.run).name} gen {state['gen']} "
        f"(fitness {fit})\nTHETA = [\n    {theta_lines}\n]\nagent = build(THETA)"
    ))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)

    # Validate the artifact exactly as Kaggle loads it.
    fn = load_agent(str(out))
    r = FastEnv(9999).run([fn, load_agent("starter")])
    print(f"exported {out} ({out.stat().st_size} bytes), gen {state['gen']}, fitness {fit}")
    print(f"smoke episode vs starter (seed 9999): {r}")
    assert r[0] > 3000, "exported agent scored below passive baseline"


if __name__ == "__main__":
    main()
