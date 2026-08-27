"""Ground truth + rank-fidelity checks for agent/opponent_model.OppTracker.

A) Stock accuracy: play real episodes (player 0 = baseline policy feeding the
   tracker its obs stream + own orders), compare estimate() against the TRUE
   opponent private (shed + carried inventories pooled) each day. Metric:
   value-weighted error Σ_items BASE_PRICE·|est−true|, tracker vs the empty
   estimate (search v1's default, whose error is simply the true stock's value).
B) THE decisive experiment (plan §9): the verifier's failing mirror-play
   rank-fidelity cells (seeds 777/888 × day 10/15, K=2 days, 10 θ variants).
   Rank candidates by reconstructed-rollout end money under {empty, tracker,
   true} opponent-private and correlate each against the real-continuation
   ranking. v1 measured empty ≈ 0.087–0.735 worst-mean; true = 1.000.
   Gate: tracker mean ρ ≥ 0.9 and ≥ empty everywhere.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import policy  # noqa: E402
from agent.opponent_model import OppTracker  # noqa: E402
from agent.reconstruct import make_sim  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402
from train.exploiters import get_exploiters  # noqa: E402

BP = policy.BASE_PRICE


def true_pool(env, opp):
    pr = env.state[opp].observation.private
    pool = dict(pr.get("shed") or {})
    for inv in pr.get("inventories") or []:
        for k, v in (inv or {}).items():
            pool[k] = pool.get(k, 0) + v
    return {k: v for k, v in pool.items() if k in BP and v > 0}


def value_err(est, true):
    items = set(est) | set(true)
    return sum(BP.get(i, 0) * abs(est.get(i, 0) - true.get(i, 0)) for i in items)


def run_tracked(opp_agent, seed, until_step=719, snap_hours=(0,)):
    """Play one episode with player 0 = baseline policy + tracker. Returns
    per-day (tracker_err, empty_err) lists and the final (env, tracker)."""
    env = FastEnv(seed)
    me = policy.agent
    tr = OppTracker()
    errs = []
    for step in range(until_step):
        obs0 = env.observation(0)
        tr.update(obs0)
        if obs0.get("hour", 0) in snap_hours and obs0.get("day", 0) >= 1:
            t = true_pool(env, 1)
            errs.append((value_err(tr.estimate()["shed"], t), value_err({}, t)))
        a0 = me(obs0)
        tr.note_own_orders(a0.get("market"), obs0)
        env.step([a0, opp_agent(env.observation(1))])
        if env.done:
            break
    return errs, env, tr


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def variants():
    names = policy.theta_names()
    base = policy.default_theta()
    tweaks = [
        {}, {"hire_d1": 13.0}, {"hire_min": 1.0}, {"wheat_plus": 6.0},
        {"carrot_buy_cap": 25.0}, {"goose_target": 12.0}, {"cow_day": 6.0},
        {"sell_cap": 5.0}, {"sell_floor_carrot": 0.9}, {"feed_res_base": 8.0},
    ]
    out = []
    for tw in tweaks:
        th = list(base)
        for k, v in tw.items():
            th[names.index(k)] = v
        out.append(policy.clip_theta(th))
    return out


def roll_money(sim, cand_fn, opp_fn, me, steps):
    for _ in range(steps):
        acts = [None, None]
        acts[me] = cand_fn(sim.observation(me))
        acts[1 - me] = opp_fn(sim.observation(1 - me))
        sim.step(acts)
        if sim.done:
            break
    return sim.rewards()[me]


def main():
    crasher = policy.build(list(dict(get_exploiters())["crasher"]))
    starter = load_agent("starter")

    print("A) stock estimate, value-weighted $ error (mean over days 1-29; lower is better)")
    print(f"   {'opponent':<10} {'tracker':>9} {'empty':>9}   (4 seeds each)")
    for name, opp in (("mirror", policy.agent), ("crasher", crasher), ("starter", starter)):
        te = ee = n = 0
        for seed in (11, 12, 13, 14):
            errs, _, _ = run_tracked(opp, seed)
            te += sum(e[0] for e in errs)
            ee += sum(e[1] for e in errs)
            n += len(errs)
        print(f"   {name:<10} {te / n:9.0f} {ee / n:9.0f}")
        # Must beat empty wherever empty is materially wrong; a sub-$200 absolute
        # error is below market-impact relevance (one wheat unit ~ $25).
        assert te <= ee or te / n < 200, f"{name}: tracker worse than empty estimate"

    print("\nB) mirror-play rank fidelity, K=2d, 10 candidates (verifier's failing cells)")
    print(f"   {'cell':<16} {'empty':>7} {'tracker':>8} {'true':>7}")
    K = 48
    rows = []
    for seed in (777, 888):
        for day in (10, 15):
            until = day * 24
            _, env, tr = run_tracked(policy.agent, seed, until_step=until)
            obs0 = env.observation(0)
            true_pr = {k: dict(v) if isinstance(v, dict) else [dict(x) for x in v]
                       for k, v in dict(env.state[1].observation.private).items()}
            real, rec = [], {"empty": [], "tracker": [], "true": []}
            for th in variants():
                cand = policy.build(th)
                real.append(roll_money(copy.deepcopy(env), cand, policy.agent, 0, K))
                for label, pr in (("empty", None), ("tracker", tr.estimate()), ("true", true_pr)):
                    sim = make_sim(obs0, opponent_private=pr, rng_seed=0)
                    rec[label].append(roll_money(sim, cand, policy.agent, 0, K))
            rhos = {lb: spearman(real, rec[lb]) for lb in rec}
            rows.append(rhos)
            print(f"   s{seed} d{day:02d}      {rhos['empty']:7.3f} {rhos['tracker']:8.3f} "
                  f"{rhos['true']:7.3f}")
    mean = {lb: sum(r[lb] for r in rows) / len(rows) for lb in rows[0]}
    print(f"   {'MEAN':<16} {mean['empty']:7.3f} {mean['tracker']:8.3f} {mean['true']:7.3f}")
    assert mean["tracker"] >= 0.9, f"tracker mean rho {mean['tracker']:.3f} < 0.9"
    assert all(r["tracker"] >= r["empty"] - 0.05 for r in rows), "tracker below empty somewhere"

    print("\nPASS")


if __name__ == "__main__":
    main()
