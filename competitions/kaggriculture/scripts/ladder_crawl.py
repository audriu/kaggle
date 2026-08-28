"""Crawl the Kaggle ladder's matchmaking graph and download top-bot replays.

Why: plan §9 reassessment (2026-08-28) — our league is self-referential and local
gains stopped translating to the ladder (Elo-like score; matchmaking by rating).
The only source of truth about the meta above ~500 rating is real episodes.

Mechanics (all public, no auth):
  POST www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes
       {"submissionId": N} -> every episode that submission played, with both
       agents' submissionId/teamId/reward/initialScore/updatedScore.
  GET  www.kaggleusercontent.com/episodes/<episodeId>.json  (browser UA required)
       -> full replay: configuration + 720 steps of state/actions.

BFS: start from our submissionIds, always expand the highest-rated unvisited
submission seen so far; a bot's episode list spans its whole climb, so each hop
can jump hundreds of rating points. Stops when the frontier is exhausted or
--max-subs listings have been made. Everything is cached in tmp_logs/ (gitignored)
so re-runs are incremental; ~1 request/second to stay polite.

Usage:
  python scripts/ladder_crawl.py --seed-subs 55825553,55817229 --max-subs 60
  python scripts/ladder_crawl.py --download-top 8 --eps-per-team 4   # after a crawl
"""

import argparse
import heapq
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tmp_logs"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


PACE_S = 4.0  # unconditional gap between requests; Kaggle 429s aggressive crawlers


def _get(url, data=None, tries=5):
    """Paced + backoff fetch. EVERY request sleeps first (the 429 incident that
    motivated this: an error path that skipped the polite sleep turned one bad
    listing into a hammering loop)."""
    for attempt in range(tries):
        time.sleep(PACE_S * (1 + attempt))
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        if data is not None:
            req.add_header("Content-Type", "application/json")
            req.data = json.dumps(data).encode()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                wait = 90 * (attempt + 1)
                print(f"    429; backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def list_episodes(sub_id):
    p = CACHE / "listings" / f"sub-{sub_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    d = _get(LIST_URL, {"submissionId": int(sub_id)})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return d


def crawl(seed_subs, max_subs):
    """Best-first crawl toward high ratings. Returns {sub_id: info}."""
    index_path = CACHE / "ladder_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    seen_best = {}   # sub_id -> best observed score
    heap = []        # (-score, sub_id)
    for s in seed_subs:
        heapq.heappush(heap, (0.0, int(s)))
    visited = set(int(k) for k in index)
    listed = 0

    while heap and listed < max_subs:
        _, sub = heapq.heappop(heap)
        if sub in visited:
            continue
        visited.add(sub)
        try:
            d = list_episodes(sub)
        except Exception as e:
            print(f"  sub {sub}: {e}", file=sys.stderr)
            continue
        listed += 1
        eps = d.get("episodes", [])
        my_scores, my_team = [], None
        for ep in eps:
            for ag in ep.get("agents", []):
                aid, sc = ag.get("submissionId"), ag.get("updatedScore") or 0.0
                if aid is None:
                    continue
                if aid == sub:
                    my_scores.append(sc)
                    my_team = ag.get("teamId")
                else:
                    if sc > seen_best.get(aid, -1):
                        seen_best[aid] = sc
                        if aid not in visited:
                            heapq.heappush(heap, (-sc, aid))
        top = max(my_scores) if my_scores else 0.0
        index[str(sub)] = {
            "team": my_team, "best_score": top, "n_eps": len(eps),
            "episodes": [
                {"id": ep["id"], "end": ep.get("endTime"),
                 "agents": [{"sub": a.get("submissionId"), "score": a.get("updatedScore"),
                             "reward": a.get("reward"), "team": a.get("teamId")}
                            for a in ep.get("agents", [])]}
                for ep in eps],
        }
        print(f"[{listed:3d}] sub {sub:<9d} team {my_team} best {top:7.1f} "
              f"eps {len(eps):3d} frontier {len(heap)}")
        index_path.write_text(json.dumps(index))
    return index


def download_top(index, n_teams, eps_per_team):
    """Full replays for the best-rated submissions' most recent episodes."""
    ranked = sorted(index.items(), key=lambda kv: -kv[1]["best_score"])
    picked, teams = [], set()
    for sub, info in ranked:
        if info["team"] in teams:
            continue
        teams.add(info["team"])
        eps = sorted(info["episodes"], key=lambda e: e.get("end") or "", reverse=True)
        picked.append((sub, info, eps[:eps_per_team]))
        if len(picked) >= n_teams:
            break
    out = CACHE / "replays"
    out.mkdir(parents=True, exist_ok=True)
    for sub, info, eps in picked:
        print(f"team {info['team']} sub {sub} best {info['best_score']:.1f}: "
              f"{len(eps)} replays")
        for ep in eps:
            p = out / f"episode-{ep['id']}.json"
            if p.exists():
                continue
            try:
                d = _get(f"https://www.kaggleusercontent.com/episodes/{ep['id']}.json")
                p.write_text(json.dumps(d))
                print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")
            except Exception as e:
                print(f"  episode {ep['id']}: {e}", file=sys.stderr)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-subs", default="55825553,55817229,55808651")
    ap.add_argument("--max-subs", type=int, default=60)
    ap.add_argument("--download-top", type=int, default=0)
    ap.add_argument("--eps-per-team", type=int, default=4)
    args = ap.parse_args()

    index = crawl([int(s) for s in args.seed_subs.split(",") if s], args.max_subs)
    best = sorted(index.values(), key=lambda v: -v["best_score"])[:10]
    print("\ntop crawled:", [(v["team"], round(v["best_score"], 1)) for v in best])
    if args.download_top:
        download_top(index, args.download_top, args.eps_per_team)


if __name__ == "__main__":
    main()
