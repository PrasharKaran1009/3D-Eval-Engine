"""
Layer 6A - Aggregation : votes.json  ->  per-parameter Bradley-Terry ELOs
=========================================================================

Turns the collected ballots into ELO scores, exactly per the 6A design rules:

  - BRUTAL SUBSTITUTION: if a human voted a matchup, that ballot is used and the
    VLM-judge ballot for the same matchup is dropped (no weighting, no compare).
  - Per (test case, parameter) we build a pairwise win matrix. A 'tie' vote adds
    0.5 to each side (Bradley-Terry has no native draw).
  - Bradley-Terry strengths are fit with the MM algorithm + a light anchor prior
    (regularization), so a model that wins/loses every pair can't run off to
    +/-infinity, and a disconnected graph still resolves.
  - Strengths map onto an ELO scale centered at BASELINE = 1300.

RANKING uses the 'overall' axis only. The other four axes are computed and kept
as reference/diagnostics -- they are NOT the ranking signal.

Pure standard-library; no numpy needed (matrices are tiny: ~7 models).

Usage:
    python3 aggregate.py                 # read votes.json, print + write elos.json
    python3 aggregate.py --votes path    # custom votes file
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VOTES_PATH = os.path.join(HERE, "votes.json")
OUT_PATH = os.path.join(HERE, "elos.json")

BASELINE = 1300.0                     # every model is anchored here; gaps carry meaning
SCALE = 400.0 / math.log(10.0)        # logistic strength -> ELO points
RANK_PARAM = "overall"                # only this axis decides the leaderboard
ANCHOR_GAMES = 1.0                    # regularization: virtual even games vs a fixed anchor
MAX_ITERS = 2000
EPS = 1e-9


# ---------------------------------------------------------------- load + substitute
def load_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- collect votes first (viewer or vlm_judge.py).")
    with open(path) as f:
        return json.load(f)


def _is_vlm(rec: dict) -> bool:
    return str(rec.get("voter", "")).startswith("vlm")


def apply_substitution(records: list[dict]) -> list[dict]:
    """
    Brutal substitution at the matchup level: if ANY human ballot exists for a
    matchup, keep only the human ballot(s) for it and drop the VLM's.
    """
    by_matchup: dict = defaultdict(list)
    for r in records:
        by_matchup[r["matchup_id"]].append(r)

    kept = []
    for recs in by_matchup.values():
        humans = [r for r in recs if not _is_vlm(r)]
        kept.extend(humans if humans else recs)   # human present -> humans win; else VLM
    return kept


# ---------------------------------------------------------------- build win matrices
def build_pair_wins(records: list[dict]) -> dict:
    """
    -> wins[prompt_id][param][(winner, loser)] = float
    A 'tie' contributes 0.5 to each direction.
    """
    wins: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for r in records:
        tc, a, b = r["prompt_id"], r["model_a"], r["model_b"]
        for param, choice in r.get("ballots", {}).items():
            w = wins[tc][param]
            if choice == "A":
                w[(a, b)] += 1.0
            elif choice == "B":
                w[(b, a)] += 1.0
            elif choice == "tie":
                w[(a, b)] += 0.5
                w[(b, a)] += 0.5
    return wins


# ---------------------------------------------------------------- Bradley-Terry (MM)
def fit_bradley_terry(pair_wins: dict) -> dict:
    """
    pair_wins[(i, j)] = times i beat j (ties already split 0.5/0.5).
    Returns {model: ELO}. Anchor prior keeps every strength finite + connected.
    """
    players = sorted({m for pair in pair_wins for m in pair})
    if not players:
        return {}
    if len(players) == 1:
        return {players[0]: BASELINE}

    total_wins = {m: 0.0 for m in players}
    for (i, j), w in pair_wins.items():
        total_wins[i] += w

    n = defaultdict(float)                         # games between each unordered pair
    for (i, j), w in pair_wins.items():
        n[frozenset((i, j))] += w

    p = {m: 1.0 for m in players}                  # strengths; anchor fixed at 1.0
    for _ in range(MAX_ITERS):
        new_p, max_delta = {}, 0.0
        for i in players:
            denom = ANCHOR_GAMES / (p[i] + 1.0)    # virtual games vs the fixed anchor
            for j in players:
                if j == i:
                    continue
                nij = n.get(frozenset((i, j)), 0.0)
                if nij:
                    denom += nij / (p[i] + p[j])
            wi = total_wins[i] + 0.5 * ANCHOR_GAMES
            new_p[i] = wi / denom if denom > 0 else p[i]
            max_delta = max(max_delta, abs(new_p[i] - p[i]))
        p = new_p
        if max_delta < EPS:
            break

    logs = {m: math.log(v) for m, v in p.items()}
    mean = sum(logs.values()) / len(logs)          # center the field on BASELINE
    return {m: BASELINE + SCALE * (logs[m] - mean) for m in players}


# ---------------------------------------------------------------- orchestrate
def compute(records: list[dict]) -> dict:
    kept = apply_substitution(records)
    wins = build_pair_wins(kept)

    out: dict = {"by_test_case": {}, "overall_leaderboard": []}
    overall_acc: dict = defaultdict(list)

    for tc in sorted(wins):
        out["by_test_case"][tc] = {}
        for param in sorted(wins[tc]):
            elos = fit_bradley_terry(wins[tc][param])
            out["by_test_case"][tc][param] = {m: round(e, 1) for m, e in elos.items()}
            if param == RANK_PARAM:
                for m, e in elos.items():
                    overall_acc[m].append(e)

    # cross-test-case leaderboard = mean Overall ELO (the ranking signal)
    board = [{"model": m, "overall_elo": round(sum(v) / len(v), 1), "test_cases": len(v)}
             for m, v in overall_acc.items()]
    board.sort(key=lambda r: r["overall_elo"], reverse=True)
    out["overall_leaderboard"] = board
    return out


def _print_summary(out: dict) -> None:
    board = out["overall_leaderboard"]
    print("\n=== 6A Leaderboard (ranked by mean Overall ELO) ===")
    if board:
        for rank, row in enumerate(board, 1):
            print(f"  {rank:>2}. {row['model']:<6} {row['overall_elo']:>7.1f}  "
                  f"({row['test_cases']} test case(s))")
    else:
        print("  (no 'overall' votes yet -- ranking needs the 'overall' axis)")

    print("\n=== Per-parameter ELOs (reference / diagnostics) ===")
    for tc, params in out["by_test_case"].items():
        print(f"\n  {tc}")
        for param, elos in params.items():
            ranked = sorted(elos.items(), key=lambda kv: kv[1], reverse=True)
            tag = "  <- RANKS" if param == RANK_PARAM else ""
            print(f"    {param}{tag}")
            print("      " + "  ".join(f"{m}:{e:.0f}" for m, e in ranked))


def main() -> None:
    ap = argparse.ArgumentParser(description="6A: votes.json -> Bradley-Terry ELOs")
    ap.add_argument("--votes", default=VOTES_PATH, help="path to votes.json")
    ap.add_argument("--out", default=OUT_PATH, help="where to write elos.json")
    args = ap.parse_args()

    records = load_records(args.votes)
    out = compute(records)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    _print_summary(out)
    print(f"\n[out] ELOs -> {args.out}")


if __name__ == "__main__":
    main()
