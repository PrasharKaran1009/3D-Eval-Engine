"""
Layer 6A - Hybrid Judge Resolver   (VLM first pass -> Human review -> Final vote)
=================================================================================

Implements the "Final vote" box of updated.png's Hybrid flow. The two halves
already exist:

    * VLM first pass  -> vlm_judge.py writes ballots tagged voter="vlm:<model>"
    * Human review    -> the blind pair viewer writes ballots tagged voter="human"
                          (pre-filled with the VLM's ballot so the human ACCEPTS,
                          MODIFIES, or OVERWRITES it — see preview.html)

This module resolves them into exactly ONE final ballot per matchup using the
hybrid rule (same rule aggregate.py enforces, made explicit + auditable here):

    human review present  -> human ballot wins   (decided_by = "human-override")
    only a human ballot    -> human ballot         (decided_by = "human")
    only a VLM ballot      -> VLM ballot stands     (decided_by = "vlm")

It writes final_votes.json and prints a coverage summary so you can see how much
of the VLM first pass a human has actually reviewed.

Usage:
    python3 hybrid.py                 # resolve votes.json -> final_votes.json
    python3 hybrid.py --votes path --out path
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import aggregate  # sibling module — reuse _is_vlm so the rule lives in one place

HERE = os.path.dirname(os.path.abspath(__file__))
VOTES_PATH = os.path.join(HERE, "votes.json")
OUT_PATH = os.path.join(HERE, "final_votes.json")


def _latest(recs: list[dict]) -> dict:
    """Most recent ballot by timestamp (falls back to last seen)."""
    return max(recs, key=lambda r: r.get("ts", ""))


def resolve(records: list[dict]) -> tuple[list[dict], dict]:
    by_matchup: dict = defaultdict(list)
    for r in records:
        by_matchup[r["matchup_id"]].append(r)

    final: list[dict] = []
    summary = {"total": 0, "human_override": 0, "human_only": 0, "vlm_only": 0}

    for mid, recs in sorted(by_matchup.items()):
        humans = [r for r in recs if not aggregate._is_vlm(r)]
        vlms = [r for r in recs if aggregate._is_vlm(r)]
        if humans and vlms:
            decided, chosen = "human-override", _latest(humans)
            summary["human_override"] += 1
        elif humans:
            decided, chosen = "human", _latest(humans)
            summary["human_only"] += 1
        else:
            decided, chosen = "vlm", _latest(vlms)
            summary["vlm_only"] += 1
        summary["total"] += 1
        final.append({
            "matchup_id": mid,
            "prompt_id": chosen["prompt_id"],
            "model_a": chosen["model_a"], "model_b": chosen["model_b"],
            "ballots": chosen["ballots"],
            "voter": chosen.get("voter", "?"),
            "decided_by": decided,
            "vlm_first_pass": _latest(vlms)["ballots"] if vlms else None,
            "ts": chosen.get("ts"),
        })
    return final, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="6A hybrid resolver -> final_votes.json")
    ap.add_argument("--votes", default=VOTES_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.votes):
        raise SystemExit(f"{args.votes} not found — collect votes first "
                         "(viewer and/or vlm_judge.py).")
    records = json.load(open(args.votes))
    final, summary = resolve(records)
    json.dump({"rule": "human review overrides VLM first pass",
               "summary": summary, "final_votes": final},
              open(args.out, "w"), indent=2)

    rev = summary["human_override"] + summary["human_only"]
    print("=== Hybrid resolution (VLM first pass -> human review -> final) ===")
    print(f"  matchups resolved : {summary['total']}")
    print(f"  human-reviewed    : {rev}"
          f"  (override={summary['human_override']}, human-only={summary['human_only']})")
    print(f"  VLM-only (unreviewed): {summary['vlm_only']}")
    if summary["total"]:
        pct = 100.0 * rev / summary["total"]
        print(f"  human review coverage: {pct:.0f}%")
    print(f"\n[out] final votes -> {args.out}")


if __name__ == "__main__":
    main()
