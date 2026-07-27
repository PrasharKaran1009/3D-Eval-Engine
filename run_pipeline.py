"""
Pipeline runner: Layer 3 -> 4 -> 5
==================================

Drives the matchup object through the first three layers and writes the
enriched result. Each matchup is the single source of truth, carrying its data
from origin (Layer 3) through availability (Layer 4) to metrics (Layer 5).

    Layer 3  generate_matchups()  ->  105 matchups {model_a, model_b, prompt_id}
    Layer 4  scheduler.schedule() ->  + glb paths, status
    Layer 5  cache.process()      ->  + metrics_a/b, cache paths

Output: enriched_matchups.json (ready for 6A voting + 6B scoring).

Usage:
    python run_pipeline.py            # run all 105 matchups
    python run_pipeline.py --limit 3  # quick smoke test on first 3
    python run_pipeline.py --force    # ignore cache, recompute metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for layer in ("3 - Round Robin Pair Generator",
              "4 - Generation Prep Scheduler",
              "Cache Storage Layer"):
    p = os.path.join(ROOT, layer)
    if p not in sys.path:
        sys.path.insert(0, p)

import generate_matchups          # noqa: E402  Layer 3
import scheduler                  # noqa: E402  Layer 4
import cache                      # noqa: E402  Layer 5

OUT_PATH = os.path.join(ROOT, "enriched_matchups.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Layers 3->4->5")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N matchups (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute metrics even if cached")
    args = ap.parse_args()

    # --- Layer 3 -----------------------------------------------------------
    matchups = generate_matchups.generate_matchups()
    if args.limit:
        matchups = matchups[:args.limit]
    print(f"[L3] generated {len(matchups)} matchups")

    # --- Layer 4 -----------------------------------------------------------
    matchups = scheduler.schedule_all(matchups)
    ready = sum(1 for m in matchups if m.status == "ready")
    print(f"[L4] scheduled: {ready} ready, {len(matchups) - ready} missing")

    # --- Layer 5 -----------------------------------------------------------
    print(f"[L5] extracting metrics (cached per prompt/model)...")
    done = 0
    for m in matchups:
        cache.process(m, force=args.force)
        done += 1
        if done % 10 == 0 or done == len(matchups):
            print(f"      {done}/{len(matchups)} matchups cached")

    # --- Output ------------------------------------------------------------
    with open(OUT_PATH, "w") as f:
        json.dump([m.to_dict() for m in matchups], f, indent=2)

    cached = sum(1 for m in matchups if m.status == "cached")
    print(f"\nDone. {cached}/{len(matchups)} matchups fully enriched.")
    print(f"Enriched matchups -> {OUT_PATH}")
    print(f"Per-model cache   -> {os.path.join('Cache Storage Layer', 'cache')}/")


if __name__ == "__main__":
    main()
