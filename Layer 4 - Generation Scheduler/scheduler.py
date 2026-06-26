"""
Layer 4 - Generation Scheduler
==============================

Takes ONE matchup at a time and resolves whether both models' GLBs already
exist for that prompt:

    - exist  -> enrich matchup with glb_a / glb_b paths, status = "ready"
    - missing -> status = "missing"  (placeholder for future live API generation)

In this prototype all 35 GLBs are pre-provided, so every matchup passes through
as "ready". The lookup uses the asset manifest built by asset_indexer.py (the
registry of what has been "generated").

Usage:
    from scheduler import schedule, schedule_all
    matchup = schedule(matchup)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER3 = os.path.join(HERE, "..", "Layer 3 - Pair Generation")
if LAYER3 not in sys.path:
    sys.path.insert(0, LAYER3)

from matchup import Matchup            # noqa: E402
import asset_indexer                   # noqa: E402  (sibling module, same folder)

MANIFEST_PATH = os.path.join(HERE, "manifest.json")


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    """
    Load the asset manifest {assets: {prompt_id: {model_id: glb_path}}}.

    If it doesn't exist yet, build it on the fly from the TEST CASES folder so
    the scheduler is usable standalone.
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: build the index live.
    roster = asset_indexer.load_roster()
    index, problems = asset_indexer.build_index(asset_indexer.DEFAULT_ROOT, roster)
    if problems:
        raise RuntimeError(
            "Cannot build manifest cleanly:\n  " + "\n  ".join(problems)
        )
    return {"assets": index}


def _lookup(assets: dict, prompt_id: str, model_id: str) -> str | None:
    return assets.get(prompt_id, {}).get(model_id)


def schedule(matchup: Matchup, manifest: dict | None = None) -> Matchup:
    """
    Resolve GLB availability for one matchup and enrich it in place.

    Returns the same (mutated) matchup so it can be chained into Layer 5.
    """
    assets = (manifest or load_manifest()).get("assets", {})

    glb_a = _lookup(assets, matchup.prompt_id, matchup.model_a)
    glb_b = _lookup(assets, matchup.prompt_id, matchup.model_b)

    matchup.glb_a = glb_a
    matchup.glb_b = glb_b

    if glb_a and glb_b:
        matchup.status = "ready"          # both present -> proceed to Layer 5
    else:
        # Placeholder for live generation: in the real system this is where a
        # missing model's output would be generated via API and cached.
        matchup.status = "missing"
    return matchup


def schedule_all(matchups: list[Matchup]) -> list[Matchup]:
    """Schedule a whole batch, loading the manifest once."""
    manifest = load_manifest()
    return [schedule(m, manifest) for m in matchups]


def _main() -> None:
    # Demo: schedule the matchups from Layer 3 and report readiness.
    import generate_matchups
    matchups = schedule_all(generate_matchups.generate_matchups())
    ready = sum(1 for m in matchups if m.status == "ready")
    missing = [m for m in matchups if m.status != "ready"]
    print(f"Scheduled {len(matchups)} matchups: {ready} ready, {len(missing)} missing")
    for m in matchups[:3]:
        print(f"  [{m.matchup_id:>3}] {m.prompt_id} {m.status}: "
              f"{os.path.basename(m.glb_a or '?')}  vs  "
              f"{os.path.basename(m.glb_b or '?')}")
    if missing:
        print("Missing examples:")
        for m in missing[:5]:
            print(f"  {m.prompt_id} {m.model_a}/{m.model_b}")


if __name__ == "__main__":
    _main()
