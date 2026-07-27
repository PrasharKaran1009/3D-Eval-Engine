"""
4 - Generation Prep Scheduler
==============================

Takes ONE matchup at a time and resolves each side's GLB through the generator
for that model's source (see providers.py — the BYOA "open circuit"):

    - resolved -> enrich matchup with glb_a / glb_b paths, status = "ready"
    - missing  -> status = "missing"

For BYOM models, "resolve" means look the uploaded mesh up in the asset manifest
built by asset_indexer.py. For BYOA models, it means generate-on-demand + cache
(a documented stub today). The scheduler itself never branches on source — it
asks providers.get_generator() and calls resolve(), so registering a BYOA model
routes that cell through the BYOA generator with zero scheduler changes.

Usage:
    from scheduler import schedule, schedule_all
    matchup = schedule(matchup)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER3 = os.path.join(HERE, "..", "3 - Round Robin Pair Generator")
if LAYER3 not in sys.path:
    sys.path.insert(0, LAYER3)

from matchup import Matchup            # noqa: E402
import asset_indexer                   # noqa: E402  (sibling module, same folder)
import providers                       # noqa: E402  (generator dispatch — BYOA seam)

MANIFEST_PATH = os.path.join(HERE, "manifest.json")


def _roster_by_id(manifest: dict) -> dict:
    """model_id -> registry entry (source, endpoint, …). Falls back to the
    Layer 3 registry if the manifest didn't embed the roster."""
    roster = manifest.get("models") or asset_indexer.load_roster()
    return {m["id"]: m for m in roster}


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


def schedule(matchup: Matchup, manifest: dict | None = None,
             roster_by_id: dict | None = None) -> Matchup:
    """
    Resolve each side's GLB through its model's generator and enrich in place.

    Returns the same (mutated) matchup so it can be chained into Layer 5.
    A BYOA cell whose generator isn't wired yet raises through resolve(); we
    catch it and mark the side missing so one unconfigured model can't abort a
    whole mixed run.
    """
    manifest = manifest or load_manifest()
    roster = roster_by_id or _roster_by_id(manifest)

    def resolve(model_id: str) -> str | None:
        model = roster.get(model_id, {"id": model_id, "source": "BYOM"})
        gen = providers.get_generator(model.get("source", "BYOM"), manifest)
        try:
            return gen.resolve(model, matchup.prompt_id)
        except providers.BYOAGenerationNotConfigured as e:
            print(f"  [L4] {e}")
            return None

    matchup.glb_a = resolve(matchup.model_a)
    matchup.glb_b = resolve(matchup.model_b)
    matchup.status = "ready" if (matchup.glb_a and matchup.glb_b) else "missing"
    return matchup


def schedule_all(matchups: list[Matchup]) -> list[Matchup]:
    """Schedule a whole batch, loading the manifest + roster once."""
    manifest = load_manifest()
    roster = _roster_by_id(manifest)
    return [schedule(m, manifest, roster) for m in matchups]


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
