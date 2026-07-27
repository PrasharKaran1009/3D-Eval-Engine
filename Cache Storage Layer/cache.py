"""
Cache Storage Layer
======================

Receives a "ready" matchup from Layer 4 and, for each of its two models:

    1. Computes a cache dir:  cache/{prompt_id}/{model_id}/
    2. If metrics.json already exists -> reuse (no recompute)
    3. Else -> run metrics_extractor on the GLB, then write:
           model.glb       (copy of the source asset, self-contained cache)
           metrics.json    (raw geometry metrics)
           metadata.json   (provenance: ids, source path, size, timestamps)
    4. Attaches metrics + cache paths back onto the matchup.

Caching is keyed per (prompt, model), so although each model appears in many
matchups, its mesh is only analysed ONCE. 7 models x 5 prompts = 35 extractions
total, even across 105 matchups.

The enriched matchup (now carrying metrics_a/metrics_b) continues to 6A/6B.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import trimesh
import pymeshlab

import shutil

from metrics_extractor import extract_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER3 = os.path.join(HERE, "..", "3 - Round Robin Pair Generator")
if LAYER3 not in sys.path:
    sys.path.insert(0, LAYER3)
from matchup import Matchup            # noqa: E402

CACHE_ROOT = os.path.join(HERE, "cache")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_dir(prompt_id: str, model_id: str) -> str:
    return os.path.join(CACHE_ROOT, prompt_id, model_id)


def _process_one(prompt_id: str, model_id: str, model_name: str,
                 glb_path: str, force: bool = False) -> tuple[dict, str]:
    """
    Ensure one (prompt, model) is cached. Returns (metrics, cache_dir).

    Reuses an existing metrics.json unless force=True.
    """
    cdir = _cache_dir(prompt_id, model_id)
    metrics_path = os.path.join(cdir, "metrics.json")

    if os.path.exists(metrics_path) and not force:
        with open(metrics_path) as f:
            return json.load(f), cdir

    os.makedirs(cdir, exist_ok=True)
    metrics = extract_metrics(glb_path)

    # Cache the asset as-is so the entry is self-contained (6A reads from here).
    shutil.copy2(glb_path, os.path.join(cdir, "model.glb"))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "model_id": model_id,
        "model_name": model_name,
        "prompt_id": prompt_id,
        "source_glb": glb_path,
        "file_size_bytes": os.path.getsize(glb_path),
        "extracted_at": _now(),
        "tools": {
            "trimesh": trimesh.__version__,
            "pymeshlab": getattr(pymeshlab, "version", lambda: "unknown")()
            if hasattr(pymeshlab, "version") else "unknown",
        },
    }
    with open(os.path.join(cdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metrics, cdir


def process(matchup: Matchup, force: bool = False) -> Matchup:
    """
    Cache + extract metrics for both models of a matchup, enriching it in place.

    Skips matchups Layer 4 marked as not "ready".
    """
    if matchup.status not in ("ready", "cached"):
        return matchup

    metrics_a, cache_a = _process_one(
        matchup.prompt_id, matchup.model_a, matchup.name_a,
        matchup.glb_a, force=force)
    metrics_b, cache_b = _process_one(
        matchup.prompt_id, matchup.model_b, matchup.name_b,
        matchup.glb_b, force=force)

    matchup.metrics_a, matchup.cache_a = metrics_a, cache_a
    matchup.metrics_b, matchup.cache_b = metrics_b, cache_b
    matchup.status = "cached"
    return matchup


def process_all(matchups: list[Matchup], force: bool = False) -> list[Matchup]:
    return [process(m, force=force) for m in matchups]


if __name__ == "__main__":
    # Demo: cache the first matchup end-to-end.
    import generate_matchups
    sys.path.insert(0, os.path.join(HERE, "..", "4 - Generation Prep Scheduler"))
    import scheduler

    m = scheduler.schedule(generate_matchups.generate_matchups()[0])
    m = process(m)
    print(f"{m.prompt_id}: {m.name_a} vs {m.name_b}  [{m.status}]")
    print("metrics_a:", json.dumps(m.metrics_a, indent=2))
