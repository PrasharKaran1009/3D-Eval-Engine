"""
Layer 3 - Pair Generation (Round Robin)
=======================================

Generates all unique pairwise comparisons (nC2) for a set of models.

For n models this yields n(n-1)/2 unique unordered pairs:
    n = 7  ->  7 * 6 / 2  =  21 pairs

This layer is intentionally stateless and dependency-free. It reads the model
roster from `models.json` (the single source of truth for n + names) and emits
the round-robin schedule that every downstream layer (5A voting, 5B metrics,
6 fusion) iterates over.

Model identity:
    Each model has a stable `id` (M1..Mn) used as the join key everywhere
    downstream, and a human-readable `name` used only for display. IDs never
    change even if names are edited, which keeps cached votes/metrics valid.

Usage (CLI):
    python pair_generator.py                 # reads models.json, writes pairs.json
    python pair_generator.py --out pairs.json

Usage (import):
    from pair_generator import load_models, generate_pairs
    models = load_models()
    pairs = generate_pairs([m["id"] for m in models])
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

# --- Config -----------------------------------------------------------------
# Roster lives in models.json next to this script — the single source of truth
# for n + ids + display names. There is deliberately no hardcoded model count
# here, so the two can never drift out of sync.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models.json")
# ---------------------------------------------------------------------------


def load_models(config_path: str = CONFIG_PATH) -> list[dict]:
    """
    Load the model roster from models.json.

    Returns a list of {"id", "name"} dicts. Validates that the declared `n`
    matches the number of entries and that all ids are unique.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    models = cfg.get("models", [])
    declared_n = cfg.get("n", len(models))

    if declared_n != len(models):
        raise ValueError(
            f"models.json: declared n={declared_n} but found {len(models)} entries"
        )
    ids = [m["id"] for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError(f"models.json: model ids must be unique, got {ids}")
    return models


def generate_pairs(models: list[str]) -> list[tuple[str, str]]:
    """
    Generate all unique unordered pairs (round-robin) from a list of models.

    Order is deterministic: it follows the input order, so the schedule is
    reproducible across runs (important for caching and resumable voting).
    """
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models to form a pair, got {len(models)}")
    if len(set(models)) != len(models):
        raise ValueError(f"Model names must be unique, got duplicates in: {models}")
    return list(combinations(models, 2))


def expected_pair_count(n: int) -> int:
    """n(n-1)/2 — the number of unique pairs for n models."""
    return n * (n - 1) // 2


def build_schedule(models: list[dict]) -> dict:
    """
    Build a serializable schedule object from a roster of {"id", "name"} dicts.

    Each pair carries a stable `pair_id` plus both ids and display names, so
    votes/metrics can key on the ids while the UI can show readable labels.
    """
    name_by_id = {m["id"]: m["name"] for m in models}
    ids = [m["id"] for m in models]
    pairs = generate_pairs(ids)
    return {
        "n_models": len(models),
        "models": models,
        "pair_count": len(pairs),
        "pairs": [
            {
                "pair_id": idx,
                "a": a, "b": b,
                "a_name": name_by_id[a], "b_name": name_by_id[b],
            }
            for idx, (a, b) in enumerate(pairs)
        ],
    }


def save_schedule(schedule: dict, out_path: str) -> None:
    """Write the schedule to a JSON file (pretty-printed)."""
    with open(out_path, "w") as f:
        json.dump(schedule, f, indent=2)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3 - Round Robin Pair Generator")
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help="Path to models.json roster (default: models.json next to this script)",
    )
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "pairs.json"),
        help="Output JSON path (default: pairs.json next to this script)",
    )
    args = parser.parse_args()

    models = load_models(args.config)
    schedule = build_schedule(models)
    save_schedule(schedule, args.out)

    print(f"Models ({schedule['n_models']}):")
    for m in models:
        print(f"  {m['id']}: {m['name']}")
    print()
    print(f"Unique pairs (nC2): {schedule['pair_count']}  "
          f"[expected {expected_pair_count(len(models))}]")
    print(f"Schedule written to: {args.out}")
    print()
    for p in schedule["pairs"]:
        print(f"  [{p['pair_id']:>2}]  {p['a_name']}  vs  {p['b_name']}")


if __name__ == "__main__":
    _main()
