"""
Layer 3 - Matchup Generator
===========================

Round-robin over models, crossed with every prompt, to emit the full set of
matchup objects that flow downstream.

    21 model-pairs (7C2)  x  5 prompts  =  105 matchups

Each matchup is the core unit {model_a, model_b, prompt_id} plus stable ids and
display names. Output: matchups.json (a list of Matchup dicts).

Usage:
    python generate_matchups.py
    from generate_matchups import generate_matchups
    matchups = generate_matchups()
"""

from __future__ import annotations

import json
import os

from pair_generator import load_models, generate_pairs
from matchup import Matchup

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(HERE, "prompts.json")
OUT_PATH = os.path.join(HERE, "matchups.json")


def load_prompts(path: str = PROMPTS_PATH) -> list[dict]:
    """Load [{'id','description'}, ...] from prompts.json."""
    with open(path) as f:
        prompts = json.load(f)["prompts"]
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"prompts.json: prompt ids must be unique, got {ids}")
    return prompts


def generate_matchups() -> list[Matchup]:
    """Build all (pair x prompt) matchups as enriched-from-birth objects."""
    models = load_models()
    name_by_id = {m["id"]: m["name"] for m in models}
    ids = [m["id"] for m in models]
    pairs = generate_pairs(ids)            # list of (a, b) tuples
    prompts = load_prompts()

    matchups: list[Matchup] = []
    mid = 0
    for prompt in prompts:
        for pair_id, (a, b) in enumerate(pairs):
            matchups.append(Matchup(
                matchup_id=mid,
                prompt_id=prompt["id"],
                pair_id=pair_id,
                model_a=a, model_b=b,
                name_a=name_by_id[a], name_b=name_by_id[b],
            ))
            mid += 1
    return matchups


def save_matchups(matchups: list[Matchup], path: str = OUT_PATH) -> None:
    with open(path, "w") as f:
        json.dump([m.to_dict() for m in matchups], f, indent=2)


def _main() -> None:
    models = load_models()
    prompts = load_prompts()
    matchups = generate_matchups()
    save_matchups(matchups)

    pairs_per_prompt = len(matchups) // len(prompts)
    print(f"Models:   {len(models)}")
    print(f"Prompts:  {len(prompts)}  ({', '.join(p['id'] for p in prompts)})")
    print(f"Pairs/prompt: {pairs_per_prompt}")
    print(f"Total matchups: {len(matchups)}  "
          f"(= {pairs_per_prompt} x {len(prompts)})")
    print(f"Written to: {OUT_PATH}\n")
    print("Sample (first 3):")
    for m in matchups[:3]:
        print(f"  [{m.matchup_id:>3}] {m.prompt_id}  "
              f"{m.name_a}  vs  {m.name_b}")


if __name__ == "__main__":
    _main()
