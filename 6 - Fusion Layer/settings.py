"""
Global Settings (updated.png · Registry Layer "Global Settings" panel)
======================================================================

One place for the value judgments the pipeline makes: fusion weights, the MCDA
method, hard-constraint thresholds, and the Pareto toggle. The diagram shows
these as user-configurable, so they live in a JSON file (settings.json) the
wizard can GET/PATCH — not as constants buried in fuse.py.

`load()` merges the on-disk file over DEFAULTS, so a partial settings.json (or
none at all) still yields a complete, valid config. That keeps older runs and a
fresh clone working without a migration step.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "settings.json")

# The neutral, documented baseline. Every field the fusion layer reads has a
# default here, so `load()` can never return a half-populated config.
DEFAULTS = {
    "fusion": {
        # subjective vs objective split; objective = 1 - subjective
        "subjective_weight": 0.60,
        # which aggregator turns the 2 criteria into one score
        "mcda_method": "SAW",          # SAW | TOPSIS | PROMETHEE | VIKOR | ELECTRE
        "pareto_filter": False,         # screen out dominated models before ranking
        "vikor_v": 0.5,                 # VIKOR strategy weight (0.5 = consensus)
        "electre_veto": 0.35,           # ELECTRE veto threshold on z-score gap
    },
    # weights INSIDE the subjective core (the 5 voted axes)
    "parameter_weights": {
        "prompt_alignment": 0.30,
        "texture_material": 0.20,
        "form_proportion": 0.20,
        "mesh_integrity": 0.15,
        "overall": 0.15,
    },
    # Step-1 hard constraints from the diagram. A model failing a REJECT rule is
    # ranked last (never silently dropped — a leaderboard must list everyone);
    # a PENALTY multiplies its objective score.
    "hard_constraints": {
        "reject_non_manifold": False,       # reject if non-manifold ratio > threshold
        "non_manifold_ratio_max": 0.05,
        "watertight_penalty": 0.85,          # x objective score if not watertight
        "min_faces": 100,                    # reject degenerate meshes below this
    },
    # objective-core dimension weights (mirror of 6B; surfaced for the wizard)
    "objective_weights": {
        "integrity": 0.50, "texturing": 0.35, "symmetry": 0.15, "efficiency": 0.0,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load(path: str = SETTINGS_PATH) -> dict:
    """Full settings = DEFAULTS deep-merged with settings.json (if present)."""
    if os.path.exists(path):
        with open(path) as f:
            return _merge(DEFAULTS, json.load(f))
    return json.loads(json.dumps(DEFAULTS))   # deep copy


def save(patch: dict, path: str = SETTINGS_PATH) -> dict:
    """Merge a partial patch into the current settings and persist."""
    current = load(path)
    merged = _merge(current, patch or {})
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


if __name__ == "__main__":
    print(json.dumps(load(), indent=2))
