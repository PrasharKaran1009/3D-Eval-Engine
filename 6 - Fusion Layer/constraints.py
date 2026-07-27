"""
Layer 7 - Hard Constraints + Pareto Filter  (updated.png · Fusion Steps 1-2)
============================================================================

Step 1 (Hard Constraints) and Step 2 (Pareto Filter) of the diagram's Fusion
layer. Both operate on raw geometry, so this module reads the Layer 5 cache
directly (the same source 6B scores from) and is driven by Layer 7 settings.

Hard constraints, per the diagram:
    - Broken / degenerate mesh  -> REJECT   (ranked last, never silently dropped)
    - Not watertight            -> PENALTY  (objective score x factor)
    - Non-manifold beyond a ratio -> optional REJECT
A rejected model still appears on the leaderboard — a benchmark must account for
every entrant — but is forced below all feasible models and tagged with a reason.

Pareto filter (optional): flags models dominated on BOTH criteria by some other
model. We only FLAG (never drop): the leaderboard still ranks everyone; the flag
just marks who sits off the efficient frontier.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "Cache Storage Layer", "cache")


def _load_metrics() -> dict:
    """-> {model_id: [metrics_dict per test case]} from the Layer 5 cache."""
    out = defaultdict(list)
    for mpath in glob.glob(os.path.join(CACHE, "*", "*", "metrics.json")):
        mid = os.path.basename(os.path.dirname(mpath))
        with open(mpath) as f:
            out[mid].append(json.load(f))
    return out


def evaluate(models: list, cfg: dict) -> dict:
    """
    Apply hard constraints to each model across its test cases.

    Returns {model_id: {"reject": bool, "penalty": float, "reasons": [...]}}.
    `penalty` is a multiplier applied to the objective score (1.0 = untouched).
    Aggregated across a model's meshes: a model is rejected if ANY of its meshes
    fails a REJECT rule; the penalty is the harshest applied to any mesh.
    """
    hc = cfg.get("hard_constraints", {})
    metrics = _load_metrics()
    verdict = {}
    for mid in models:
        rows = metrics.get(mid, [])
        reject, penalty, reasons = False, 1.0, []
        for m in rows:
            faces = m.get("face_count") or 0
            if faces < hc.get("min_faces", 0):
                reject = True
                reasons.append(f"degenerate mesh ({faces} faces < {hc['min_faces']})")
            if hc.get("reject_non_manifold"):
                nm = ((m.get("non_manifold_edges") or 0)
                      + (m.get("non_manifold_vertices") or 0)) / max(faces, 1)
                if nm > hc.get("non_manifold_ratio_max", 1.0):
                    reject = True
                    reasons.append(f"non-manifold ratio {nm:.3f} > "
                                   f"{hc['non_manifold_ratio_max']}")
            if not m.get("is_watertight"):
                penalty = min(penalty, hc.get("watertight_penalty", 1.0))
                if "not watertight" not in reasons:
                    reasons.append("not watertight")
        verdict[mid] = {"reject": reject, "penalty": penalty,
                        "reasons": sorted(set(reasons))}
    return verdict


def pareto_flags(criteria: dict) -> dict:
    """
    criteria: {model: {"subjective": x, "objective": y}} (both higher-better).
    -> {model: dominated_bool}. m is dominated if another model is >= on both
    criteria and strictly > on at least one.
    """
    alts = list(criteria)
    flags = {}
    for a in alts:
        ca = criteria[a]
        dominated = False
        for b in alts:
            if a == b:
                continue
            cb = criteria[b]
            ge = (cb["subjective"] >= ca["subjective"] and cb["objective"] >= ca["objective"])
            gt = (cb["subjective"] > ca["subjective"] or cb["objective"] > ca["objective"])
            if ge and gt:
                dominated = True
                break
        flags[a] = dominated
    return flags
