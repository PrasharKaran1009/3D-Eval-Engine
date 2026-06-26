"""
Layer 7 - Fusion
================

Combines the two cores into one final score per model:
    6A  -> Overall ELO        (subjective, human/VLM preference; ~1300-centered)
    6B  -> objective composite (geometry quality; 0-100)

Method: SAW (Simple Additive Weighting) on z-normalized criteria, mapped back
onto the ELO scale so the FINAL score is centered at 1300 (same language as 6A).
We deliberately do NOT use TOPSIS/Pareto:
  - only 2 criteria -> heavy MCDA adds opacity, not accuracy;
  - Pareto would EXCLUDE dominated models, but a leaderboard must rank ALL of them.

The weights are the one unavoidable value judgment in any MCDA. We handle that
honestly, not by hiding it:
  - PRIMARY = equal weights (0.5 / 0.5): the neutral, defensible default.
  - SENSITIVITY = the same ranking recomputed across a sweep of weights, so you
    can SEE whether the order even depends on the weight (stable => bias is moot).

Mesh Integrity is represented once (inside 6B's composite); 6A contributes only
its holistic Overall ELO, so there's no explicit double-count to merge.
Worst-axis (lowest of 6A's 4 component axes) is reported as a consistency flag
and used only to break near-ties.

Inputs:  Layer 6A .../elos.json , Layer 6B .../objective_scores.json
Output:  fused_scores.json + printed leaderboard & sensitivity table.

Usage:  python3 fuse.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ELOS_PATH = os.path.join(ROOT, "Layer 6A - Subjective Pipeline", "elos.json")
OBJ_PATH = os.path.join(ROOT, "Layer 6B - Objective Pipeline", "objective_scores.json")
OUT_PATH = os.path.join(HERE, "fused_scores.json")

BASELINE = 1300.0
SCALE = 400.0 / math.log(10.0)          # ELO-like spread, so fused score reads like an ELO
W_PRIMARY = 0.5                          # subjective weight; objective = 1 - it
SENS_SUBJECTIVE = [0.7, 0.6, 0.5, 0.4, 0.3]   # weight-sensitivity sweep
COMPONENTS = ["prompt_alignment", "texture_material", "form_proportion", "mesh_integrity"]


def zscores(d: dict) -> dict:
    vals = list(d.values())
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0      # guard all-equal
    return {k: (v - mu) / sd for k, v in d.items()}


def worst_axis(elos: dict, models: list) -> dict:
    """Lowest of a model's 4 component axes (cross-test-case mean) -> weakness flag."""
    vals = defaultdict(lambda: defaultdict(list))
    for _tc, params in elos.get("by_test_case", {}).items():
        for comp in COMPONENTS:
            for m, e in params.get(comp, {}).items():
                vals[m][comp].append(e)
    out = {}
    for m in models:
        means = [statistics.mean(vals[m][c]) for c in COMPONENTS if vals[m].get(c)]
        out[m] = round(min(means), 0) if means else None
    return out


def main() -> None:
    if not (os.path.exists(ELOS_PATH) and os.path.exists(OBJ_PATH)):
        raise SystemExit("Need elos.json (run 6A aggregate.py) and "
                         "objective_scores.json (run 6B score.py) first.")
    elos = json.load(open(ELOS_PATH))
    obj = json.load(open(OBJ_PATH))

    sub = {r["model"]: r["overall_elo"] for r in elos["overall_leaderboard"]}
    objc = {r["model"]: r["obj_score"] for r in obj["objective_leaderboard"]}
    models = sorted(set(sub) & set(objc))          # rank only models present in both
    if not models:
        raise SystemExit("No models common to 6A and 6B.")

    z_sub = zscores({m: sub[m] for m in models})
    z_obj = zscores({m: objc[m] for m in models})
    worst = worst_axis(elos, models)

    def fuse(w_sub: float) -> dict:
        return {m: BASELINE + SCALE * (w_sub * z_sub[m] + (1 - w_sub) * z_obj[m])
                for m in models}

    # ---- primary leaderboard (equal weights), worst-axis breaks near-ties ----
    primary = fuse(W_PRIMARY)
    board = sorted(models, key=lambda m: (round(primary[m], 1),
                                          worst[m] if worst[m] is not None else -1),
                   reverse=True)
    leaderboard = [{
        "rank": i + 1, "model": m,
        "fused_score": round(primary[m], 1),
        "subjective_elo": round(sub[m], 1),
        "objective_score": round(objc[m], 1),
        "worst_axis_elo": worst[m],
    } for i, m in enumerate(board)]

    # ---- sensitivity: rank of each model across the weight sweep ----
    sens = {}
    for w in SENS_SUBJECTIVE:
        ranking = sorted(models, key=lambda m: primary_w(fuse, w, m), reverse=True)
        for pos, m in enumerate(ranking, 1):
            sens.setdefault(m, {})[f"sub{int(w*100)}"] = pos
    stability = {m: {"min_rank": min(sens[m].values()), "max_rank": max(sens[m].values()),
                     "stable": min(sens[m].values()) == max(sens[m].values())}
                 for m in models}

    out = {
        "method": "SAW (z-normalized weighted sum) -> 1300-centered ELO scale",
        "primary_weights": {"subjective": W_PRIMARY, "objective": round(1 - W_PRIMARY, 2)},
        "leaderboard": leaderboard,
        "sensitivity_ranks": sens,
        "sensitivity_stability": stability,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    _print(out)
    print(f"\n[out] fused scores -> {OUT_PATH}")


def primary_w(fuse, w, m):
    return fuse(w)[m]


def _print(out: dict) -> None:
    print(f"\n=== Layer 7 Fused Leaderboard  (weights {out['primary_weights']}, "
          f"centered at 1300) ===")
    print(f"  {'#':>2} {'model':<5} {'FUSED':>7} {'subj':>7} {'obj':>6} {'worst':>6}")
    for r in out["leaderboard"]:
        print(f"  {r['rank']:>2} {r['model']:<5} {r['fused_score']:>7.1f} "
              f"{r['subjective_elo']:>7.1f} {r['objective_score']:>6.1f} "
              f"{str(r['worst_axis_elo']):>6}")
    print("\n=== Weight sensitivity (rank under each subjective-weight scenario) ===")
    cols = [f"sub{int(w*100)}" for w in SENS_SUBJECTIVE]
    print(f"  {'model':<5} " + " ".join(f"{c:>6}" for c in cols) + "   stable?")
    for r in out["leaderboard"]:
        m = r["model"]
        ranks = " ".join(f"{out['sensitivity_ranks'][m][c]:>6}" for c in cols)
        print(f"  {m:<5} {ranks}   {'yes' if out['sensitivity_stability'][m]['stable'] else 'NO'}")


if __name__ == "__main__":
    main()
