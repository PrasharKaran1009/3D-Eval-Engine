"""
6 - Fusion Layer
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
  - PRIMARY = the configured split (default subjective 0.60 / objective 0.40,
    from Layer 7 settings.py / settings.json; the diagram's default).
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

import mcda            # Layer 7 MCDA method library (Fusion Step 3)
import constraints     # Layer 7 hard constraints + Pareto (Fusion Steps 1-2)
import settings        # Layer 7 global settings (weights, method, thresholds)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ELOS_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "elos.json")
OBJ_PATH = os.path.join(ROOT, "5B - Objective Pipeline", "objective_scores.json")
OUT_PATH = os.path.join(HERE, "fused_scores.json")

BASELINE = 1300.0
SCALE = 400.0 / math.log(10.0)          # ELO-like spread, so fused score reads like an ELO

# Weights + method + thresholds now come from Layer 7 settings.py (settings.json),
# because the diagram shows them as user-configurable. The module-level defaults
# below are read at call time from settings.load() so a wizard edit takes effect
# on the next fuse without a code change.
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


def subjective_blend(elos: dict, models: list, param_weights: dict) -> dict:
    """Weighted subjective ELO per model: param_weights-weighted mean of each
    parameter's cross-test-case mean ELO (normalized over the params present)."""
    vals = defaultdict(lambda: defaultdict(list))
    for _tc, params in elos.get("by_test_case", {}).items():
        for param, per_model in params.items():
            for m, e in per_model.items():
                vals[m][param].append(e)
    out = {}
    for m in models:
        num = den = 0.0
        for p, w in param_weights.items():
            if vals[m].get(p):
                num += w * statistics.mean(vals[m][p])
                den += w
        out[m] = round(num / den, 1) if den else None
    return out


def _to_elo(scores: dict) -> dict:
    """Map any method's preference scores onto the shared 1300-centered ELO
    scale by z-normalizing them and applying the ELO spread. This is what lets
    every MCDA method report on the same axis the leaderboard already speaks."""
    z = zscores(scores)
    return {m: BASELINE + SCALE * z[m] for m in scores}


def main() -> None:
    if not (os.path.exists(ELOS_PATH) and os.path.exists(OBJ_PATH)):
        raise SystemExit("Need elos.json (run 6A aggregate.py) and "
                         "objective_scores.json (run 6B score.py) first.")
    elos = json.load(open(ELOS_PATH))
    obj = json.load(open(OBJ_PATH))

    cfg = settings.load()
    fcfg = cfg["fusion"]
    method = fcfg["mcda_method"]
    w_sub = fcfg["subjective_weight"]
    param_weights = cfg["parameter_weights"]

    sub_models = [r["model"] for r in elos["overall_leaderboard"]]
    sub = subjective_blend(elos, sub_models, param_weights)
    objc = {r["model"]: r["obj_score"] for r in obj["objective_leaderboard"]}
    models = sorted(m for m in (set(sub) & set(objc)) if sub[m] is not None)
    if not models:
        raise SystemExit("No models common to 6A and 6B.")

    worst = worst_axis(elos, models)

    # ---- Step 1: hard constraints (reject / penalty) ---------------------- #
    verdict = constraints.evaluate(models, cfg)
    obj_eff = {m: objc[m] * verdict[m]["penalty"] for m in models}   # penalized objective

    # ---- Step 3: MCDA aggregation on the 2 criteria ----------------------- #
    weights = {"subjective": w_sub, "objective": round(1 - w_sub, 4)}
    benefit = {"subjective": True, "objective": True}

    def run_method(wsub: float) -> dict:
        matrix = {m: {"subjective": sub[m], "objective": obj_eff[m]} for m in models}
        pref = mcda.score(method, matrix,
                          {"subjective": wsub, "objective": 1 - wsub}, benefit,
                          vikor_v=fcfg["vikor_v"], electre_veto=fcfg["electre_veto"])
        return _to_elo(pref)

    fused = run_method(w_sub)

    # ---- Step 2: Pareto flags (optional; flag, never drop) ---------------- #
    crit = {m: {"subjective": sub[m], "objective": obj_eff[m]} for m in models}
    pareto = constraints.pareto_flags(crit) if fcfg["pareto_filter"] else {m: False for m in models}

    # ---- rank: feasible first; rejected forced last; worst-axis breaks ties  #
    def sort_key(m):
        return (0 if verdict[m]["reject"] else 1,       # rejected -> bottom
                round(fused[m], 1),
                worst[m] if worst[m] is not None else -1)
    board = sorted(models, key=sort_key, reverse=True)

    leaderboard = [{
        "rank": i + 1, "model": m,
        "fused_score": round(fused[m], 1),
        "subjective_elo": round(sub[m], 1),
        "objective_score": round(objc[m], 1),
        "objective_penalized": round(obj_eff[m], 1),
        "worst_axis_elo": worst[m],
        "rejected": verdict[m]["reject"],
        "penalty": verdict[m]["penalty"],
        "constraint_reasons": verdict[m]["reasons"],
        "pareto_dominated": pareto[m],
    } for i, m in enumerate(board)]

    # ---- sensitivity: rank of each model across the weight sweep ----------- #
    sens = {}
    for w in SENS_SUBJECTIVE:
        scored = run_method(w)
        ranking = sorted(models, key=lambda m: (0 if verdict[m]["reject"] else 1, scored[m]),
                         reverse=True)
        for pos, m in enumerate(ranking, 1):
            sens.setdefault(m, {})[f"sub{int(w*100)}"] = pos
    stability = {m: {"min_rank": min(sens[m].values()), "max_rank": max(sens[m].values()),
                     "stable": min(sens[m].values()) == max(sens[m].values())}
                 for m in models}

    out = {
        "method": f"{method} on z-normalized criteria -> 1300-centered ELO scale",
        "mcda_method": method,
        "primary_weights": {"subjective": weights["subjective"],
                            "objective": weights["objective"]},
        "parameter_weights": param_weights,
        "hard_constraints": cfg["hard_constraints"],
        "pareto_filter": fcfg["pareto_filter"],
        "leaderboard": leaderboard,
        "sensitivity_ranks": sens,
        "sensitivity_stability": stability,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    _print(out)
    print(f"\n[out] fused scores -> {OUT_PATH}")


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
