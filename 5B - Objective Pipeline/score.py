"""
Layer 6B - Objective Scoring
============================

Turns Layer 5's raw geometry metrics into 0-100 per-dimension scores and one
objective composite per model. The objective counterpart of 6A's ELO step.

It is NOT Bradley-Terry (there are no pairwise matches) -- it is deterministic
rubric scoring over ABSOLUTE measurements:

    1. scale-free ratios   (counts / face_count, so big meshes aren't punished)
    2. rank-normalize each metric to [1,100] across the models in a test case
       (rank-based = robust to the wild outliers in generative meshes;
        floor of 1 so the geometric mean below never collapses to 0)
    3. average metrics into 4 dimensions
    4. NON-COMPENSATORY composite via weighted GEOMETRIC mean -- a near-zero
       dimension (e.g. a shattered mesh) drags the whole score down and can't be
       bought back by a great texture. (Integrity also carries the top weight.)

Dimensions:
    Integrity   - boundary/non-manifold/fragmentation ratios + watertight bonus
    Texturing   - has-texture gate, resolution, texel density, UV uniformity
    Symmetry    - reflective symmetry score
    Efficiency  - leanness (fewer faces = leaner); simple proxy, low weight

Output: objective_scores.json + printed summary. The composite is what Layer 7
fuses with 6A's Overall ELO (and Mesh Integrity merges across the two cores).

Reads Layer 5 cache directly; no extraction here. Pure stdlib + numpy/scipy.

Usage:
    python3 score.py
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "Cache Storage Layer", "cache")
OUT_PATH = os.path.join(HERE, "objective_scores.json")

# Composite weights (documented + adjustable). Integrity dominates.
# Efficiency is REPORT-ONLY (weight 0): "leanness" unfairly penalizes detailed
# high-poly models (e.g. M6 had the best texture+symmetry but tanked on poly
# count), so it's still computed/shown as a diagnostic but doesn't move the rank.
WEIGHTS = {"integrity": 0.50, "texturing": 0.35, "symmetry": 0.15, "efficiency": 0.0}


# ---------------------------------------------------------------- load
def load_cache() -> dict:
    """-> {test_case: {model_id: metrics_dict}}"""
    data: dict = defaultdict(dict)
    for mpath in glob.glob(os.path.join(CACHE, "*", "*", "metrics.json")):
        model_id = os.path.basename(os.path.dirname(mpath))
        tc = os.path.basename(os.path.dirname(os.path.dirname(mpath)))
        with open(mpath) as f:
            data[tc][model_id] = json.load(f)
    return data


# ---------------------------------------------------------------- rank -> [1,100]
def rank_to_100(pairs: list, higher_better: bool) -> dict:
    """
    pairs: [(model_id, value_or_None)]. None = worst. Ties get the average rank.
    Best model -> 100, worst -> 1 (floor avoids log(0) in the geometric mean).
    """
    ids = [m for m, _ in pairs]
    n = len(ids)
    if n == 1:
        return {ids[0]: 100.0}
    worst = -math.inf if higher_better else math.inf      # push None to the bottom
    vals = np.array([(worst if v is None else v) for _, v in pairs], dtype=float)
    oriented = vals if higher_better else -vals           # higher oriented = better
    ranks = rankdata(oriented, method="average")          # 1..n, higher = better
    return {ids[i]: round(1 + 99 * (ranks[i] - 1) / (n - 1), 1) for i in range(n)}


# ---------------------------------------------------------------- derived metrics
def derived(m: dict) -> dict:
    fc = m.get("face_count") or 1
    tex_px = m.get("texture_pixels")
    area = m.get("surface_area") or 1.0
    return {
        "boundary_ratio": (m.get("boundary_edges") or 0) / fc,
        "nonmanifold_ratio": ((m.get("non_manifold_edges") or 0)
                              + (m.get("non_manifold_vertices") or 0)) / fc,
        "frag": (m.get("connected_components") or 0) / fc,
        "texel_density": (tex_px / area) if tex_px else None,
    }


# ---------------------------------------------------------------- score one test case
def score_test_case(models: dict) -> dict:
    ids = sorted(models)
    d = {i: derived(models[i]) for i in ids}

    def sm(getter, higher_better):
        return rank_to_100([(i, getter(i)) for i in ids], higher_better)

    # Integrity sub-metrics (all "lower raw = better" except the watertight bonus)
    boundary = sm(lambda i: d[i]["boundary_ratio"], False)
    nonman = sm(lambda i: d[i]["nonmanifold_ratio"], False)
    frag = sm(lambda i: d[i]["frag"], False)
    watert = {i: (100.0 if models[i].get("is_watertight") else 1.0) for i in ids}

    # Texturing sub-metrics
    hastex = {i: (100.0 if models[i].get("has_texture") else 1.0) for i in ids}
    texpx = sm(lambda i: models[i].get("texture_pixels"), True)
    texden = sm(lambda i: d[i]["texel_density"], True)
    uvunif = sm(lambda i: models[i].get("uv_stretch_cv"), False)   # lower CV = better

    symm = sm(lambda i: models[i].get("symmetry_score"), True)
    effic = sm(lambda i: models[i].get("face_count"), False)       # leaner = better

    out = {}
    for i in ids:
        dims = {
            "integrity": float(np.mean([boundary[i], nonman[i], frag[i], watert[i]])),
            "texturing": float(np.mean([hastex[i], texpx[i], texden[i], uvunif[i]])),
            "symmetry": symm[i],
            "efficiency": effic[i],
        }
        # non-compensatory: weighted geometric mean
        obj = math.exp(sum(WEIGHTS[k] * math.log(max(dims[k], 1e-9)) for k in WEIGHTS))
        out[i] = {
            "name": models[i].get("model_name")
                    or _name_from_metadata(models[i]) or i,
            "dimensions": {k: round(v, 1) for k, v in dims.items()},
            "obj_score": round(obj, 1),
        }
    return out


def _name_from_metadata(metrics: dict):
    return None   # names live in metadata.json; metrics.json has none -> fall back to id


# ---------------------------------------------------------------- orchestrate
def compute() -> dict:
    cache = load_cache()
    by_tc, acc = {}, defaultdict(list)
    for tc in sorted(cache):
        scored = score_test_case(cache[tc])
        by_tc[tc] = scored
        for mid, row in scored.items():
            acc[mid].append(row["obj_score"])

    board = [{"model": m, "obj_score": round(sum(v) / len(v), 1), "test_cases": len(v)}
             for m, v in acc.items()]
    board.sort(key=lambda r: r["obj_score"], reverse=True)
    return {"weights": WEIGHTS, "by_test_case": by_tc, "objective_leaderboard": board}


def write(out: dict, path: str = OUT_PATH) -> dict:
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def run() -> dict:
    """
    Compute + persist objective_scores.json. Importable entry point.

    main.py's run_pipeline() calls this right after the Layer 5 cache is filled,
    because the Fusion layer HARD-REQUIRES objective_scores.json and a registry
    fork deletes it — leaving this to a manual CLI step meant a fresh model set
    could never build a leaderboard.
    """
    return write(compute())


def _print(out: dict) -> None:
    print("\n=== 6B Objective Leaderboard (mean composite across test cases) ===")
    if not out["objective_leaderboard"]:
        # Empty cache (fresh clone / post-fork). Report it instead of raising a
        # StopIteration on the dimension-breakdown sample below.
        print("  (no cached meshes yet — run `python3 main.py` to fill the Layer 5 cache)")
        return
    for rank, r in enumerate(out["objective_leaderboard"], 1):
        print(f"  {rank:>2}. {r['model']:<5} {r['obj_score']:>6.1f}  ({r['test_cases']} TCs)")
    print(f"\n  weights: {out['weights']}")
    # show one test case's dimension breakdown as a sanity check
    tc = next(iter(out["by_test_case"]))
    print(f"\n=== {tc} dimension breakdown ===")
    print(f"  {'model':<5} {'integ':>6} {'textr':>6} {'symm':>6} {'effic':>6} {'OBJ':>6}")
    rows = sorted(out["by_test_case"][tc].items(),
                  key=lambda kv: kv[1]["obj_score"], reverse=True)
    for mid, row in rows:
        dd = row["dimensions"]
        print(f"  {mid:<5} {dd['integrity']:>6} {dd['texturing']:>6} "
              f"{dd['symmetry']:>6} {dd['efficiency']:>6} {row['obj_score']:>6}")


def main() -> None:
    out = run()
    _print(out)
    print(f"\n[out] objective scores -> {OUT_PATH}")


if __name__ == "__main__":
    main()
