"""
Layer 7 - MCDA method library  (updated.png · Fusion Step 3)
============================================================

The four algorithms the diagram names, plus SAW, behind one interface:

    method(matrix, weights, benefit) -> {alt: preference_score}   (higher = better)

    matrix   : {alt: {criterion: raw_value}}
    weights  : {criterion: weight}       (need not sum to 1; normalized here)
    benefit  : {criterion: True/False}   (True = higher is better)

Every method returns a score where HIGHER IS BETTER, so fuse.py can treat them
interchangeably and map any of them onto the shared 1300-centered ELO scale.
VIKOR's native Q is lower-better, so we return (1 - Q) to keep the convention.

Pure stdlib + math; no numpy (matrices are ~7x2). Each method is deterministic.

Why several methods at all, for only two criteria? Mostly so the choice is the
user's and is transparent: SAW and TOPSIS will usually agree here, but ELECTRE's
veto and VIKOR's compromise strategy can reorder near-ties differently, and
exposing that is more honest than hard-coding one aggregator.
"""

from __future__ import annotations

import math

METHODS = ("SAW", "TOPSIS", "PROMETHEE", "VIKOR", "ELECTRE")


# --------------------------------------------------------------------------- #
#  shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _norm_weights(weights: dict) -> dict:
    tot = sum(weights.values()) or 1.0
    return {k: v / tot for k, v in weights.items()}


def _criteria(matrix: dict) -> list:
    return sorted({c for row in matrix.values() for c in row})


def _oriented(matrix: dict, benefit: dict) -> dict:
    """Flip cost criteria so every criterion is higher-better, in-place copy."""
    return {a: {c: (v if benefit.get(c, True) else -v) for c, v in row.items()}
            for a, row in matrix.items()}


def _rescale(scores: dict) -> dict:
    """Min-max the final scores into [0,1] for a comparable preference value."""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {a: 0.5 for a in scores}
    return {a: (s - lo) / (hi - lo) for a, s in scores.items()}


# --------------------------------------------------------------------------- #
#  SAW — Simple Additive Weighting (linear, on min-max normalized criteria)    #
# --------------------------------------------------------------------------- #
def saw(matrix: dict, weights: dict, benefit: dict) -> dict:
    w = _norm_weights(weights)
    crits = _criteria(matrix)
    col = {c: [matrix[a].get(c, 0.0) for a in matrix] for c in crits}
    out = {}
    for a in matrix:
        s = 0.0
        for c in crits:
            lo, hi = min(col[c]), max(col[c])
            v = matrix[a].get(c, 0.0)
            n = (v - lo) / (hi - lo) if hi - lo > 1e-12 else 1.0
            if not benefit.get(c, True):
                n = 1.0 - n
            s += w.get(c, 0.0) * n
        out[a] = s
    return _rescale(out)


# --------------------------------------------------------------------------- #
#  TOPSIS — closeness to the ideal, vector-normalized                          #
# --------------------------------------------------------------------------- #
def topsis(matrix: dict, weights: dict, benefit: dict) -> dict:
    w = _norm_weights(weights)
    crits = _criteria(matrix)
    alts = list(matrix)
    denom = {c: math.sqrt(sum(matrix[a].get(c, 0.0) ** 2 for a in alts)) or 1.0
             for c in crits}
    v = {a: {c: w.get(c, 0.0) * (matrix[a].get(c, 0.0) / denom[c]) for c in crits}
         for a in alts}
    ideal, anti = {}, {}
    for c in crits:
        vals = [v[a][c] for a in alts]
        ideal[c] = max(vals) if benefit.get(c, True) else min(vals)
        anti[c] = min(vals) if benefit.get(c, True) else max(vals)
    out = {}
    for a in alts:
        dp = math.sqrt(sum((v[a][c] - ideal[c]) ** 2 for c in crits))
        dn = math.sqrt(sum((v[a][c] - anti[c]) ** 2 for c in crits))
        out[a] = dn / (dp + dn) if (dp + dn) > 1e-12 else 0.0    # closeness, higher better
    return out


# --------------------------------------------------------------------------- #
#  PROMETHEE II — net outranking flow (linear preference function)             #
# --------------------------------------------------------------------------- #
def promethee(matrix: dict, weights: dict, benefit: dict) -> dict:
    w = _norm_weights(weights)
    crits = _criteria(matrix)
    alts = list(matrix)
    m = _oriented(matrix, benefit)
    rng = {c: (max(m[a][c] for a in alts) - min(m[a][c] for a in alts)) or 1.0
           for c in crits}

    def pref(a, b):   # aggregated preference degree of a over b
        return sum(w.get(c, 0.0) * max(0.0, (m[a][c] - m[b][c]) / rng[c]) for c in crits)

    n = len(alts)
    out = {}
    for a in alts:
        pos = sum(pref(a, b) for b in alts if b != a) / (n - 1 or 1)
        neg = sum(pref(b, a) for b in alts if b != a) / (n - 1 or 1)
        out[a] = pos - neg                                       # net flow, higher better
    return _rescale(out)


# --------------------------------------------------------------------------- #
#  VIKOR — compromise ranking (returns 1 - Q so higher = better)               #
# --------------------------------------------------------------------------- #
def vikor(matrix: dict, weights: dict, benefit: dict, v: float = 0.5) -> dict:
    w = _norm_weights(weights)
    crits = _criteria(matrix)
    alts = list(matrix)
    best, worst = {}, {}
    for c in crits:
        vals = [matrix[a].get(c, 0.0) for a in alts]
        best[c] = max(vals) if benefit.get(c, True) else min(vals)
        worst[c] = min(vals) if benefit.get(c, True) else max(vals)

    S, R = {}, {}
    for a in alts:
        terms = []
        for c in crits:
            span = (best[c] - worst[c]) or 1.0
            d = w.get(c, 0.0) * (best[c] - matrix[a].get(c, 0.0)) / span
            terms.append(d)
        S[a] = sum(terms)
        R[a] = max(terms)

    s_lo, s_hi = min(S.values()), max(S.values())
    r_lo, r_hi = min(R.values()), max(R.values())
    out = {}
    for a in alts:
        qs = (S[a] - s_lo) / ((s_hi - s_lo) or 1.0)
        qr = (R[a] - r_lo) / ((r_hi - r_lo) or 1.0)
        Q = v * qs + (1 - v) * qr        # lower Q = better compromise
        out[a] = 1.0 - Q                 # flip so higher = better
    return out


# --------------------------------------------------------------------------- #
#  ELECTRE — net concordance minus discordance, with a veto threshold          #
# --------------------------------------------------------------------------- #
def electre(matrix: dict, weights: dict, benefit: dict, veto: float = 0.35) -> dict:
    w = _norm_weights(weights)
    crits = _criteria(matrix)
    alts = list(matrix)
    m = _oriented(matrix, benefit)
    rng = {c: (max(m[a][c] for a in alts) - min(m[a][c] for a in alts)) or 1.0
           for c in crits}

    def concordance(a, b):
        return sum(w.get(c, 0.0) for c in crits if m[a][c] >= m[b][c])

    def vetoed(a, b):     # b outscores a on some criterion by more than `veto`
        return any((m[b][c] - m[a][c]) / rng[c] > veto for c in crits)

    out = {a: 0.0 for a in alts}
    for a in alts:
        for b in alts:
            if a == b:
                continue
            if concordance(a, b) >= 0.5 and not vetoed(a, b):
                out[a] += 1.0            # a outranks b
            if concordance(b, a) >= 0.5 and not vetoed(b, a):
                out[a] -= 1.0            # a is outranked by b
    return _rescale(out)


# --------------------------------------------------------------------------- #
#  dispatch                                                                    #
# --------------------------------------------------------------------------- #
def score(method: str, matrix: dict, weights: dict, benefit: dict,
          *, vikor_v: float = 0.5, electre_veto: float = 0.35) -> dict:
    method = (method or "SAW").upper()
    if method == "SAW":
        return saw(matrix, weights, benefit)
    if method == "TOPSIS":
        return topsis(matrix, weights, benefit)
    if method == "PROMETHEE":
        return promethee(matrix, weights, benefit)
    if method == "VIKOR":
        return vikor(matrix, weights, benefit, v=vikor_v)
    if method == "ELECTRE":
        return electre(matrix, weights, benefit, veto=electre_veto)
    raise ValueError(f"unknown MCDA method {method!r}; choose from {METHODS}")
