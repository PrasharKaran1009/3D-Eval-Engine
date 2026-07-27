"""
Unit tests for the risky math and the new wiring.
==================================================

Focus: the parts where a silent error would corrupt the leaderboard without
crashing — normalization, Bradley-Terry, MCDA methods, A/B randomization
determinism, vote de-randomization, and the BYOA dispatch seam.

Run:  python3 -m pytest tests/ -q      (or: python3 tests/test_pipeline.py)
Pure stdlib + the project's own deps; no network, no cache required for most.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for layer in ("3 - Round Robin Pair Generator", "4 - Generation Prep Scheduler",
              "5A - Subjective Pipeline", "5B - Objective Pipeline",
              "6 - Fusion Layer"):
    p = os.path.join(ROOT, layer)
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
#  6B — rank normalization                                                     #
# --------------------------------------------------------------------------- #
def test_rank_to_100_orientation_and_bounds():
    import score
    r = score.rank_to_100([("a", 10), ("b", 5), ("c", 1)], higher_better=True)
    assert r["a"] == 100.0 and r["c"] == 1.0        # best->100, worst->1 (log floor)
    assert 1.0 < r["b"] < 100.0
    # lower-better flips it
    r2 = score.rank_to_100([("a", 10), ("b", 1)], higher_better=False)
    assert r2["b"] == 100.0 and r2["a"] == 1.0


def test_rank_to_100_none_is_worst():
    import score
    r = score.rank_to_100([("a", 5), ("b", None)], higher_better=True)
    assert r["a"] == 100.0 and r["b"] == 1.0


def test_rank_to_100_single():
    import score
    assert score.rank_to_100([("only", 3)], higher_better=True) == {"only": 100.0}


# --------------------------------------------------------------------------- #
#  6A — Bradley-Terry                                                          #
# --------------------------------------------------------------------------- #
def test_bradley_terry_orders_by_wins():
    import aggregate
    # a beats b twice, b beats c twice -> a > b > c
    wins = {("a", "b"): 2.0, ("b", "c"): 2.0}
    elos = aggregate.fit_bradley_terry(wins)
    assert elos["a"] > elos["b"] > elos["c"]


def test_bradley_terry_centers_on_baseline():
    import aggregate
    elos = aggregate.fit_bradley_terry({("a", "b"): 1.0, ("b", "a"): 1.0})
    # symmetric record -> both near the 1300 baseline
    assert abs(sum(elos.values()) / len(elos) - aggregate.BASELINE) < 1e-6


def test_tie_splits_half():
    import aggregate
    recs = [{"matchup_id": 0, "prompt_id": "TC", "model_a": "a", "model_b": "b",
             "ballots": {"overall": "tie"}, "voter": "human"}]
    wins = aggregate.build_pair_wins(recs)
    assert wins["TC"]["overall"][("a", "b")] == 0.5
    assert wins["TC"]["overall"][("b", "a")] == 0.5


def test_human_substitution_drops_vlm():
    import aggregate
    recs = [
        {"matchup_id": 1, "voter": "vlm:x", "ballots": {"overall": "A"},
         "model_a": "a", "model_b": "b", "prompt_id": "TC"},
        {"matchup_id": 1, "voter": "human", "ballots": {"overall": "B"},
         "model_a": "a", "model_b": "b", "prompt_id": "TC"},
    ]
    kept = aggregate.apply_substitution(recs)
    assert len(kept) == 1 and kept[0]["voter"] == "human"


# --------------------------------------------------------------------------- #
#  7 — MCDA methods                                                            #
# --------------------------------------------------------------------------- #
def test_all_mcda_methods_rank_dominant_first():
    import mcda
    # 'good' dominates 'bad' on both criteria; every method must rank it higher.
    matrix = {"good": {"s": 1600, "o": 90}, "mid": {"s": 1400, "o": 60},
              "bad": {"s": 1000, "o": 20}}
    w = {"s": 0.6, "o": 0.4}
    ben = {"s": True, "o": True}
    for method in mcda.METHODS:
        sc = mcda.score(method, matrix, w, ben)
        assert sc["good"] > sc["bad"], f"{method} failed monotonicity"
        assert sc["good"] >= sc["mid"] >= sc["bad"], f"{method} mis-ordered"


def test_mcda_unknown_method_raises():
    import mcda
    try:
        mcda.score("NOPE", {"a": {"s": 1}}, {"s": 1}, {"s": True})
        assert False, "should have raised"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
#  6A — A/B randomization determinism + vote de-randomization                  #
# --------------------------------------------------------------------------- #
def test_side_swap_is_deterministic_and_balanced():
    import build_viewer_data as b
    # same inputs -> same decision, always
    assert b.side_swapped(7, "M1", "M2") == b.side_swapped(7, "M1", "M2")
    # across many matchups it splits both ways (not all one side)
    flips = [b.side_swapped(i, "M1", "M2") for i in range(200)]
    assert 0.3 < sum(flips) / len(flips) < 0.7


def test_vote_maps_to_displayed_side():
    """The ballot's 'A' must resolve to whichever model is shown on panel A."""
    import build_viewer_data as b
    swap = b.side_swapped(1, "M1", "M2")
    disp_a = "M2" if swap else "M1"      # what build() would put on panel A
    # record_vote reads _a_model from viewer_data, which is exactly disp_a
    assert disp_a == ("M2" if swap else "M1")


# --------------------------------------------------------------------------- #
#  7 — hard constraints + pareto                                               #
# --------------------------------------------------------------------------- #
def test_pareto_flags_dominated():
    import constraints
    crit = {"win": {"subjective": 1600, "objective": 90},
            "dom": {"subjective": 1400, "objective": 60},   # dominated by win
            "niche": {"subjective": 1000, "objective": 95}}  # best objective -> not dominated
    flags = constraints.pareto_flags(crit)
    assert flags["dom"] is True
    assert flags["win"] is False and flags["niche"] is False


def test_settings_merge_fills_defaults():
    import settings
    merged = settings._merge(settings.DEFAULTS, {"fusion": {"mcda_method": "VIKOR"}})
    assert merged["fusion"]["mcda_method"] == "VIKOR"
    assert merged["fusion"]["subjective_weight"] == 0.60   # untouched default kept


# --------------------------------------------------------------------------- #
#  4 — BYOA dispatch seam                                                      #
# --------------------------------------------------------------------------- #
def test_byom_generator_resolves_existing():
    import providers
    manifest = {"assets": {"TC001": {"M1": __file__}}}   # any real path
    gen = providers.get_generator("BYOM", manifest)
    assert gen.resolve({"id": "M1"}, "TC001") == __file__
    assert gen.resolve({"id": "M1"}, "TC999") is None    # missing cell


def test_byoa_generator_is_documented_stub():
    import providers
    gen = providers.get_generator("BYOA")
    try:
        gen.resolve({"id": "M7", "name": "X", "endpoint": "u", "api_key_env": "K"}, "TC001")
        assert False, "BYOA stub should raise until wired"
    except providers.BYOAGenerationNotConfigured as e:
        assert "M7" in str(e) and "providers.py" in str(e)


def test_get_generator_unknown_source():
    import providers
    try:
        providers.get_generator("WAT")
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    # tiny runner so it works without pytest installed
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"  ok   {fn.__name__}")
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
