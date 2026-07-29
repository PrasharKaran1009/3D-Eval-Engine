"""
main.py — pipeline driver
=========================

Single entry point for the prototype. Currently wires:

    Layer 3  Pair Generation      -> all matchups {model_a, model_b, prompt_id}
    Layer 4  Generation Scheduler  -> resolve GLB availability  (STREAMED)
    Layer 5  Output Cache          -> extract 6B metrics, cache (STREAMED)
    Layer 6A Subjective (preview)   -> Three.js viewer w/ shaded/textured/wireframe

Streaming: each matchup flows all the way through 4 -> 5 before the next one
starts, so a single Matchup object is the unit moving through the pipeline.

As later layers (6A voting, 6B scoring, 7 fusion, ...) come online, add steps to
run_pipeline() and the CLI below.

Usage:
    python main.py                 # run pipeline (cached) + launch preview viewer
    python main.py --no-viewer     # pipeline only
    python main.py --viewer-only   # skip pipeline, just launch viewer (uses cache)
    python main.py --force         # recompute metrics, ignore cache
    python main.py --limit 5       # only first 5 matchups (smoke test)
    python main.py --port 8000     # viewer server port
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import shutil
import sys
import webbrowser
import zipfile
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
for _layer in ("2 - Input Layer",
               "3 - Round Robin Pair Generator",
               "4 - Generation Prep Scheduler",
               "Cache Storage Layer",
               "5A - Subjective Pipeline",
               "5B - Objective Pipeline",
               "6 - Fusion Layer"):
    _p = os.path.join(ROOT, _layer)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import byom_validator             # noqa: E402  Layer 2
import generate_matchups          # noqa: E402  Layer 3
import scheduler                  # noqa: E402  Layer 4
import cache                      # noqa: E402  Layer 5
import build_viewer_data          # noqa: E402  Layer 6A
import score as objective_score   # noqa: E402  Layer 5B
import settings                    # noqa: E402  Layer 7 (global settings)

ENRICHED_PATH = os.path.join(ROOT, "enriched_matchups.json")

# --------------------------------------------------------------------------- #
#  BALLOT STORAGE — committed demo vs. live session.                          #
#                                                                              #
#  VOTES_PATH is the COMMITTED 105-ballot demo log. At runtime it is READ-ONLY: #
#  nothing the server does may rewrite it, or a single visitor clicking through #
#  the wizard would destroy the shipped demo (and dirty the working tree).      #
#  Every live ballot goes to SESSION_VOTES_PATH instead, which is gitignored    #
#  and seeded from the frozen example on first use — so the out-of-the-box      #
#  experience is identical, but a visitor can only ever mutate their own copy.  #
# --------------------------------------------------------------------------- #
VOTES_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "votes.json")
FINAL_VOTES_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "final_votes.json")
RUNS_DIR = os.path.join(ROOT, "runs")
SESSION_DIR = os.path.join(RUNS_DIR, "session")
SESSION_VOTES_PATH = os.path.join(SESSION_DIR, "votes.json")

# Frozen demo catalog (committed, read-only). Model ids here are permanent.
CATALOG_PATH = os.path.join(ROOT, "benchmark", "catalog.json")

# Entry point: the Setup Studio wizard (starts at the Welcome page — pick BYOM
# and the judge). The blind pair viewer is reachable from its Evaluate step.
ENTRY_REL = "frontend/index.html"
VIEWER_REL = "5A - Subjective Pipeline/preview.html"


VIEWER_DATA_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "viewer_data.json")


def _display_sides(matchup_id) -> dict:
    """
    Resolve which real model is on the LEFT (A) and RIGHT (B) panel for a matchup.

    viewer_data.json is the authority, because Layer 6A deterministically
    randomizes which model gets shown as A (see build_viewer_data.side_swapped).
    Reading the pair order from enriched_matchups instead would map every ballot
    back to the *unswapped* order and silently invert half the votes.
    """
    with open(VIEWER_DATA_PATH) as f:
        entries = json.load(f).get("matchups", [])
    for m in entries:
        if m["matchup_id"] == matchup_id:
            return {"matchup_id": m["matchup_id"], "prompt_id": m["prompt_id"],
                    "a_model": m["_a_model"], "b_model": m["_b_model"]}
    raise KeyError(f"unknown matchup_id {matchup_id}")


def ensure_session_votes() -> str:
    """
    Return the path of the live (session) ballot log, creating it if needed.

    Seeded from runs/example/votes.json — falling back to the committed demo
    log — so a fresh clone starts out showing exactly the 105-ballot example
    that the committed elos/fused scores were computed from. After that first
    copy the committed file is never touched again: reads and writes both go
    through the session file, which is gitignored.
    """
    os.makedirs(SESSION_DIR, exist_ok=True)
    if not os.path.exists(SESSION_VOTES_PATH):
        seed = []
        for src in (os.path.join(EXAMPLE_DIR, "votes.json"), VOTES_PATH):
            if os.path.exists(src):
                try:
                    seed = _load_json(src)
                    break
                except (ValueError, OSError):
                    continue
        with open(SESSION_VOTES_PATH, "w") as f:
            json.dump(seed, f, indent=2)
    return SESSION_VOTES_PATH


def record_vote(payload: dict) -> dict:
    """
    Record one vote in the SESSION ballot log. The browser sends only matchup_id + per-
    parameter winners ('A' | 'B' | 'tie'); we resolve the real (blind) model ids
    here, server-side, so the judge never saw them but Bradley-Terry can later.

    UPSERT, not append: a vote is keyed by (matchup_id, voter). Re-voting a
    matchup REPLACES that voter's earlier ballot instead of stacking a second
    one, which would silently double-weight the pair in the Bradley-Terry fit.
    Different voters still each get a ballot (needed for inter-rater agreement),
    and 'human' vs 'vlm:<model>' remain distinct keys (needed for the hybrid
    flow, where a human review must sit alongside the VLM first pass).
    """
    m = _display_sides(payload["matchup_id"])

    record = {
        "matchup_id": m["matchup_id"],
        "prompt_id": m["prompt_id"],
        # A/B here are the sides AS DISPLAYED, de-randomized back to real ids.
        "model_a": m["a_model"],
        "model_b": m["b_model"],
        "ballots": payload["ballots"],    # {param: 'A'|'B'|'tie'}
        "voter": payload.get("voter", "anon"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    votes_path = ensure_session_votes()
    votes = []
    if os.path.exists(votes_path):
        with open(votes_path) as f:
            votes = json.load(f)

    key = (record["matchup_id"], record["voter"])
    for i, v in enumerate(votes):
        if (v.get("matchup_id"), v.get("voter")) == key:
            votes[i] = record
            replaced = True
            break
    else:
        votes.append(record)
        replaced = False

    with open(votes_path, "w") as f:
        json.dump(votes, f, indent=2)
    return {"ok": True, "total_votes": len(votes), "replaced": replaced}


# --------------------------------------------------------------------------- #
#  Read-through API for the frontend "Setup Studio" — serves the REAL registry, #
#  validation, pairs and results so the wizard runs live instead of on mocks.   #
# --------------------------------------------------------------------------- #
L3_DIR = os.path.join(ROOT, "1 - Registry Layer")
FUSED_PATH = os.path.join(ROOT, "6 - Fusion Layer", "fused_scores.json")
ELOS_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "elos.json")
OBJ_PATH = os.path.join(ROOT, "5B - Objective Pipeline", "objective_scores.json")
GENLOG_PATH = os.path.join(ROOT, "benchmark_assets", "generation_log.json")


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _gen_seconds(text) -> float | None:
    """'4 minutes 6 seconds' -> 246.0 ; tolerant of singular/missing parts."""
    if not text:
        return None
    import re
    m = re.search(r"(\d+)\s*min", text)
    s = re.search(r"(\d+)\s*sec", text)
    total = (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)
    return float(total) or None


# --------------------------------------------------------------------------- #
#  DEMO CATALOG — benchmark/catalog.json                                       #
#                                                                              #
#  The frozen, committed roster a visitor picks from in demo mode. Its ids are  #
#  PERMANENT: the 105 example ballots, the Layer 5 mesh cache and every         #
#  computed score are keyed by "M3", so demo mode writes catalog entries        #
#  VERBATIM (id included) instead of re-deriving ids from the selection order.  #
#  That is what lets a visitor change which demo models they compare without    #
#  invalidating the cache or re-attaching history to the wrong model.           #
# --------------------------------------------------------------------------- #
def load_catalog() -> dict:
    """Read the demo catalog. Read-through (no caching) so an edit lands live."""
    if not os.path.exists(CATALOG_PATH):
        raise ValueError(f"demo catalog missing: {os.path.relpath(CATALOG_PATH, ROOT)}")
    cat = _load_json(CATALOG_PATH)
    return {"catalog_version": cat.get("catalog_version", 1),
            "models": cat.get("models", []),
            "test_cases": cat.get("test_cases", [])}


def _testcase_zip(tc: str) -> str | None:
    """Locate the archive for a test case, if any.

    Both spellings are accepted: `TC001.zip` (the contract's name) and
    `TC001 GLB.zip` (what currently ships), so renaming the packs on disk
    can't silently break the bootstrap.
    """
    for cand in (f"{tc}.zip", f"{tc} GLB.zip"):
        p = os.path.join(TESTCASES_DIR, cand)
        if os.path.exists(p):
            return p
    return None


def _testcase_dir(tc: str) -> str:
    return os.path.join(TESTCASES_DIR, f"{tc} GLB")


def _testcase_available(tc: str) -> bool:
    """A test case is usable if its meshes are extracted OR still zipped."""
    folder = _testcase_dir(tc)
    if os.path.isdir(folder) and any(
            e.name.lower().endswith((".glb", ".gltf", ".obj", ".fbx"))
            for e in os.scandir(folder) if e.is_file()):
        return True
    return _testcase_zip(tc) is not None


def api_catalog() -> dict:
    """The demo roster + which test cases this checkout can actually run."""
    cat = load_catalog()
    return {"models": cat["models"], "test_cases": cat["test_cases"],
            "available": [tc["id"] for tc in cat["test_cases"]
                          if _testcase_available(tc["id"])]}


def ensure_testcases(tc_ids) -> dict:
    """
    Lazily unpack `TEST CASES/<TC>.zip` -> `TEST CASES/<TC> GLB/`.

    The mesh packs ship zipped (they are large and gitignored when loose), so a
    fresh checkout has archives but no folders and Layer 4 would find nothing to
    index. This extracts ONLY the selected test cases — unpacking all five costs
    minutes and gigabytes a 2-model smoke run never needs — and is idempotent:
    an already-extracted folder is left alone.

    stdlib zipfile only. Members are path-checked (no absolute paths, no `..`
    escapes) and macOS resource forks (`__MACOSX/`, `._*`, `.DS_Store`) are
    dropped so the extracted folder holds exactly the meshes Layer 2 validates.
    """
    extracted, skipped, missing = [], [], []
    for raw in (tc_ids or []):
        tc = str(raw).strip().upper()
        if not re.fullmatch(r"TC\d{3}", tc):
            continue
        folder = _testcase_dir(tc)
        if os.path.isdir(folder):
            skipped.append(tc)
            continue
        zpath = _testcase_zip(tc)
        if zpath is None:
            missing.append(tc)
            continue
        prefix = f"{tc} GLB/"
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                if info.is_dir() or name.startswith("__MACOSX/"):
                    continue
                # the archives wrap everything in "<TC> GLB/" — strip it so the
                # meshes land in `folder`, not `folder/<TC> GLB/`
                rel = name[len(prefix):] if name.startswith(prefix) else name
                base = os.path.basename(rel)
                if not base or base.startswith("._") or base == ".DS_Store":
                    continue
                if os.path.isabs(rel) or ".." in rel.split("/"):
                    continue                      # zip-slip guard
                dest = os.path.join(folder, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        extracted.append(tc)
        print(f"[pack] extracted {os.path.basename(zpath)} -> "
              f"{os.path.relpath(folder, ROOT)}/")
    return {"extracted": extracted, "present": skipped, "missing": missing}


def ensure_registry() -> bool:
    """
    Seed the WORKING registry from the frozen catalog when it is absent.

    `1 - Registry Layer/models.json` + `prompts.json` are *generated working
    state*, not source: every registry save rewrites them, so a demo visitor who
    compares 2 of the 7 models legitimately shrinks them to 2 entries. The
    canonical 7x5 benchmark lives in `benchmark/catalog.json`, which nothing at
    runtime ever writes.

    Without this, a fresh clone (where the working files are gitignored) would
    start with no roster at all, and a visitor's narrowing selection would look
    like permanent data loss. Seeding restores the full catalog whenever either
    file is missing — so "delete them and restart" is the reset button.

    Returns True if anything was written. Never overwrites an existing file: a
    BYOM user's own roster must survive a restart.
    """
    if os.path.exists(MODELS_PATH) and os.path.exists(PROMPTS_PATH):
        return False
    try:
        cat = load_catalog()
    except (ValueError, OSError) as e:
        print(f"[L1] cannot seed registry from catalog: {e}")
        return False

    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    if not os.path.exists(MODELS_PATH):
        models = [{
            "id": m["id"], "name": m["name"], "source": "BYOM",
            "format": (m.get("format") or "glb").lower(),
            "match": copy.deepcopy(m.get("match") or {}),
            "metadata": {k: m[k] for k in ("vendor", "credits")
                         if m.get(k) not in (None, "")},
        } for m in cat["models"]]
        with open(MODELS_PATH, "w") as f:
            json.dump({"registry_version": 1, "n": len(models),
                       "models": models}, f, indent=2)
        print(f"[L1] seeded models.json from catalog ({len(models)} models)")
    if not os.path.exists(PROMPTS_PATH):
        prompts = [{"id": t["id"], "subject": t.get("subject", ""),
                    "description": t.get("description", ""),
                    "reference_images": list(t.get("reference_images") or [])}
                   for t in cat["test_cases"]]
        with open(PROMPTS_PATH, "w") as f:
            json.dump({"benchmark_version": 1, "prompts": prompts}, f, indent=2)
        print(f"[L1] seeded prompts.json from catalog ({len(prompts)} test cases)")
    return True


def reset_benchmark() -> dict:
    """Restore the full 7x5 catalog roster, discarding the current selection.

    The wizard's "use the whole benchmark again" escape hatch. Deliberately does
    NOT touch votes, scores or the mesh cache — catalog ids are fixed, so a
    widened selection still lines up with everything already computed.
    """
    for p in (MODELS_PATH, PROMPTS_PATH):
        if os.path.exists(p):
            os.remove(p)
    ensure_registry()
    models = _load_json(MODELS_PATH)["models"]
    _write_fingerprint(_registry_fingerprint(models), "demo")
    return {"ok": True, "mode": "demo", "models": len(models),
            "test_cases": len(_load_json(PROMPTS_PATH)["prompts"])}


def api_registry() -> dict:
    models = _load_json(os.path.join(L3_DIR, "models.json"))["models"]
    prompts = _load_json(os.path.join(L3_DIR, "prompts.json"))["prompts"]
    bench = [{"id": p["id"], "subject": p.get("subject", ""),
              "description": p.get("description", ""),
              "reference_images": p.get("reference_images", [])} for p in prompts]
    # Path is derived from the registry, not hardcoded: whatever the models are
    # tagged as (BYOM today; BYOA once that box is dropped in) is what the wizard
    # branches on. Mixed registries report the majority source.
    srcs = [m.get("source", "BYOM").upper() for m in models]
    path = max(set(srcs), key=srcs.count) if srcs else "BYOM"
    judge = _load_json(JUDGE_PATH) if os.path.exists(JUDGE_PATH) else None
    # `mode` tells the wizard which lane the live registry came from, so a
    # reload resumes on the right screen instead of defaulting to BYOM.
    return {"path": path, "mode": _read_mode(), "models": models,
            "benchmark": bench, "judge": judge}


def api_validate() -> dict:
    return byom_validator.validate_byom()


def api_pairs() -> dict:
    import pair_generator
    models = pair_generator.load_models()
    ids = [m["id"] for m in models]
    name = {m["id"]: m["name"] for m in models}
    pairs = pair_generator.generate_pairs(ids)
    return {"ok": True, "n": len(ids), "total": len(pairs),
            "pairs": [{"a": a, "b": b, "a_name": name[a], "b_name": name[b]}
                      for a, b in pairs]}


def api_results() -> dict:
    reg = _load_json(os.path.join(L3_DIR, "models.json"))["models"]
    name_by_id = {m["id"]: m["name"] for m in reg}
    ids = list(name_by_id)

    params = ["prompt_alignment", "texture_material", "form_proportion",
              "mesh_integrity", "overall"]
    # Post-fork (a fresh benchmark with no votes yet) the score files don't
    # exist. Return a valid, empty payload flagged `pending` so the UI shows
    # "vote first" instead of 500ing.
    if not all(os.path.exists(p) for p in (FUSED_PATH, ELOS_PATH, OBJ_PATH)):
        return {"models": reg, "fused": [], "param_elo": {}, "dims": {},
                "params": params, "gen_seconds": {}, "weights": {}, "pending": True}

    fused = _load_json(FUSED_PATH)
    leaderboard = [{"rank": r["rank"], "model": r["model"],
                    "name": name_by_id.get(r["model"], r["model"]),
                    "fused": r["fused_score"], "sub": r["subjective_elo"],
                    "obj": r["objective_score"], "worst": r["worst_axis_elo"]}
                   for r in fused["leaderboard"]]

    elos = _load_json(ELOS_PATH)["by_test_case"]
    obj = _load_json(OBJ_PATH)["by_test_case"]
    dim_keys = ["integrity", "texturing", "symmetry", "efficiency"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    param_elo, dims = {}, {}
    for mid in ids:
        param_elo[mid] = {p: mean([elos[tc][p][mid] for tc in elos
                                   if p in elos[tc] and mid in elos[tc][p]])
                          for p in params}
        dims[mid] = {k: mean([obj[tc][mid]["dimensions"][k] for tc in obj
                              if mid in obj[tc]]) for k in dim_keys}

    # real "cost" axis: mean generation time (seconds) from the benchmark log
    gen_seconds = {}
    if os.path.exists(GENLOG_PATH):
        tcs = _load_json(GENLOG_PATH).get("test_cases", {})
        for mid in ids:
            secs = [_gen_seconds(tcs[tc]["per_model"].get(mid, {}).get("generation_time"))
                    for tc in tcs if mid in tcs[tc].get("per_model", {})]
            gen_seconds[mid] = mean(secs)

    # the Bradley-Terry pairwise win matrices (aggregated across test cases per
    # parameter) — the raw evidence the subjective ELOs are fit from
    pairwise = _load_json(ELOS_PATH).get("pairwise", {})

    return {"models": reg, "fused": leaderboard, "param_elo": param_elo,
            "dims": dims, "params": params, "gen_seconds": gen_seconds,
            "weights": fused.get("primary_weights", {}), "pairwise": pairwise}


API_GET = {
    "/api/catalog": api_catalog,
    "/api/registry": api_registry,
    "/api/validate": api_validate,
    "/api/pairs": api_pairs,
    "/api/results": api_results,
    "/api/settings": lambda: settings.load(),
}


# --------------------------------------------------------------------------- #
#  Write-through API — the wizard's edits actually land on disk here, which is  #
#  what turns the Setup Studio from a mockup into the real control surface.     #
# --------------------------------------------------------------------------- #
MODELS_PATH = os.path.join(L3_DIR, "models.json")
PROMPTS_PATH = os.path.join(L3_DIR, "prompts.json")
JUDGE_PATH = os.path.join(ROOT, "1 - Registry Layer", "judge.json")
TESTCASES_DIR = os.path.join(ROOT, "TEST CASES")
LEADERBOARD_PATH = os.path.join(ROOT, "7 - Leaderboard", "leaderboard.html")
CACHE_ROOT = os.path.join(ROOT, "Cache Storage Layer", "cache")

# --------------------------------------------------------------------------- #
#  RUNS — one benchmark = one run.                                             #
#                                                                              #
#  A "run" is the votes + computed scores + registry for a given model set.    #
#  The demo (the 105 hand-cast ballots) is frozen ONCE as the permanent,       #
#  restorable `runs/example/`. When someone registers a DIFFERENT model set    #
#  (the fingerprint of names+sources changes), we fork: archive the current    #
#  run, then reset scores + the SESSION votes so a brand-new, empty ballot log  #
#  forms for them — their voting never touches the example, and re-saving the   #
#  SAME set (refining metadata) preserves everything. Demo-mode selections are  #
#  exempt from the cache wipe (see _save_registry_demo).                        #
#                                                                              #
#  RUNS_DIR / SESSION_VOTES_PATH are declared at the top of the module (the     #
#  ballot-storage block) because record_vote needs them.                        #
# --------------------------------------------------------------------------- #
EXAMPLE_DIR = os.path.join(RUNS_DIR, "example")
RUN_STATE_PATH = os.path.join(RUNS_DIR, "current.json")

# The small files that define a run's inputs + results. The multi-GB Layer 5
# mesh cache is deliberately NOT snapshotted — it's regenerable from TEST CASES
# via the schedule step, so snapshots stay tiny.
#
# NOTE "votes.json" maps to the SESSION log, not the committed 5A file. Both
# directions matter: snapshotting must capture the visitor's real ballots, and
# restoring must never write over the committed 105-ballot demo.
RUN_FILES = {
    "models.json": MODELS_PATH,
    "prompts.json": PROMPTS_PATH,
    "judge.json": JUDGE_PATH,
    "votes.json": SESSION_VOTES_PATH,
    "elos.json": ELOS_PATH,
    "objective_scores.json": OBJ_PATH,
    "fused_scores.json": FUSED_PATH,
    "viewer_data.json": VIEWER_DATA_PATH,
    "leaderboard.html": LEADERBOARD_PATH,
    "enriched_matchups.json": ENRICHED_PATH,
}


def _registry_fingerprint(models: list) -> str:
    import hashlib
    key = "|".join(sorted(
        f"{(m.get('name') or '').strip().lower()}::{(m.get('source') or 'BYOM').upper()}"
        for m in models))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _run_state() -> dict:
    if os.path.exists(RUN_STATE_PATH):
        try:
            return _load_json(RUN_STATE_PATH)
        except (ValueError, OSError):
            return {}
    return {}


def _read_fingerprint():
    return _run_state().get("fingerprint")


def _detect_mode() -> str:
    """Infer the mode of a registry written before mode was persisted.

    A registry whose (id, name) pairs are all catalog entries can only have come
    from demo mode, so a pre-existing checkout (and the committed M1..M7 state)
    is recognised as "demo" instead of being mislabelled BYOM.
    """
    try:
        models = _load_json(MODELS_PATH).get("models", [])
        cat = {(m["id"], m["name"].strip().lower()) for m in load_catalog()["models"]}
    except (ValueError, OSError, KeyError):
        return "byom"
    if not models:
        return "byom"
    return "demo" if all((m.get("id"), (m.get("name") or "").strip().lower()) in cat
                         for m in models) else "byom"


def _read_mode() -> str:
    """The persisted registry mode ("demo" | "byom").

    Stored in runs/current.json next to the fingerprint — the run pointer is
    already the "what is live right now" file, and keeping it out of models.json
    means the registry stays a pure model list that Layer 3/4 can read untouched.
    Falls back to _detect_mode() for state written before this key existed.
    """
    mode = str(_run_state().get("mode") or "").lower()
    return mode if mode in ("demo", "byom") else _detect_mode()


def _write_fingerprint(fp: str, mode: str | None = None) -> None:
    """Persist the run pointer. `mode` defaults to whatever is already stored."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    if mode not in ("demo", "byom"):
        mode = _read_mode()
    with open(RUN_STATE_PATH, "w") as f:
        json.dump({"fingerprint": fp, "mode": mode,
                   "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def _snapshot_run(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for fname, src in RUN_FILES.items():
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))


def ensure_example_snapshot() -> None:
    """Freeze the current run as the permanent example — once, and only if there
    are real votes to preserve. Idempotent: never overwrites an existing one."""
    if os.path.isdir(EXAMPLE_DIR):
        return
    try:
        if _load_json(ensure_session_votes()):     # non-empty ballots only
            _snapshot_run(EXAMPLE_DIR)
    except (ValueError, OSError):
        pass


def restore_example() -> dict:
    """Copy the frozen example run back over the live files (one-click 'demo').

    The example's ballots land in the SESSION log (RUN_FILES maps votes.json
    there), so 'Load example' resets the visitor's own copy and leaves the
    committed 105-ballot file byte-identical.
    """
    if not os.path.isdir(EXAMPLE_DIR):
        raise ValueError("no example snapshot exists yet")
    for fname, dest in RUN_FILES.items():
        src = os.path.join(EXAMPLE_DIR, fname)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
    # final_votes.json is DERIVED from the ballots; a leftover from another run
    # would otherwise outrank the restored ones in aggregate.load_records().
    try:
        os.remove(FINAL_VOTES_PATH)
    except FileNotFoundError:
        pass
    _write_fingerprint(_registry_fingerprint(_load_json(MODELS_PATH)["models"]),
                       _detect_mode())
    return {"ok": True, "restored": "example", "mode": _read_mode()}


def _fork_run(wipe_cache: bool = True) -> None:
    """Start a fresh benchmark: preserve the demo as the example (first time),
    archive the current live run, then wipe votes + scores (+ mesh cache) so an
    empty session votes.json forms. (save_registry restarts model ids at M1 on a
    BYOM fork.)

    `wipe_cache=False` is for the demo lane, whose ids come from the frozen
    catalog: the Layer 5 cache is keyed by (test case, model id) and those cells
    stay valid for any subset of the catalog, so throwing away minutes of mesh
    analysis on a selection change would be pure loss.

    The committed 5A votes.json is NEVER written here — only the session copy.
    """
    ensure_example_snapshot()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _snapshot_run(os.path.join(RUNS_DIR, f"archive_{stamp}"))
    for dest in (ELOS_PATH, OBJ_PATH, FUSED_PATH, VIEWER_DATA_PATH, ENRICHED_PATH,
                 FINAL_VOTES_PATH):
        try:
            os.remove(dest)
        except FileNotFoundError:
            pass
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(SESSION_VOTES_PATH, "w") as f:
        json.dump([], f)
    if wipe_cache and os.path.isdir(CACHE_ROOT):
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)


def _run_scripts(*specs) -> None:
    """Run layer scripts in order. A spec is either "path/to.py" or a tuple
    ("path/to.py", "--flag", "value") — the args form is what lets the refresh
    chain point hybrid.py at the session ballots instead of the committed ones."""
    import subprocess
    for spec in specs:
        script, *extra = (spec,) if isinstance(spec, str) else list(spec)
        r = subprocess.run([sys.executable, os.path.join(ROOT, script), *extra],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{script}: {(r.stderr or '')[-300:]}")


def _derive_match(name: str, all_names: list[str]) -> dict:
    """
    Build an asset_indexer match rule for a newly registered model.

    A file matches if its lowercased name contains ALL of `all` and NONE of
    `none`. Since Layer 2 now enforces `TC<nnn>_<ModelName>.<fmt>`, the model
    name itself is the reliable keyword. The `none` list exists for the subset
    trap: "Hyper 3D Rodin 2" is a substring of "Hyper 3D Rodin 2.5", so without
    it BOTH rules fire on the 2.5 file and asset_indexer discards the match as
    ambiguous.
    """
    low = name.strip().lower()
    longer = [o.strip().lower() for o in all_names
              if o.strip().lower() != low and low in o.strip().lower()]
    return {"all": [low], "none": longer} if longer else {"all": [low]}


def _demo_registry(payload: dict) -> tuple[list[dict], list[dict]]:
    """
    Resolve a demo selection into (models, prompts) taken from the catalog VERBATIM.

    Free-text rows are ignored on purpose: in demo mode the only thing the client
    gets to choose is WHICH catalog entries take part. Everything else — the id,
    the asset_indexer match rule, the vendor/credits metadata — is copied from
    benchmark/catalog.json, because a client-supplied id or match rule is exactly
    how the mesh cache and the 105 example ballots would end up pointing at the
    wrong model.
    """
    cat = load_catalog()
    by_name = {m["name"].strip().lower(): m for m in cat["models"]}

    models, seen = [], set()
    for row in (payload.get("models") or []):
        name = (row.get("name") or "").strip() if isinstance(row, dict) else str(row).strip()
        if not name:
            continue
        entry = by_name.get(name.lower())
        if entry is None:
            known = ", ".join(m["name"] for m in cat["models"])
            raise ValueError(f"{name!r} is not in the demo catalog (choose from: {known})")
        if entry["id"] in seen:                      # same model listed twice
            continue
        seen.add(entry["id"])
        models.append({
            "id": entry["id"],                       # FIXED by the catalog
            "name": entry["name"],
            "source": "BYOM",                        # the demo meshes are pre-generated
            "format": (entry.get("format") or "glb").lower(),
            "match": copy.deepcopy(entry.get("match") or {}),
            "metadata": {k: entry[k] for k in ("vendor", "credits")
                         if entry.get(k) not in (None, "")},
        })
    if not models:
        raise ValueError("demo mode needs at least one catalog model selected")

    wanted = {(b.get("id") if isinstance(b, dict) else b) for b in
              (payload.get("benchmark") or payload.get("test_cases") or [])}
    wanted = {str(w).strip().upper() for w in wanted if w}
    chosen = [t for t in cat["test_cases"] if t["id"] in wanted] if wanted \
        else list(cat["test_cases"])
    if not chosen:
        raise ValueError("demo mode needs at least one catalog test case selected")
    prompts = [{"id": t["id"], "subject": t.get("subject", ""),
                "description": t.get("description", ""),
                "reference_images": list(t.get("reference_images") or [])}
               for t in chosen]
    return models, prompts


def _save_registry_demo(payload: dict) -> dict:
    """
    Write a demo selection to models.json / prompts.json with catalog ids intact.

    Fork policy differs from BYOM by design:
      demo -> demo : NEVER forks. Ids are frozen by the catalog, so the ballots,
                     the scores and the (expensive) Layer 5 mesh cache all stay
                     correctly keyed for any subset — narrowing 7 models to 2
                     must not cost the visitor minutes of mesh analysis.
      byom -> demo : forks. The live run's "M1" is somebody else's model; its
                     cached meshes and votes would silently re-attach to catalog
                     models, so archive it and start clean.
    """
    models, prompts = _demo_registry(payload)
    new_fp = _registry_fingerprint(models)
    prev_fp, prev_mode = _read_fingerprint(), _read_mode()
    forked = prev_mode != "demo" and prev_fp is not None and prev_fp != new_fp
    if forked:
        _fork_run(wipe_cache=True)

    with open(MODELS_PATH, "w") as f:
        json.dump({"registry_version": 1, "n": len(models), "models": models},
                  f, indent=2)
    with open(PROMPTS_PATH, "w") as f:
        json.dump({"benchmark_version": 1, "prompts": prompts}, f, indent=2)
    if payload.get("judge"):
        with open(JUDGE_PATH, "w") as f:
            json.dump(payload["judge"], f, indent=2)

    _write_fingerprint(new_fp, "demo")
    return {"ok": True, "mode": "demo", "path": "BYOM", "forked": forked,
            "models": [{"id": m["id"], "name": m["name"]} for m in models],
            "test_cases": [p["id"] for p in prompts]}


def save_registry(payload: dict) -> dict:
    """
    Persist the wizard's registry to models.json / prompts.json / judge.json.

    `mode` selects the lane (default "byom", i.e. the original behaviour):
      "demo" — the client only names catalog models; entries are copied from
               benchmark/catalog.json verbatim (see _save_registry_demo).
      "byom" — free-text rows from the wizard, ids derived below.

    In BYOM mode model ids are STABLE BY NAME, never positional. Votes, the
    Layer 5 cache and every computed score are keyed by "M3"; if re-saving the
    registry renumbered a reordered list, all of that history would silently
    re-attach to the wrong model. So an existing name keeps its id, a new name
    takes the next free id, and a removed model's id is retired, not recycled.
    """
    mode = str(payload.get("mode") or "byom").strip().lower()
    if mode not in ("demo", "byom"):
        raise ValueError(f"unknown mode {mode!r} — expected 'demo' or 'byom'")
    if mode == "demo":
        return _save_registry_demo(payload)

    incoming = [m for m in payload.get("models", []) if (m.get("name") or "").strip()]
    if not incoming:
        raise ValueError("registry must contain at least one named model")

    names = [m["name"].strip() for m in incoming]
    if len(set(names)) != len(names):
        raise ValueError("duplicate model names — each must be unique")

    # ---- fork detection: a changed model SET starts a fresh benchmark ------- #
    # Fingerprint the resolved (name, source) pairs. If it differs from the
    # current run's, this is a new benchmark: fork (archive + reset votes/cache)
    # and restart ids at M1. Re-saving the same set (metadata edits) preserves.
    resolved_src = [{"name": m["name"].strip(),
                     "source": (m.get("source") or payload.get("path") or "BYOM")}
                    for m in incoming]
    new_fp = _registry_fingerprint(resolved_src)
    prev_fp = _read_fingerprint()
    forked = prev_fp is not None and prev_fp != new_fp
    if forked:
        _fork_run()
        existing = []                    # fresh ids from M1; cache/votes cleared
    else:
        existing = _load_json(MODELS_PATH).get("models", []) \
            if os.path.exists(MODELS_PATH) else []

    id_by_name = {m["name"]: m["id"] for m in existing}
    match_by_name = {m["name"]: m.get("match") for m in existing}
    used = {m["id"] for m in existing}

    def next_id() -> str:
        i = 1
        while f"M{i}" in used:
            i += 1
        used.add(f"M{i}")
        return f"M{i}"

    models = []
    for m in incoming:
        name = m["name"].strip()
        mid = id_by_name.get(name) or next_id()
        meta = dict(m.get("metadata") or {})
        for k in ("vendor", "credits"):
            if m.get(k) not in (None, ""):
                meta[k] = m[k]
        entry = {
            "id": mid,
            "name": name,
            "source": (m.get("source") or payload.get("path") or "BYOM").upper(),
            "format": (m.get("format") or "glb").lower(),
            # keep a hand-tuned rule if this model already had one
            "match": match_by_name.get(name) or _derive_match(name, names),
            "metadata": meta,
        }
        # BYOA needs where to call and which env var holds the key. Stored, but
        # never the key itself — see Layer 4 providers.py.
        if entry["source"] == "BYOA":
            entry["endpoint"] = m.get("endpoint", "")
            entry["api_key_env"] = m.get("api_key_env") or f"{mid}_API_KEY"
        models.append(entry)

    with open(MODELS_PATH, "w") as f:
        json.dump({"registry_version": 1, "n": len(models), "models": models}, f, indent=2)

    # ---- benchmark suite -------------------------------------------------- #
    bench = payload.get("benchmark")
    if bench:
        old = {p["id"]: p for p in _load_json(PROMPTS_PATH)["prompts"]} \
            if os.path.exists(PROMPTS_PATH) else {}
        prompts = []
        for b in bench:
            prev = old.get(b["id"], {})
            prompts.append({
                "id": b["id"],
                "subject": b.get("subject", prev.get("subject", "")),
                "description": b.get("description", prev.get("description", "")),
                "reference_images": b.get("reference_images",
                                          prev.get("reference_images", [])),
            })
        with open(PROMPTS_PATH, "w") as f:
            json.dump({"benchmark_version": 1, "prompts": prompts}, f, indent=2)

    # ---- judge config ----------------------------------------------------- #
    if payload.get("judge"):
        with open(JUDGE_PATH, "w") as f:
            json.dump(payload["judge"], f, indent=2)

    _write_fingerprint(new_fp, "byom")
    return {"ok": True, "mode": "byom",
            "models": [{"id": m["id"], "name": m["name"]} for m in models],
            "path": models[0]["source"], "forked": forked}


def save_upload(tc: str, model_name: str, ext: str, blob: bytes) -> dict:
    """
    Write one uploaded mesh into TEST CASES/<TC> GLB/ under the canonical
    `TC<nnn>_<ModelName>.<ext>` name that Layer 2 validates and Layer 4 indexes.
    Naming it correctly on the way IN is why the strict gate can stay strict.
    """
    tc = (tc or "").strip().upper()
    if not re.fullmatch(r"TC\d{3}", tc):
        raise ValueError(f"bad test-case id {tc!r} — expected TC001-style")
    name = (model_name or "").strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"bad model name {model_name!r}")
    ext = (ext or "glb").lower().lstrip(".")
    if ext not in {"glb", "gltf", "obj", "fbx"}:
        raise ValueError(f"unsupported format .{ext}")
    if not blob:
        raise ValueError("empty upload")

    folder = os.path.join(TESTCASES_DIR, f"{tc} GLB")
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{tc}_{name}.{ext}")
    with open(dest, "wb") as f:
        f.write(blob)
    return {"ok": True, "path": os.path.relpath(dest, ROOT),
            "bytes": len(blob), "replaced": False}


class ViewerHandler(SimpleHTTPRequestHandler):
    """Static file server + JSON API (GET /api/*, POST /api/vote|refresh)."""

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in API_GET:
            self._respond(API_GET[path])
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        handler = {
            "/api/vote": self._do_vote,
            "/api/refresh": self._do_refresh,
            "/api/schedule": self._do_schedule,
            "/api/registry": self._do_registry,
            "/api/upload": self._do_upload,
            "/api/settings": self._do_settings,
            "/api/load-example": self._do_load_example,
            "/api/reset-benchmark": lambda: reset_benchmark(),
        }.get(path)
        if handler is None:
            self.send_error(404)
        else:
            self._respond(handler)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _respond(self, fn):
        try:
            result = fn()
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except SystemExit as e:
            # run_pipeline signals a failed Layer 2 gate with SystemExit, which
            # derives from BaseException — so without this clause it escapes both
            # handlers below, kills the worker thread, and the browser sees a
            # dropped connection (curl reports 000) instead of the validation
            # report it needs to show the user.
            self.send_error(400, str(e) or "pipeline aborted")
        except (KeyError, ValueError) as e:
            self.send_error(400, str(e))
        except Exception as e:                       # noqa: BLE001
            self.send_error(500, str(e))

    def _do_vote(self):
        return record_vote(self._body())

    def _do_refresh(self):
        """Recompute the leaderboard from EXISTING votes (no re-voting).

        hybrid.py runs FIRST so the VLM-first-pass / human-review resolution is
        applied before Bradley-Terry sees the ballots. score.py (Layer 5B) runs
        before fuse.py because fuse hard-requires objective_scores.json — and
        re-deriving it from the current cache is what keeps the objective half
        honest after a new model set replaced the meshes.

        The ballots come from the SESSION log, passed explicitly to hybrid.py so
        the committed 105-ballot demo file is never read as the live input.
        aggregate.py is then called WITHOUT --votes on purpose: its default path
        makes it prefer final_votes.json (which hybrid just rebuilt from the
        session), whereas an explicit --votes would bypass the hybrid resolution
        entirely — see aggregate.load_records().
        """
        votes_path = ensure_session_votes()
        try:
            n_votes = len(_load_json(votes_path))
        except (ValueError, OSError):
            n_votes = 0
        if not n_votes:
            # Nothing to fit. Bail out with a flag instead of letting
            # Bradley-Terry / fuse.py die on an empty matrix — and, critically,
            # without letting aggregate.py fall back to the committed demo log.
            return {"ok": True, "pending": True, "votes": 0,
                    "detail": "no ballots in this session yet — vote first"}
        _run_scripts(("5A - Subjective Pipeline/hybrid.py",
                      "--votes", votes_path, "--out", FINAL_VOTES_PATH),
                     "5A - Subjective Pipeline/aggregate.py",
                     "5B - Objective Pipeline/score.py",
                     "6 - Fusion Layer/fuse.py",
                     "7 - Leaderboard/build_leaderboard.py")
        return {"ok": True, "votes": n_votes}

    def _do_schedule(self):
        """Run the setup pipeline (L2 gate -> L3 -> L4 -> L5) and rebuild the
        viewer feed. This is what the wizard's 'Run scheduler' button calls.

        The selected test cases' mesh packs are unzipped FIRST: the Layer 2 gate
        is strict about every registered model having a mesh for every test case,
        so on a fresh checkout (zips, no folders) it would fail the run before
        Layer 4 ever looked for assets.
        """
        body = self._body()
        try:
            tc_ids = [p["id"] for p in _load_json(PROMPTS_PATH)["prompts"]]
        except (ValueError, OSError, KeyError):
            tc_ids = []
        packs = ensure_testcases(tc_ids)
        matchups = run_pipeline(limit=body.get("limit"),
                                force=bool(body.get("force")),
                                validate=body.get("validate", True))
        n_viewer = build_viewer_data.build()
        ready = sum(1 for m in matchups if m.status in ("ready", "cached"))
        return {"ok": True, "matchups": len(matchups), "ready": ready,
                "viewer_entries": n_viewer, "packs": packs}

    def _do_registry(self):
        """Persist the wizard's registry edits to models.json / prompts.json."""
        return save_registry(self._body())

    def _do_load_example(self):
        """Restore the frozen demo run (the 105 example ballots + its scores)."""
        result = restore_example()
        try:
            build_viewer_data.build()        # rebuild the blind viewer feed
        except SystemExit:
            pass                              # cache may be absent after a fork
        return result

    def _do_settings(self):
        """Merge a partial settings patch (fusion weights, MCDA method, …)."""
        return settings.save(self._body())

    def _do_upload(self):
        """Store one uploaded mesh under TEST CASES/<TC> GLB/ canonically.

        Body: {tc, model_name, ext, content_b64}. The mesh is base64 so the
        whole upload is one JSON POST — fine for the prototype's file sizes.
        """
        b = self._body()
        blob = base64.b64decode(b.get("content_b64", ""))
        return save_upload(b.get("tc"), b.get("model_name"),
                           b.get("ext", "glb"), blob)

    def end_headers(self):
        # Never cache HTML/JSON: the single-file frontend changes often and a
        # stale cached copy silently shows old UI/data. (Meshes/images still cache.)
        p = self.path.split("?", 1)[0]
        if p.endswith((".html", ".json")) or p.endswith("/") or p in API_GET:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, *a):                       # keep console quiet
        pass


def run_pipeline(limit: int | None = None, force: bool = False,
                 validate: bool = True) -> list:
    """
    Stream matchups through Layers 3 -> 4 -> 5, one object at a time.

    Returns the list of fully enriched Matchup objects.
    """
    # --- Layer 2: BYOM input validation (pre-flight gate) ---
    if validate:
        report = byom_validator.validate_byom()
        if report["ok"]:
            print(f"[L2] BYOM input validated — {len(report['checks'])} checks passed")
        else:
            print(byom_validator.format_report(report))
            n_fail = sum(1 for c in report["checks"] if not c["ok"])
            raise SystemExit(
                f"[L2] input validation failed ({n_fail} check(s)). Fix the items "
                f"above, or re-run with --skip-validation to bypass the gate."
            )

    # --- Layer 3: origin ---
    matchups = generate_matchups.generate_matchups()
    if limit:
        matchups = matchups[:limit]
    print(f"[L3] generated {len(matchups)} matchups")

    # Load the asset manifest + roster once (read-only registry), then stream.
    manifest = scheduler.load_manifest()
    roster_by_id = scheduler._roster_by_id(manifest)

    print(f"[L4->L5] streaming each matchup through scheduler -> cache ...")
    ready = cached = 0
    for i, m in enumerate(matchups, 1):
        scheduler.schedule(m, manifest, roster_by_id)   # Layer 4: enrich availability
        if m.status == "ready":
            ready += 1
            cache.process(m, force=force)        # Layer 5: enrich metrics
            if m.status == "cached":
                cached += 1
        if i % 15 == 0 or i == len(matchups):
            print(f"        {i}/{len(matchups)}  (ready={ready}, cached={cached})")

    with open(ENRICHED_PATH, "w") as f:
        json.dump([m.to_dict() for m in matchups], f, indent=2)
    print(f"[out] enriched matchups -> {ENRICHED_PATH}")

    # --- Layer 5B: objective scoring (raw metrics -> 0-100 rubric + composite) ---
    # Runs here, on the cache we just filled, because the Fusion layer refuses to
    # run without objective_scores.json and _fork_run() deletes it for a new
    # model set. Leaving it to the CLI meant a fresh BYOM registry could vote but
    # never build a leaderboard.
    obj = objective_score.run()
    print(f"[5B] objective scores -> {len(obj['objective_leaderboard'])} model(s) "
          f"across {len(obj['by_test_case'])} test case(s)")
    return matchups


def launch_viewer(port: int = 8000, host: str = "127.0.0.1",
                  open_browser: bool = True) -> None:
    """Serve the project root and (locally) open the wizard in the browser."""
    # Freeze the current run as the restorable example, and seed the run
    # fingerprint so the first wizard save of the SAME model set isn't mistaken
    # for a new benchmark (which would reset the demo votes).
    ensure_registry()           # working roster, seeded from benchmark/catalog.json
    ensure_example_snapshot()
    ensure_session_votes()      # live ballot log, seeded from the example
    if _read_fingerprint() is None and os.path.exists(MODELS_PATH):
        try:
            _write_fingerprint(_registry_fingerprint(_load_json(MODELS_PATH)["models"]))
        except (ValueError, OSError):
            pass

    try:
        n_v = build_viewer_data.build()        # regenerate viewer_data.json
        print(f"[6A] viewer data ready ({n_v} matchups)")
    except SystemExit as e:                    # no cache yet (fresh deploy) — ok
        print(f"[6A] viewer data not built yet ({e}); vote step will build it")

    handler = partial(ViewerHandler, directory=ROOT)
    httpd = ThreadingHTTPServer((host, port), handler)
    shown = host if host != "0.0.0.0" else "127.0.0.1"
    entry = f"http://{shown}:{port}/{ENTRY_REL}"
    print(f"[serve] bound {host}:{port}")
    print(f"[entry] Setup Studio (Welcome) -> {entry}")
    print(f"[6A]    blind pair viewer      -> http://{shown}:{port}/{VIEWER_REL}")
    print("      (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(entry)
        except Exception:                      # headless / no display — fine
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[6A] viewer stopped.")
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="3D Model Eval pipeline driver")
    ap.add_argument("--no-viewer", action="store_true", help="run pipeline only")
    ap.add_argument("--viewer-only", action="store_true",
                    help="skip pipeline, just launch viewer (uses existing cache)")
    ap.add_argument("--force", action="store_true", help="recompute metrics")
    ap.add_argument("--limit", type=int, default=None, help="first N matchups only")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)),
                    help="viewer server port (env: PORT)")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="bind address; use 0.0.0.0 to deploy (env: HOST)")
    ap.add_argument("--no-open", action="store_true",
                    help="don't open a browser (set automatically when deploying)")
    ap.add_argument("--skip-validation", action="store_true",
                    help="bypass the Layer 2 BYOM input validation gate")
    args = ap.parse_args()

    # The working roster is generated state; seed it from the frozen catalog
    # before anything reads it (a fresh clone has no models.json at all).
    ensure_registry()

    if not args.viewer_only:
        # Same reason as _do_schedule: unzip the selected packs before the strict
        # Layer 2 gate looks for meshes on disk.
        try:
            ensure_testcases([p["id"] for p in _load_json(PROMPTS_PATH)["prompts"]])
        except (ValueError, OSError, KeyError):
            pass
        run_pipeline(limit=args.limit, force=args.force,
                     validate=not args.skip_validation)

    if not args.no_viewer:
        # Opening a browser only makes sense on a local bind; a container
        # (0.0.0.0) or an explicit --no-open serves headless.
        open_browser = not args.no_open and args.host in ("127.0.0.1", "localhost")
        launch_viewer(port=args.port, host=args.host, open_browser=open_browser)


if __name__ == "__main__":
    main()
