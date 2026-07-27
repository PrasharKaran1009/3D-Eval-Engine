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
import json
import os
import re
import shutil
import sys
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
for _layer in ("2 - Input Layer",
               "3 - Round Robin Pair Generator",
               "4 - Generation Prep Scheduler",
               "Cache Storage Layer",
               "5A - Subjective Pipeline",
               "6 - Fusion Layer"):
    _p = os.path.join(ROOT, _layer)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import byom_validator             # noqa: E402  Layer 2
import generate_matchups          # noqa: E402  Layer 3
import scheduler                  # noqa: E402  Layer 4
import cache                      # noqa: E402  Layer 5
import build_viewer_data          # noqa: E402  Layer 6A
import settings                    # noqa: E402  Layer 7 (global settings)

ENRICHED_PATH = os.path.join(ROOT, "enriched_matchups.json")
VOTES_PATH = os.path.join(ROOT, "5A - Subjective Pipeline", "votes.json")
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


def record_vote(payload: dict) -> dict:
    """
    Record one vote in votes.json. The browser sends only matchup_id + per-
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
    votes = []
    if os.path.exists(VOTES_PATH):
        with open(VOTES_PATH) as f:
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

    with open(VOTES_PATH, "w") as f:
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
    return {"path": path, "models": models, "benchmark": bench, "judge": judge}


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
#  run, then reset votes/scores/mesh-cache so a brand-new, empty votes.json     #
#  forms for them — their voting never touches the example, and re-saving the   #
#  SAME set (refining metadata) preserves everything.                          #
# --------------------------------------------------------------------------- #
RUNS_DIR = os.path.join(ROOT, "runs")
EXAMPLE_DIR = os.path.join(RUNS_DIR, "example")
RUN_STATE_PATH = os.path.join(RUNS_DIR, "current.json")

# The small files that define a run's inputs + results. The multi-GB Layer 5
# mesh cache is deliberately NOT snapshotted — it's regenerable from TEST CASES
# via the schedule step, so snapshots stay tiny.
RUN_FILES = {
    "models.json": MODELS_PATH,
    "prompts.json": PROMPTS_PATH,
    "judge.json": JUDGE_PATH,
    "votes.json": VOTES_PATH,
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


def _read_fingerprint():
    if os.path.exists(RUN_STATE_PATH):
        try:
            return _load_json(RUN_STATE_PATH).get("fingerprint")
        except (ValueError, OSError):
            return None
    return None


def _write_fingerprint(fp: str) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(RUN_STATE_PATH, "w") as f:
        json.dump({"fingerprint": fp,
                   "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def _snapshot_run(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for fname, src in RUN_FILES.items():
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))


def ensure_example_snapshot() -> None:
    """Freeze the current run as the permanent example — once, and only if there
    are real votes to preserve. Idempotent: never overwrites an existing one."""
    if not os.path.isdir(EXAMPLE_DIR) and os.path.exists(VOTES_PATH):
        try:
            if _load_json(VOTES_PATH):            # non-empty ballots only
                _snapshot_run(EXAMPLE_DIR)
        except (ValueError, OSError):
            pass


def restore_example() -> dict:
    """Copy the frozen example run back over the live files (one-click 'demo')."""
    if not os.path.isdir(EXAMPLE_DIR):
        raise ValueError("no example snapshot exists yet")
    for fname, dest in RUN_FILES.items():
        src = os.path.join(EXAMPLE_DIR, fname)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
    _write_fingerprint(_registry_fingerprint(_load_json(MODELS_PATH)["models"]))
    return {"ok": True, "restored": "example"}


def _fork_run() -> None:
    """Start a fresh benchmark: preserve the demo as the example (first time),
    archive the current live run, then wipe votes + scores + mesh cache so an
    empty votes.json forms. (save_registry restarts model ids at M1 on a fork.)"""
    ensure_example_snapshot()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _snapshot_run(os.path.join(RUNS_DIR, f"archive_{stamp}"))
    for dest in (ELOS_PATH, OBJ_PATH, FUSED_PATH, VIEWER_DATA_PATH, ENRICHED_PATH):
        try:
            os.remove(dest)
        except FileNotFoundError:
            pass
    os.makedirs(os.path.dirname(VOTES_PATH), exist_ok=True)
    with open(VOTES_PATH, "w") as f:
        json.dump([], f)
    if os.path.isdir(CACHE_ROOT):
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)


def _run_scripts(*rel_paths: str) -> None:
    import subprocess
    for script in rel_paths:
        r = subprocess.run([sys.executable, os.path.join(ROOT, script)],
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


def save_registry(payload: dict) -> dict:
    """
    Persist the wizard's registry to models.json / prompts.json / judge.json.

    Model ids are STABLE BY NAME, never positional. Votes (votes.json), the
    Layer 5 cache and every computed score are keyed by "M3"; if re-saving the
    registry renumbered a reordered list, all of that history would silently
    re-attach to the wrong model. So an existing name keeps its id, a new name
    takes the next free id, and a removed model's id is retired, not recycled.
    """
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

    _write_fingerprint(new_fp)
    return {"ok": True, "models": [{"id": m["id"], "name": m["name"]} for m in models],
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
        except (KeyError, ValueError) as e:
            self.send_error(400, str(e))
        except Exception as e:                       # noqa: BLE001
            self.send_error(500, str(e))

    def _do_vote(self):
        return record_vote(self._body())

    def _do_refresh(self):
        """Recompute the leaderboard from EXISTING votes (no re-voting).

        hybrid.py runs FIRST so the VLM-first-pass / human-review resolution is
        applied before Bradley-Terry sees the ballots.
        """
        _run_scripts("5A - Subjective Pipeline/hybrid.py",
                     "5A - Subjective Pipeline/aggregate.py",
                     "6 - Fusion Layer/fuse.py",
                     "7 - Leaderboard/build_leaderboard.py")
        return {"ok": True}

    def _do_schedule(self):
        """Run the setup pipeline (L2 gate -> L3 -> L4 -> L5) and rebuild the
        viewer feed. This is what the wizard's 'Run scheduler' button calls."""
        body = self._body()
        matchups = run_pipeline(limit=body.get("limit"),
                                force=bool(body.get("force")),
                                validate=body.get("validate", True))
        n_viewer = build_viewer_data.build()
        ready = sum(1 for m in matchups if m.status in ("ready", "cached"))
        return {"ok": True, "matchups": len(matchups), "ready": ready,
                "viewer_entries": n_viewer}

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
    return matchups


def launch_viewer(port: int = 8000, host: str = "127.0.0.1",
                  open_browser: bool = True) -> None:
    """Serve the project root and (locally) open the wizard in the browser."""
    # Freeze the current run as the restorable example, and seed the run
    # fingerprint so the first wizard save of the SAME model set isn't mistaken
    # for a new benchmark (which would reset the demo votes).
    ensure_example_snapshot()
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

    if not args.viewer_only:
        run_pipeline(limit=args.limit, force=args.force,
                     validate=not args.skip_validation)

    if not args.no_viewer:
        # Opening a browser only makes sense on a local bind; a container
        # (0.0.0.0) or an explicit --no-open serves headless.
        open_browser = not args.no_open and args.host in ("127.0.0.1", "localhost")
        launch_viewer(port=args.port, host=args.host, open_browser=open_browser)


if __name__ == "__main__":
    main()
