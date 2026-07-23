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
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
for _layer in ("Layer 2 - Input Normalization",
               "Layer 3 - Pair Generation",
               "Layer 4 - Generation Scheduler",
               "Layer 5 - Output Cache",
               "Layer 6A - Subjective Pipeline"):
    _p = os.path.join(ROOT, _layer)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import byom_validator             # noqa: E402  Layer 2
import generate_matchups          # noqa: E402  Layer 3
import scheduler                  # noqa: E402  Layer 4
import cache                      # noqa: E402  Layer 5
import build_viewer_data          # noqa: E402  Layer 6A

ENRICHED_PATH = os.path.join(ROOT, "enriched_matchups.json")
VOTES_PATH = os.path.join(ROOT, "Layer 6A - Subjective Pipeline", "votes.json")
# Entry point: the Setup Studio wizard (starts at the Welcome page — pick BYOM
# and the judge). The blind pair viewer is reachable from its Evaluate step.
ENTRY_REL = "frontend/index.html"
VIEWER_REL = "Layer 6A - Subjective Pipeline/preview.html"


def _matchup_lookup() -> dict:
    """matchup_id -> {model_a, model_b, names, prompt_id} from enriched data."""
    with open(ENRICHED_PATH) as f:
        return {m["matchup_id"]: m for m in json.load(f)}


def record_vote(payload: dict) -> dict:
    """
    Append one vote to votes.json. The browser sends only matchup_id + per-
    parameter winners ('A' | 'B' | 'tie'); we resolve the real (blind) model ids
    here, server-side, so the human never saw them but Bradley-Terry can later.
    """
    lookup = _matchup_lookup()
    m = lookup.get(payload["matchup_id"])
    if m is None:
        raise KeyError(f"unknown matchup_id {payload['matchup_id']}")

    record = {
        "matchup_id": m["matchup_id"],
        "prompt_id": m["prompt_id"],
        "model_a": m["model_a"],          # A side == model_a (resolved here)
        "model_b": m["model_b"],
        "ballots": payload["ballots"],    # {param: 'A'|'B'|'tie'}
        "voter": payload.get("voter", "anon"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    votes = []
    if os.path.exists(VOTES_PATH):
        with open(VOTES_PATH) as f:
            votes = json.load(f)
    votes.append(record)
    with open(VOTES_PATH, "w") as f:
        json.dump(votes, f, indent=2)
    return {"ok": True, "total_votes": len(votes)}


# --------------------------------------------------------------------------- #
#  Read-through API for the frontend "Setup Studio" — serves the REAL registry, #
#  validation, pairs and results so the wizard runs live instead of on mocks.   #
# --------------------------------------------------------------------------- #
L3_DIR = os.path.join(ROOT, "Layer 3 - Pair Generation")
FUSED_PATH = os.path.join(ROOT, "Layer 7 - Fusion", "fused_scores.json")
ELOS_PATH = os.path.join(ROOT, "Layer 6A - Subjective Pipeline", "elos.json")
OBJ_PATH = os.path.join(ROOT, "Layer 6B - Objective Pipeline", "objective_scores.json")
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
    return {"path": "BYOM", "models": models, "benchmark": bench}


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

    fused = _load_json(FUSED_PATH)
    leaderboard = [{"rank": r["rank"], "model": r["model"],
                    "name": name_by_id.get(r["model"], r["model"]),
                    "fused": r["fused_score"], "sub": r["subjective_elo"],
                    "obj": r["objective_score"], "worst": r["worst_axis_elo"]}
                   for r in fused["leaderboard"]]

    elos = _load_json(ELOS_PATH)["by_test_case"]
    obj = _load_json(OBJ_PATH)["by_test_case"]
    params = ["prompt_alignment", "texture_material", "form_proportion",
              "mesh_integrity", "overall"]
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

    return {"models": reg, "fused": leaderboard, "param_elo": param_elo,
            "dims": dims, "params": params, "gen_seconds": gen_seconds,
            "weights": fused.get("primary_weights", {})}


API_GET = {
    "/api/registry": api_registry,
    "/api/validate": api_validate,
    "/api/pairs": api_pairs,
    "/api/results": api_results,
}


class ViewerHandler(SimpleHTTPRequestHandler):
    """Static file server + JSON API (GET /api/*, POST /api/vote|refresh)."""

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in API_GET:
            self._respond(API_GET[path])
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/vote":
            self._respond(self._do_vote)
        elif self.path == "/api/refresh":
            self._respond(self._do_refresh)
        else:
            self.send_error(404)

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
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        return record_vote(payload)

    def _do_refresh(self):
        """Recompute the leaderboard from EXISTING votes (no re-voting)."""
        import subprocess
        for script in ("Layer 6A - Subjective Pipeline/aggregate.py",
                       "Layer 7 - Fusion/fuse.py",
                       "Layer 8 - Leaderboard/build_leaderboard.py"):
            r = subprocess.run([sys.executable, os.path.join(ROOT, script)],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{script}: {(r.stderr or '')[-200:]}")
        return {"ok": True}

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

    # Load the asset manifest once (read-only registry), then stream.
    manifest = scheduler.load_manifest()

    print(f"[L4->L5] streaming each matchup through scheduler -> cache ...")
    ready = cached = 0
    for i, m in enumerate(matchups, 1):
        scheduler.schedule(m, manifest)          # Layer 4: enrich availability
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


def launch_viewer(port: int = 8000) -> None:
    """Serve the project root and open the blind pair viewer in the browser."""
    n_v = build_viewer_data.build()            # regenerate viewer_data.json
    print(f"[6A] viewer data ready ({n_v} matchups)")

    handler = partial(ViewerHandler, directory=ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    entry = f"http://127.0.0.1:{port}/{ENTRY_REL}"
    print(f"[entry] Setup Studio (Welcome) -> {entry}")
    print(f"[6A]    blind pair viewer      -> http://127.0.0.1:{port}/{VIEWER_REL}")
    print("      (Ctrl+C to stop)")
    webbrowser.open(entry)
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
    ap.add_argument("--port", type=int, default=8000, help="viewer server port")
    ap.add_argument("--skip-validation", action="store_true",
                    help="bypass the Layer 2 BYOM input validation gate")
    args = ap.parse_args()

    if not args.viewer_only:
        run_pipeline(limit=args.limit, force=args.force,
                     validate=not args.skip_validation)

    if not args.no_viewer:
        launch_viewer(port=args.port)


if __name__ == "__main__":
    main()
