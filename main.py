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
for _layer in ("Layer 3 - Pair Generation",
               "Layer 4 - Generation Scheduler",
               "Layer 5 - Output Cache",
               "Layer 6A - Subjective Pipeline"):
    _p = os.path.join(ROOT, _layer)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generate_matchups          # noqa: E402  Layer 3
import scheduler                  # noqa: E402  Layer 4
import cache                      # noqa: E402  Layer 5
import build_viewer_data          # noqa: E402  Layer 6A

ENRICHED_PATH = os.path.join(ROOT, "enriched_matchups.json")
VOTES_PATH = os.path.join(ROOT, "Layer 6A - Subjective Pipeline", "votes.json")
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


class ViewerHandler(SimpleHTTPRequestHandler):
    """Static file server + a POST /api/vote endpoint for the 6A ballot."""

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

    def log_message(self, *a):                       # keep console quiet
        pass


def run_pipeline(limit: int | None = None, force: bool = False) -> list:
    """
    Stream matchups through Layers 3 -> 4 -> 5, one object at a time.

    Returns the list of fully enriched Matchup objects.
    """
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
    url = f"http://127.0.0.1:{port}/{VIEWER_REL}"
    print(f"[6A] serving at {url}")
    print("      (Ctrl+C to stop)")
    webbrowser.open(url)
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
    args = ap.parse_args()

    if not args.viewer_only:
        run_pipeline(limit=args.limit, force=args.force)

    if not args.no_viewer:
        launch_viewer(port=args.port)


if __name__ == "__main__":
    main()
