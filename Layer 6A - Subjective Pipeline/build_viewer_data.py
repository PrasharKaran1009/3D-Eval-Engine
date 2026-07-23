"""
Layer 6A - Viewer Data Builder
==============================

Turns the enriched matchups (from the pipeline) into a slim JSON the browser
viewer can consume. Each entry points at the two cached model.glb files via
root-absolute URLs (the HTTP server is rooted at the project dir).

Blind evaluation: model display names are deliberately NOT sent to the client.
Each side is just "A" / "B". The hidden model ids are kept only so a future
voting step can map a vote back to the real model server-side.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENRICHED = os.path.join(ROOT, "enriched_matchups.json")
VOTES = os.path.join(HERE, "votes.json")
OUT = os.path.join(HERE, "viewer_data.json")

# Where Layer 5 caches each model.glb, as a URL path from the server root.
CACHE_URL = "/Layer 5 - Output Cache/cache/{prompt}/{model}/model.glb"


def _vlm_ballots(votes_path: str = VOTES) -> dict:
    """matchup_id -> latest VLM first-pass ballot (for the hybrid review step)."""
    if not os.path.exists(votes_path):
        return {}
    out: dict = {}
    for r in json.load(open(votes_path)):
        if str(r.get("voter", "")).startswith("vlm"):
            out[r["matchup_id"]] = r.get("ballots", {})   # last one wins
    return out


def build(enriched_path: str = ENRICHED, out_path: str = OUT) -> int:
    """Build viewer_data.json from enriched_matchups.json. Returns count."""
    if not os.path.exists(enriched_path):
        raise SystemExit(
            f"{enriched_path} not found. Run the pipeline first (python main.py).")

    with open(enriched_path) as f:
        matchups = json.load(f)

    vlm = _vlm_ballots()

    entries = []
    for m in matchups:
        if m.get("status") != "cached":
            continue
        entries.append({
            "matchup_id": m["matchup_id"],
            "prompt_id": m["prompt_id"],
            # URLs the browser loads (root-absolute; spaces encoded client-side)
            "a_glb": CACHE_URL.format(prompt=m["prompt_id"], model=m["model_a"]),
            "b_glb": CACHE_URL.format(prompt=m["prompt_id"], model=m["model_b"]),
            # VLM first-pass ballot (hybrid): pre-fills the human review, or null
            "vlm_ballot": vlm.get(m["matchup_id"]),
            # hidden — for server-side vote mapping later, UI never shows these
            "_a_model": m["model_a"],
            "_b_model": m["model_b"],
        })

    with open(out_path, "w") as f:
        json.dump({"matchups": entries}, f, indent=2)
    return len(entries)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} matchups to {OUT}")
