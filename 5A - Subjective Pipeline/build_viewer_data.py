"""
Layer 6A - Viewer Data Builder
==============================

Turns the enriched matchups (from the pipeline) into a slim JSON the browser
viewer can consume. Each entry points at the two cached model.glb files via
root-absolute URLs (the HTTP server is rooted at the project dir).

Blind evaluation: model display names are deliberately NOT sent to the client.
Each side is just "A" / "B". The hidden model ids are kept only so the voting
step can map a vote back to the real model server-side.

A/B SIDE RANDOMIZATION (bias control)
-------------------------------------
Layer 3 always emits pairs in a fixed order (lower model id first), so without
intervention the same model would always sit in panel A. Judges have a
well-documented left/position bias, so a fixed layout systematically favors
whichever side that is — a confound that lands directly on the scores.

So each matchup is deterministically side-swapped: whether model_a is shown as
panel A or panel B is decided by a hash of the matchup, seeded stably. Same
matchup => same layout on every rebuild (reproducible, and a mid-session
rebuild won't flip what a judge is looking at), but across the field the real
models are split roughly evenly between the two panels.

`_a_model` / `_b_model` in the output are the models AS DISPLAYED (post-swap).
main.record_vote reads THESE back, so a ballot's "A beat B" always maps to the
model the judge actually saw on the left. The prompt text + reference images
are included too, so the judge (human or VLM) can assess prompt_alignment —
which carries the highest fusion weight and is otherwise unjudgeable from
geometry alone.
"""

from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENRICHED = os.path.join(ROOT, "enriched_matchups.json")
VOTES = os.path.join(HERE, "votes.json")
OUT = os.path.join(HERE, "viewer_data.json")
PROMPTS = os.path.join(ROOT, "1 - Registry Layer", "prompts.json")

# Where Layer 5 caches each model.glb, as a URL path from the server root.
CACHE_URL = "/Cache Storage Layer/cache/{prompt}/{model}/model.glb"

# Salt for the side-swap hash. Change it to reshuffle every layout at once.
SWAP_SEED = "3d-eval-blind-v1"


def side_swapped(matchup_id, model_a: str, model_b: str) -> bool:
    """Deterministic per-matchup coin flip: True => show model_b on panel A.

    Uses a stable hash (not Python's salted hash()) so the layout is identical
    across processes and reruns — the de-randomization in record_vote depends on
    exactly this decision being reproducible.
    """
    h = hashlib.sha256(f"{SWAP_SEED}|{matchup_id}|{model_a}|{model_b}".encode())
    return h.digest()[0] & 1 == 1


def _prompts_by_id(path: str = PROMPTS) -> dict:
    if not os.path.exists(path):
        return {}
    return {p["id"]: p for p in json.load(open(path)).get("prompts", [])}


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
    prompts = _prompts_by_id()

    entries = []
    for m in matchups:
        if m.get("status") != "cached":
            continue
        mid, tc = m["matchup_id"], m["prompt_id"]
        real_a, real_b = m["model_a"], m["model_b"]

        # Decide the displayed layout. If swapped, real model_b becomes panel A.
        swap = side_swapped(mid, real_a, real_b)
        disp_a, disp_b = (real_b, real_a) if swap else (real_a, real_b)

        p = prompts.get(tc, {})
        entries.append({
            "matchup_id": mid,
            "prompt_id": tc,
            # prompt context so the judge can score prompt_alignment
            "subject": p.get("subject", ""),
            "description": p.get("description", ""),
            "reference_images": p.get("reference_images", []),
            # URLs the browser loads (root-absolute; spaces encoded client-side)
            "a_glb": CACHE_URL.format(prompt=tc, model=disp_a),
            "b_glb": CACHE_URL.format(prompt=tc, model=disp_b),
            # VLM first-pass ballot (hybrid): pre-fills the human review, or null
            "vlm_ballot": vlm.get(mid),
            # hidden — DISPLAYED sides (post-swap). record_vote maps ballots back
            # through these, so "A" always means the model shown on panel A.
            "_a_model": disp_a,
            "_b_model": disp_b,
            "_swapped": swap,
        })

    with open(out_path, "w") as f:
        json.dump({"matchups": entries}, f, indent=2)
    return len(entries)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} matchups to {OUT}")
