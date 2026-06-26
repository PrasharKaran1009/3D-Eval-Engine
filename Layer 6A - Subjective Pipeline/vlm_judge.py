"""
Layer 6A - VLM Judge  (provider-agnostic: OpenAI-compatible OR native Gemini)
=============================================================================

Casts the VLM as a blind pairwise reviewer. For each matchup it sends Model A's
and Model B's rendered views to a vision model and asks for a ballot on the 5
axes (A / B / tie). The ballot is written into votes.json tagged
voter="vlm:<model>", with blind A/B resolved back to the real (hidden) model ids
from viewer_data.json -- so the judge stays blind but Bradley-Terry can map back.

PROVIDER is auto-detected from VLM_BASE_URL:
  * native Gemini   -> base contains "generativelanguage.googleapis.com" (no /openai)
                       POST {base}/models/{model}:generateContent, x-goog-api-key,
                       inline_data images. (Use for AI Studio AQ.*/AIza* keys.)
  * OpenAI-compatible -> anything else (OpenRouter, Google's /openai endpoint, etc.)
                       POST {base}/chat/completions, Bearer auth, image_url data URLs.
So switching providers = edit .env only, no code change.

Renders come from render_views.py (the SAME viewer -> images), under
RENDERS_DIR/<matchup_id>/{A,B}_{textured,shaded,wireframe}.png.
Zero third-party deps (stdlib urllib). JSON ballot requested in-prompt and parsed.
Transient 503/429/5xx retried with backoff.

Config (.env at project root):
    VLM_BASE_URL   e.g. https://openrouter.ai/api/v1
    VLM_MODEL      e.g. google/gemini-2.0-flash-exp:free   (a FREE *vision* model)
    VLM_API_KEY    your key

Usage:
    python3 vlm_judge.py --limit 3
    python3 vlm_judge.py --dry-run --limit 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIEWER_DATA = os.path.join(HERE, "viewer_data.json")
VOTES_PATH = os.path.join(HERE, "votes.json")
RENDERS_DIR = os.path.join(HERE, "renders")

RETRY_CODES = {429, 500, 502, 503, 504}
MODES = ["textured", "shaded", "wireframe"]
PARAMS = [
    ("prompt_alignment", "Prompt Alignment", "textured"),
    ("texture_material", "Texture & Material", "textured"),
    ("form_proportion", "Form & Proportion", "shaded"),
    ("mesh_integrity", "Mesh Integrity", "wireframe"),
    ("overall", "Overall", "textured"),
]
PARAM_KEYS = [p[0] for p in PARAMS]
VALID = {"A", "B", "tie"}

SYSTEM = (
    "You are a meticulous, impartial judge comparing two AI-generated 3D models, "
    "shown blind as Model A and Model B. You are given rendered views of each in "
    "three modes: Textured (appearance), Shaded (form), and Wireframe (mesh topology). "
    "Judge each axis independently and return STRICT JSON only."
)


# ---------------------------------------------------------------- config / .env
def load_env() -> None:
    for path in (os.path.join(ROOT, ".env"), os.path.join(HERE, ".env")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def config() -> tuple[str, str, str]:
    return (os.environ.get("VLM_BASE_URL", "").rstrip("/"),
            os.environ.get("VLM_MODEL", ""),
            os.environ.get("VLM_API_KEY", ""))


def is_native_gemini(base: str) -> bool:
    return "generativelanguage.googleapis.com" in base and "/openai" not in base


# ---------------------------------------------------------------- images + prompt text
def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def gather_images(renders_dir: str, matchup_id) -> dict:
    base = os.path.join(renders_dir, str(matchup_id))
    out = {"A": {}, "B": {}}
    for side in ("A", "B"):
        for mode in MODES:
            p = os.path.join(base, f"{side}_{mode}.png")
            if os.path.exists(p):
                out[side][mode] = p
    return out


def instruction() -> str:
    axes = "\n".join(f"- {k}: {label} (judge mainly in {mode} view)"
                     for k, label, mode in PARAMS)
    return (
        "Compare Model A vs Model B on each axis below. For each, answer 'A' (A better), "
        "'B' (B better), or 'tie' (equal):\n" + axes +
        '\n\nReturn ONLY a JSON object, no prose, exactly: '
        '{' + ", ".join(f'"{k}": "A|B|tie"' for k in PARAM_KEYS) + "}"
    )


# ---------------------------------------------------------------- shared HTTP w/ retry
def _post(url: str, headers: dict, body: dict, max_retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"      [{e.code}] transient; retrying in {wait}s")
                time.sleep(wait)
                last_err = e
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                last_err = e
                continue
            raise
    raise last_err


# ---------------------------------------------------------------- per-provider call
def judge_openai(base: str, model: str, key: str, images: dict) -> str:
    content = [{"type": "text", "text": instruction()}]
    for side in ("A", "B"):
        content.append({"type": "text", "text": f"--- Model {side} ---"})
        for mode in MODES:
            if mode in images[side]:
                content.append({"type": "text", "text": f"{side} / {mode}:"})
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(images[side][mode])}"}})
    body = {"model": model, "max_tokens": 800, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}]}
    j = _post(f"{base}/chat/completions",
              {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
    if "choices" not in j:
        raise ValueError(f"no choices (model {j.get('model','?')}): {str(j)[:200]}")
    content = j["choices"][0].get("message", {}).get("content")
    if not content:                          # null content = model couldn't answer (e.g. no vision)
        raise ValueError(f"empty content from model {j.get('model','?')}: {str(j)[:160]}")
    return content


def judge_native(base: str, model: str, key: str, images: dict) -> str:
    parts = [{"text": SYSTEM + "\n\n" + instruction()}]
    for side in ("A", "B"):
        parts.append({"text": f"--- Model {side} ---"})
        for mode in MODES:
            if mode in images[side]:
                parts.append({"text": f"{side} / {mode}:"})
                parts.append({"inline_data": {"mime_type": "image/png",
                                              "data": _b64(images[side][mode])}})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": 800, "temperature": 0}}
    j = _post(f"{base}/models/{model}:generateContent",
              {"x-goog-api-key": key, "Content-Type": "application/json"}, body)
    cands = j.get("candidates")
    if not cands:
        raise ValueError(f"no candidates (safety block?): {j.get('promptFeedback', j)}")
    text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    if not text.strip():
        raise ValueError(f"empty response: {str(j)[:200]}")
    return text


def parse_ballot(text: str) -> dict:
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    raw = json.loads(s[i:j + 1])
    ballot = {k: raw[k] for k in PARAM_KEYS if raw.get(k) in VALID}
    if len(ballot) != len(PARAM_KEYS):
        raise ValueError(f"ballot missing/invalid axes "
                         f"{[k for k in PARAM_KEYS if k not in ballot]}: {raw}")
    return ballot


# ---------------------------------------------------------------- votes.json I/O
def load_votes() -> list:
    if os.path.exists(VOTES_PATH):
        with open(VOTES_PATH) as f:
            return json.load(f)
    return []


def save_votes(votes: list) -> None:
    with open(VOTES_PATH, "w") as f:
        json.dump(votes, f, indent=2)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="6A VLM judge -> votes.json")
    ap.add_argument("--limit", type=int, default=None, help="first N matchups only")
    ap.add_argument("--renders", default=RENDERS_DIR, help="renders dir")
    ap.add_argument("--dry-run", action="store_true", help="build requests, skip the API")
    args = ap.parse_args()
    renders_dir = args.renders

    load_env()
    base, model, key = config()
    if not args.dry_run and not (base and model and key):
        raise SystemExit("Missing config. Set VLM_BASE_URL / VLM_MODEL / VLM_API_KEY "
                         "in .env (see .env.example).")
    native = is_native_gemini(base)
    judge = judge_native if native else judge_openai
    print(f"provider: {'native-gemini' if native else 'openai-compatible'}  "
          f"model: {model}")

    with open(VIEWER_DATA) as f:
        matchups = json.load(f).get("matchups", [])
    if args.limit:
        matchups = matchups[:args.limit]

    voter = f"vlm:{model or 'dry-run'}"
    votes = load_votes()
    ok = fail = 0

    for m in matchups:
        mid = m["matchup_id"]
        images = gather_images(renders_dir, mid)
        if not images["A"] or not images["B"]:
            print(f"[skip] matchup {mid}: no renders in {renders_dir}/{mid}/ "
                  f"(run render_views.py first)")
            fail += 1
            continue
        if args.dry_run:
            print(f"[dry-run] matchup {mid}: "
                  f"{sum(len(images[s]) for s in ('A','B'))} images ready")
            ok += 1
            continue
        try:
            ballot = parse_ballot(judge(base, model, key, images))
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError) as e:
            print(f"[fail] matchup {mid}: {e}")
            fail += 1
            continue
        votes.append({
            "matchup_id": mid, "prompt_id": m["prompt_id"],
            "model_a": m["_a_model"], "model_b": m["_b_model"],
            "ballots": ballot, "voter": voter,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        save_votes(votes)
        ok += 1
        print(f"[ok] matchup {mid}: {ballot}")

    print(f"\nDone. judged={ok} failed/skipped={fail}. votes -> {VOTES_PATH}")


if __name__ == "__main__":
    main()
