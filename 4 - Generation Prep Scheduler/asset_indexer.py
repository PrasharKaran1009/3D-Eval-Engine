"""
Layer 4 - Asset Indexer (Generation Scheduler, ingest mode)
===========================================================

In the original design Layer 4 *generates* model outputs on demand. For this
MVP all 35 GLBs already exist (7 models x 5 test cases), so generation is
skipped. Layer 4's job becomes:

    scan -> match -> validate -> index

It walks the TEST CASES folder, matches every .glb to a model id, verifies the
full 7x5 grid is present, and writes an asset manifest that downstream layers
(5 storage, 5A voting, 5B metrics) use to locate files.

Why keyword matching (not exact filenames):
    Filenames are now normalized to `TC00x_ModelName.glb`, but display names
    still differ from what's on disk in minor ways — e.g. "Hyper 3D Rodin 2.5"
    is filed as "Hyper Rodin 2.5" (no "3D"), "Pixal 3D" is lowercased "Pixal 3d".
    Exact matching would silently drop those files. Instead we derive the
    test-case id from the *folder* name and match each model by unique keywords.
    Those keywords now live in the Registry (Layer 3 `models.json`, each model's
    `match: {all, none}` block) so the registry is the single source of truth.

Usage:
    python asset_indexer.py
    python asset_indexer.py --root "TEST CASES" --out manifest.json
"""

from __future__ import annotations

import argparse
import json
import os

# Roster lives in Layer 3. We read ids/names from there so this stays in sync.
LAYER3_MODELS = os.path.join(
    os.path.dirname(__file__), "..", "1 - Registry Layer", "models.json"
)
DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "TEST CASES")

# Match rules now live in the Registry (models.json), one `match: {all, none}`
# block per model: a filename (lowercased) matches a model id if it contains ALL
# of `all` and NONE of `none`. Brand words are unique except Rodin 2 vs 2.5,
# which split on the "2.5" token.


def load_roster(path: str = LAYER3_MODELS) -> list[dict]:
    """Load the model registry entries [{'id','name','match',...}, ...]."""
    with open(path) as f:
        return json.load(f)["models"]


def match_rules(roster: list[dict]) -> dict[str, dict]:
    """Build {model_id: {'all':[...], 'none':[...]}} from the registry roster."""
    return {m["id"]: m.get("match", {}) for m in roster}


def match_model(filename: str, rules: dict[str, dict]) -> str | None:
    """Return the model id whose registry rule matches this filename, or None."""
    f = filename.lower()
    hits = []
    for model_id, rule in rules.items():
        if all(tok in f for tok in rule.get("all", [])) and \
           not any(tok in f for tok in rule.get("none", [])):
            hits.append(model_id)
    # Exactly one rule should fire; ambiguity means a rule needs tightening.
    return hits[0] if len(hits) == 1 else None


def tc_id_from_folder(folder_name: str) -> str | None:
    """'TC001 GLB' -> 'TC001'. Returns None if it doesn't look like a TC folder."""
    token = folder_name.strip().split()[0].upper()
    return token if token.startswith("TC") else None


def build_index(root: str, roster: list[dict]) -> tuple[dict, list[str]]:
    """
    Walk `root`, match every .glb, and build {tc: {model_id: relpath}}.

    Returns (index, problems). `problems` is a list of human-readable warnings
    (unmatched files, duplicates) — empty means a clean scan.
    """
    index: dict[str, dict[str, str]] = {}
    problems: list[str] = []

    # Registry-driven: match keywords and accepted file formats both come from
    # models.json, so this layer never hardcodes model knowledge.
    rules = match_rules(roster)
    valid_exts = tuple(sorted({"." + m.get("format", "glb").lower() for m in roster}))

    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        tc = tc_id_from_folder(entry)
        if tc is None:
            continue
        index.setdefault(tc, {})

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(valid_exts):
                continue  # skips .DS_Store and anything else
            model_id = match_model(fname, rules)
            rel = os.path.join(root, entry, fname)
            if model_id is None:
                problems.append(f"UNMATCHED: {rel}")
                continue
            if model_id in index[tc]:
                problems.append(
                    f"DUPLICATE: {tc}/{model_id} already set "
                    f"({index[tc][model_id]}) vs {rel}"
                )
                continue
            index[tc][model_id] = rel

    return index, problems


def validate(index: dict, roster: list[dict]) -> list[str]:
    """Check every test case has all models present. Returns list of gaps."""
    expected_ids = [m["id"] for m in roster]
    gaps: list[str] = []
    for tc in sorted(index):
        for mid in expected_ids:
            if mid not in index[tc]:
                gaps.append(f"MISSING: {tc} has no file for {mid}")
    return gaps


def _main() -> None:
    parser = argparse.ArgumentParser(description="Layer 4 - Asset Indexer")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="TEST CASES folder")
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "manifest.json"),
        help="Output manifest path",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(
            f"TEST CASES folder not found at: {args.root}\n"
            f"Move your 'TEST CASES' folder into the project root and retry."
        )

    roster = load_roster()
    name_by_id = {m["id"]: m["name"] for m in roster}
    index, problems = build_index(args.root, roster)
    gaps = validate(index, roster)

    # Report
    print(f"Scanned: {args.root}")
    print(f"Test cases found: {len(index)}  ({', '.join(sorted(index))})\n")
    for tc in sorted(index):
        print(f"{tc}:")
        for mid in [m["id"] for m in roster]:
            mark = "ok " if mid in index[tc] else "!! "
            fpath = os.path.basename(index[tc].get(mid, "<MISSING>"))
            print(f"  {mark}{mid} ({name_by_id[mid]:<20}) -> {fpath}")
        print()

    total = sum(len(v) for v in index.values())
    print(f"Matched files: {total} / {len(roster) * len(index)} expected")

    if problems or gaps:
        print("\n--- ISSUES ---")
        for p in problems + gaps:
            print(f"  {p}")
        print("\nManifest NOT written until issues are resolved.")
        raise SystemExit(1)

    manifest = {
        "root": args.root,
        "test_cases": sorted(index),
        "models": roster,
        "assets": index,
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nClean scan. Manifest written to: {args.out}")


if __name__ == "__main__":
    _main()
