"""
2 - Input Layer (BYOM Input Validation)
=====================================================

Pre-flight GATE for the Bring-Your-Own-Models path. Before Layer 3 generates a
single matchup, this validates the uploaded model set against the Registry
(Layer 1 `models.json`) and the Benchmark Suite (`prompts.json`), implementing
the four pre-checks from updated.png's "BYOM Input -> Validation" panel:

    [1] Naming Convention OK    -> every asset is  TC<nnn>_<ModelName>.<fmt>
    [2] Supported Format        -> extension is one the registry declares
    [3] No Duplicates           -> no two files map to the same (TC, model)
    [4] All Test Cases Present   -> every benchmark TC x every model has a file

Layer 4's asset_indexer is deliberately *lenient* — it keyword-matches so a run
never dies on a stray typo. This layer is deliberately *strict*: it reports
every deviation and returns a pass/fail verdict, so the pipeline can refuse to
run on malformed input instead of silently working around it.

The scan/match/coverage logic is reused from asset_indexer (single source of
truth); this module adds the strict naming + format checks on top and packages
everything into a structured report the CLI (and, later, the frontend) can show.

Usage:
    python "byom_validator.py"                 # validate default TEST CASES
    python "byom_validator.py" --root "TEST CASES" --json
    from byom_validator import validate_byom
    report = validate_byom()                    # -> {"ok": bool, "checks": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER3 = os.path.join(HERE, "..", "1 - Registry Layer")
LAYER4 = os.path.join(HERE, "..", "4 - Generation Prep Scheduler")
for _p in (LAYER4, LAYER3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset_indexer  # noqa: E402  (Layer 4 — reuse scan/match/coverage)

MODELS_PATH = os.path.join(LAYER3, "models.json")
PROMPTS_PATH = os.path.join(LAYER3, "prompts.json")
DEFAULT_ROOT = asset_indexer.DEFAULT_ROOT

# Known 3D-asset extensions — used to tell a real (mis-formatted) model file
# apart from incidental junk (.DS_Store, notes.txt) which we simply ignore.
KNOWN_3D_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".3ds", ".blend"}

# Canonical filename: TC + 3 digits + underscore + name + supported extension.
NAMING_RE = re.compile(r"^TC\d{3}_.+\.[A-Za-z0-9]+$")


def load_registry(path: str = MODELS_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["models"]


def load_benchmark_ids(path: str = PROMPTS_PATH) -> list[str]:
    with open(path) as f:
        return [p["id"] for p in json.load(f)["prompts"]]


def _iter_asset_files(root: str):
    """Yield (tc, filename) for every 3D-asset file inside TC folders."""
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        tc = asset_indexer.tc_id_from_folder(entry)
        if tc is None:
            continue
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in KNOWN_3D_EXTS:
                yield tc, fname


def _check(name: str, ok: bool, details: list[str]) -> dict:
    return {"name": name, "ok": ok, "details": details}


def validate_byom(root: str = DEFAULT_ROOT) -> dict:
    """Run all four BYOM pre-checks. Returns {"ok", "root", "checks": [...]}."""
    if not os.path.isdir(root):
        return {
            "ok": False, "root": root,
            "checks": [_check("Root folder exists", False, [f"not found: {root}"])],
        }

    roster = load_registry()
    valid_exts = tuple(sorted({"." + m.get("format", "glb").lower() for m in roster}))
    assets = list(_iter_asset_files(root))

    # [1] Naming convention -------------------------------------------------
    naming_bad = [
        f"{tc}/{fn}" for tc, fn in assets
        if not NAMING_RE.match(fn) or fn != fn.strip()
    ]

    # [2] Supported format --------------------------------------------------
    fmt_bad = [
        f"{tc}/{fn}  (.{os.path.splitext(fn)[1].lstrip('.')} not in {list(valid_exts)})"
        for tc, fn in assets
        if os.path.splitext(fn)[1].lower() not in valid_exts
    ]

    # [3] No duplicates + unmatched (reuse Layer 4 matcher) -----------------
    index, problems = asset_indexer.build_index(root, roster)
    dups = [p for p in problems if p.startswith("DUPLICATE")]
    unmatched = [p for p in problems if p.startswith("UNMATCHED")]

    # [4] Coverage: every benchmark TC x every registered model -------------
    expected_tcs = load_benchmark_ids()
    expected_ids = [m["id"] for m in roster]
    name_by_id = {m["id"]: m["name"] for m in roster}
    gaps = []
    for tc in expected_tcs:
        present = index.get(tc, {})
        for mid in expected_ids:
            if mid not in present:
                gaps.append(f"MISSING: {tc} has no file for {mid} ({name_by_id[mid]})")

    checks = [
        _check("Naming Convention OK  (TC<nnn>_<ModelName>.<fmt>)", not naming_bad, naming_bad),
        _check("Supported Format", not fmt_bad, fmt_bad),
        _check("No Duplicates", not dups, dups),
        _check("Files Match a Registered Model", not unmatched, unmatched),
        _check(f"All Test Cases Present  ({len(expected_tcs)} TC x {len(expected_ids)} models)",
               not gaps, gaps),
    ]
    return {"ok": all(c["ok"] for c in checks), "root": root, "checks": checks}


def format_report(report: dict) -> str:
    lines = [f"BYOM Input Validation  —  {report['root']}", "=" * 60]
    for c in report["checks"]:
        lines.append(f"  [{'PASS' if c['ok'] else 'FAIL'}]  {c['name']}")
        for d in c["details"]:
            lines.append(f"           - {d}")
    lines.append("=" * 60)
    lines.append("RESULT: " + ("ALL CHECKS PASSED ✓" if report["ok"]
                                else "VALIDATION FAILED ✗ — fix the items above"))
    return "\n".join(lines)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Layer 2 - BYOM Input Validation")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="TEST CASES folder")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    report = validate_byom(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    _main()
