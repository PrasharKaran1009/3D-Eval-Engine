"""
Layer 6A - Render views from the SAME viewer  (Playwright)
==========================================================

Reuses the existing Three.js viewer (preview.html) as the single rendering
source -- no second renderer. Drives it headlessly: for each matchup, for each
mode, screenshots panel A and panel B into

    renders/<matchup_id>/{A,B}_{textured,shaded,wireframe}.png

which is exactly what vlm_judge.py consumes.

It only touches the viewer's existing DOM (test-case + matchup <select>s, the
mode buttons, and the panel elements) -- nothing internal -- so the viewer needs
no changes.

PREREQUISITES (not installed in this repo yet -- this is the one piece not
verified here):
    pip install playwright
    playwright install chromium
    # and serve the viewer in another terminal:
    python3 main.py --viewer-only

Usage:
    python3 render_views.py --limit 3
    python3 render_views.py --url http://127.0.0.1:8000/Layer%206A%20-%20Subjective%20Pipeline/preview.html

PROTOTYPE: one framed view per mode. Multi-angle (orbit + capture) is a future
improvement for better defect detection -- noted, not a blocker.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER_DATA = os.path.join(HERE, "viewer_data.json")
RENDERS_DIR = os.path.join(HERE, "renders")
DEFAULT_URL = ("http://127.0.0.1:8000/Layer%206A%20-%20Subjective%20Pipeline/"
               "preview.html")
MODES = ["textured", "shaded", "wireframe"]


def matchup_index(matchups: list) -> list:
    """Attach each matchup's index *within its test case* (what the viewer's
    matchup <select> uses)."""
    seen: dict = defaultdict(int)
    rows = []
    for m in matchups:
        tc = m["prompt_id"]
        rows.append({"matchup_id": m["matchup_id"], "prompt_id": tc, "idx": seen[tc]})
        seen[tc] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Render matchup views from the viewer")
    ap.add_argument("--url", default=DEFAULT_URL, help="served preview.html URL")
    ap.add_argument("--out", default=RENDERS_DIR, help="output renders dir")
    ap.add_argument("--limit", type=int, default=None, help="first N matchups only")
    ap.add_argument("--timeout", type=int, default=30000, help="per-load timeout (ms)")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright not installed. Run:\n"
            "    pip install playwright && playwright install chromium")

    with open(VIEWER_DATA) as f:
        matchups = json.load(f).get("matchups", [])
    rows = matchup_index(matchups)
    if args.limit:
        rows = rows[:args.limit]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900},
                                device_scale_factor=2)
        page.goto(args.url, wait_until="load")
        # <option> elements are never "visible" (only on dropdown open) -> wait attached
        page.wait_for_selector("#tcsel option", state="attached", timeout=args.timeout)

        for row in rows:
            mid, tc, idx = row["matchup_id"], row["prompt_id"], row["idx"]
            # navigate via the existing selects (fire their change handlers)
            page.evaluate(
                """([tc, idx]) => {
                    const tcsel = document.getElementById('tcsel');
                    if (tcsel.value !== tc) { tcsel.value = tc; tcsel.dispatchEvent(new Event('change')); }
                    const sel = document.getElementById('sel');
                    sel.value = String(idx); sel.dispatchEvent(new Event('change'));
                }""", [tc, idx])

            # wait until both panels finish loading (the .load overlay hides)
            try:
                page.wait_for_function(
                    "[...document.querySelectorAll('.load')].every(e => e.style.display === 'none')",
                    timeout=args.timeout)
            except Exception:
                print(f"[warn] matchup {mid}: load timeout, capturing anyway")

            out_dir = os.path.join(args.out, str(mid))
            os.makedirs(out_dir, exist_ok=True)

            for mode in MODES:
                page.click(f"#modes button[data-mode='{mode}']")
                page.wait_for_timeout(500)        # let a frame render in the new mode
                # Use page.screenshot(clip=...) not locator.screenshot(): the latter
                # waits for the element to be "stable", which never happens on a
                # continuously-animating WebGL canvas (rAF loop) -> it times out.
                for side, sel_id in (("A", "#panelA"), ("B", "#panelB")):
                    box = page.locator(sel_id).bounding_box()
                    page.screenshot(path=os.path.join(out_dir, f"{side}_{mode}.png"),
                                    clip={"x": box["x"], "y": box["y"],
                                          "width": box["width"], "height": box["height"]})

            print(f"[ok] matchup {mid} ({tc}) -> {out_dir}/  ({len(MODES)*2} images)")

        browser.close()

    print(f"\nDone. renders -> {args.out}")


if __name__ == "__main__":
    main()
