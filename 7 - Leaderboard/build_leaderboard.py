"""
7 - Leaderboard page builder
==================================

Renders the fused Layer 7 results into a self-contained leaderboard.html
(dark/gold theme: Top-3 podium + ranked table). Data is baked into the HTML so
the page opens directly in a browser (no server needed). Re-run after more votes
to refresh.

Pulls:
    Layer 7 fused_scores.json      -> rank + Score (1300-centered fused ELO)
    Layer 6B objective_scores.json -> Mesh Quality / Texture star ratings
    Layer 1 models.json            -> display name, vendor, credit cost

Usage:  python3 build_leaderboard.py
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FUSED = os.path.join(ROOT, "6 - Fusion Layer", "fused_scores.json")
OBJ = os.path.join(ROOT, "5B - Objective Pipeline", "objective_scores.json")
REGISTRY = os.path.join(ROOT, "1 - Registry Layer", "models.json")
OUT = os.path.join(HERE, "leaderboard.html")


def load_meta(path: str = REGISTRY) -> dict:
    """
    Display metadata straight from the Registry (Layer 1) — the architecture's
    single source of truth for the roster.

    This used to be a hardcoded dict of the seven demo models, which meant any
    OTHER BYOM registry rendered as bare "M1…Mn" with no vendor and no cost. The
    wizard already persists vendor/credits into each model's `metadata` block
    (see main.save_registry), so read them from there.
    """
    with open(path) as f:
        models = json.load(f)["models"]
    meta = {}
    for m in models:
        extra = m.get("metadata") or {}
        credits = str(extra.get("credits", "") or "").strip()
        meta[m["id"]] = {
            "name": m.get("name") or m["id"],
            "vendor": str(extra.get("vendor", "") or ""),
            "credits": credits,          # "" = unknown; rendered as an em dash
        }
    return meta


def stars(score: float) -> str:
    n = max(1, min(5, round((score or 0) / 20)))
    return "★" * n + "☆" * (5 - n)


def dim_means() -> dict:
    """cross-test-case mean of each 6B dimension per model."""
    obj = json.load(open(OBJ))
    acc = defaultdict(lambda: defaultdict(list))
    for _tc, models in obj["by_test_case"].items():
        for mid, row in models.items():
            for k, v in row["dimensions"].items():
                acc[mid][k].append(v)
    return {mid: {k: statistics.mean(vs) for k, vs in d.items()} for mid, d in acc.items()}


def build() -> str:
    fused = json.load(open(FUSED))
    board = fused["leaderboard"]                      # already ranked
    dims = dim_means()
    n_tc = len(json.load(open(OBJ))["by_test_case"])
    meta = load_meta()

    def row_meta(mid):
        # A model in the fused board but not the registry (a retired id) still
        # gets a row — a leaderboard must account for every entrant.
        m = meta.get(mid, {"name": mid, "vendor": "", "credits": ""})
        d = dims.get(mid, {})
        return m, stars(d.get("integrity")), stars(d.get("texturing"))

    # ---- podium (top 3, arranged 2-1-3) ----
    top = board[:3]
    podium_order = [top[1], top[0], top[2]] if len(top) == 3 else top
    heights = {0: 70, 1: 100, 2: 55}                  # by display position
    podium_cells = ""
    for pos, r in enumerate(podium_order):
        m, _, _ = row_meta(r["model"])
        is_first = (r["rank"] == 1)
        podium_cells += f"""
        <div class="pod {'first' if is_first else ''}">
          {'<div class="star">★</div>' if is_first else ''}
          <div class="medal">{r['rank']}</div>
          <div class="pname">{m['name']}</div>
          <div class="pvendor">{m['vendor'].upper()}</div>
          <div class="pscore">{r['fused_score']:.0f}</div>
          <div class="bar" style="height:{heights.get(pos,55)}px"></div>
        </div>"""

    # ---- table rows ----
    rows = ""
    for r in board:
        m, mq, tx = row_meta(r["model"])
        rows += f"""
      <tr class="{'lead' if r['rank']==1 else ''}">
        <td class="rk">{r['rank']:02d}</td>
        <td class="mdl"><div class="mname">{m['name']}</div><div class="mven">{m['vendor'].upper()}</div></td>
        <td class="sc">{r['fused_score']:.0f}</td>
        <td class="st">{mq}</td>
        <td class="st">{tx}</td>
        <td class="cr{'' if m['credits'] else ' none'}">{m['credits'] or '—'}</td>
      </tr>"""

    # ---- bar chart: ALL models (padded scale so score gaps are visible) ----
    scores = [r["fused_score"] for r in board]
    smin, smax = min(scores), max(scores)
    span = (smax - smin) or 1.0
    bars = ""
    for r in board:
        m, _, _ = row_meta(r["model"])
        pct = 20 + 80 * (r["fused_score"] - smin) / span
        bars += f"""
        <div class="brow">
          <div class="blabel">{r['rank']}. {m['name']}</div>
          <div class="btrack"><div class="bfill {'top' if r['rank']==1 else ''}"
               style="width:{pct:.1f}%"></div></div>
          <div class="bval">{r['fused_score']:.0f}</div>
        </div>"""

    return _TEMPLATE.format(n_models=len(board), n_tc=n_tc,
                            podium=podium_cells, bars=bars, rows=rows, script=_SCRIPT)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>3D Eval · Leaderboard</title>
<style>
  :root {{ --bg:#0b0b0d; --card:#141416; --line:#262629; --txt:#f2f2f4; --muted:#8a8a93;
    --gold:#e0a32e; --gold2:#caa24a; --green:#5fd07a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    padding:32px; display:flex; flex-direction:column; align-items:center; gap:26px; }}
  .mono {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:1.5px; }}
  .card {{ width:min(1180px,96vw); background:var(--card); border:1px solid var(--line);
    border-radius:18px; padding:26px 30px; border-top:3px solid var(--gold);
    box-shadow:0 10px 40px rgba(0,0,0,.4); }}
  .head {{ display:flex; align-items:center; gap:18px; }}
  .logo {{ width:62px; height:62px; border-radius:14px;
    background:linear-gradient(145deg,#e0a32e,#9c6f1c); display:flex; align-items:center;
    justify-content:center; font-size:26px; color:#1a1206; box-shadow:0 0 26px rgba(224,163,46,.4); }}
  .head h1 {{ margin:0; font-size:26px; font-weight:750; }}
  .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  .refresh {{ margin-left:auto; align-self:flex-start; background:#1c1c1f; color:var(--gold);
    border:1px solid var(--gold2); border-radius:9px; padding:9px 16px; font-size:12px;
    cursor:pointer; }}
  .refresh:hover {{ background:var(--gold); color:#1a1206; }}
  .refresh:disabled {{ opacity:.5; cursor:wait; }}
  .panel {{ margin-top:24px; border:1px solid var(--line); border-radius:14px; padding:22px 26px 0; }}
  .ptitle {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
  .podium {{ display:flex; align-items:flex-end; justify-content:center; gap:60px; min-height:300px; }}
  .pod {{ display:flex; flex-direction:column; align-items:center; gap:6px; width:170px; text-align:center; }}
  .star {{ color:var(--gold); font-size:22px; filter:drop-shadow(0 0 8px rgba(224,163,46,.6)); }}
  .medal {{ width:52px; height:52px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:20px; font-weight:700; border:2px solid var(--gold2);
    color:var(--txt); }}
  .first .medal {{ background:var(--gold); color:#1a1206; border-color:var(--gold);
    box-shadow:0 0 26px rgba(224,163,46,.55); }}
  .pname {{ font-size:17px; font-weight:650; }}
  .pvendor {{ color:var(--muted); font-size:10.5px; letter-spacing:1px; }}
  .pscore {{ font-size:26px; font-weight:760; color:var(--gold); }}
  .first .pscore {{ font-size:30px; }}
  .bar {{ width:100%; border-radius:10px 10px 0 0; margin-top:6px;
    background:linear-gradient(180deg,#3a2f17,#262017); }}
  .first .bar {{ background:linear-gradient(180deg,#e0a32e,#9c6f1c);
    box-shadow:0 0 30px rgba(224,163,46,.45); }}

  .bars {{ display:flex; flex-direction:column; gap:11px; padding:4px 0 20px; }}
  .brow {{ display:flex; align-items:center; gap:14px; }}
  .blabel {{ width:180px; font-size:13px; color:var(--txt); text-align:right; }}
  .btrack {{ flex:1; height:22px; background:#1c1c1f; border-radius:6px; overflow:hidden; }}
  .bfill {{ height:100%; border-radius:6px; background:linear-gradient(90deg,#7d5816,#caa24a); }}
  .bfill.top {{ background:linear-gradient(90deg,#e0a32e,#f3c45e);
    box-shadow:0 0 14px rgba(224,163,46,.5); }}
  .bval {{ width:56px; color:var(--gold); font-weight:700; font-size:14px; }}

  table {{ width:min(1180px,96vw); border-collapse:collapse; }}
  thead th {{ color:var(--muted); font-size:11px; text-align:left; padding:14px 16px;
    border-bottom:1px solid var(--line); letter-spacing:1.5px; }}
  thead th.r, td.sc, td.cr, td.st {{ text-align:left; }}
  tbody td {{ padding:16px; border-bottom:1px solid #1c1c1f; vertical-align:middle; }}
  tr.lead td {{ background:linear-gradient(90deg,rgba(224,163,46,.10),transparent); }}
  .rk {{ color:var(--muted); font-family:ui-monospace,monospace; font-size:15px; width:48px; }}
  .mname {{ font-size:16px; font-weight:600; }}
  .mven {{ color:var(--muted); font-size:10.5px; letter-spacing:1px; margin-top:2px; }}
  .sc {{ color:var(--gold); font-size:19px; font-weight:700; width:110px; }}
  .st {{ color:var(--gold); letter-spacing:3px; font-size:15px; }}
  .cr {{ color:var(--txt); font-size:14px; }} .cr::after {{ content:" cr"; color:var(--muted); font-size:11px; }}
  /* a registry with no credit cost recorded shows "—", not "— cr" */
  .cr.none {{ color:var(--muted); }} .cr.none::after {{ content:""; }}
  .foot {{ color:var(--muted); font-size:11.5px; width:min(1180px,96vw); }}
</style></head>
<body>
  <div class="card">
    <div class="head">
      <div class="logo">◎</div>
      <div><h1>AI 3D Model Generation</h1>
        <div class="sub mono">1 SUBCATEGORY · {n_models} MODELS · {n_tc} TEST CASES · LIVE PREVIEW</div></div>
      <button id="refresh" class="refresh mono">↻ REFRESH</button>
    </div>
    <div class="panel">
      <div class="ptitle mono">TOP 3 LEADERS</div>
      <div class="podium">{podium}</div>
    </div>
    <div class="panel">
      <div class="ptitle mono">ALL {n_models} MODELS · FUSED SCORE</div>
      <div class="bars">{bars}</div>
    </div>
  </div>

  <table>
    <thead><tr>
      <th class="mono">#</th><th class="mono">MODEL</th><th class="mono r">SCORE</th>
      <th class="mono r">MESH QUALITY</th><th class="mono r">TEXTURE</th><th class="mono r">CREDITS</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <div class="foot mono">SCORE = fused subjective (Bradley-Terry ELO) + objective (geometry), centered at 1300 ·
    LIVE PREVIEW — sharpens as voting completes · stars from 6B Integrity / Texturing</div>
  {script}
</body></html>"""

# Kept out of the .format() template so its many JS braces don't need escaping.
_SCRIPT = """<script>
  const rb = document.getElementById('refresh');
  rb.onclick = async () => {
    const label = rb.textContent;
    rb.disabled = true; rb.textContent = 'RECOMPUTING…';
    try {
      const r = await fetch('/api/refresh', { method: 'POST' });
      if (!r.ok) throw new Error(await r.text());
      location.reload();   // redraw with the freshly recomputed scores
    } catch (e) {
      rb.disabled = false; rb.textContent = label;
      alert('Refresh needs the server. Open this page via\\n'
        + 'http://127.0.0.1:8000/Layer 6A... (not as a file://), then retry.\\n\\n' + e.message);
    }
  };
</script>"""


def main() -> None:
    if not (os.path.exists(FUSED) and os.path.exists(OBJ)):
        raise SystemExit("Run Layer 7 fuse.py and Layer 6B score.py first.")
    html = build()
    with open(OUT, "w") as f:
        f.write(html)
    print(f"[out] leaderboard -> {OUT}")


if __name__ == "__main__":
    main()
