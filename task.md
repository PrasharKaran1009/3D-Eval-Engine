# Remaining BYOM tasks

Demo mode, the test-case packs, Layer 5B wiring and the registry-driven leaderboard are done and
verified end to end (3 models x 2 cases: zip -> extract -> L2 -> L5 -> 5B -> vote -> fuse ->
leaderboard, with the committed 105 ballots byte-identical afterwards). These three are what's
left, highest impact first.

---

## 1. The wizard reaches Results with nothing selected, and Results shows a stale leaderboard

**What's wrong** — `Continue` never validates anything, so you can click straight from Welcome to
Results with zero models registered. Worse, Results then renders a full 7-model board, because it
draws whatever `GET /api/results` returned rather than anything tied to your selection. Both the
empty case *and* a valid 3-model selection will show the previous run's leaderboard until a real
schedule -> vote -> refresh has happened.

**Where**
- [frontend/index.html:1685](frontend/index.html:1685) — `nextBtn.onclick` calls `go(state.step+1)`
  unconditionally; there is no per-step completeness check anywhere in the file.
- [frontend/index.html:1378](frontend/index.html:1378) — the empty-state guard is
  `if(LIVE && (PENDING || !FUSED.length))`. `FUSED` is populated in `boot()` from the server's
  existing `fused_scores.json`, so it is non-empty and this branch never fires.
- `RENDERERS.pairs` already *detects* the problem ("Not enough models yet") but only prints it —
  it does not stop you leaving the step.

**Fix** — add a `canAdvance(step)` guard driving the `Continue` button's disabled state plus an
inline reason: registry needs >= 2 models, input needs >= 1 test case (and in BYOM mode every
(case, model) cell filled). Separately, make Results honest: compare the model id set in
`/api/results` against the current selection and fall through to the existing "No results yet"
panel when they differ. Gating alone does not fix the stale-board half.

**Why it matters** — the single most visible bug in the demo: a visitor who clicks through sees a
leaderboard that looks like their result and isn't.

---

## 2. The Judge Registry is cosmetic — VLM and Hybrid are unreachable

**What's wrong** — `updated.png` offers three voting sources. Only Human is reachable. `judge.json`
is written by the wizard and read by nothing that judges; `vlm_judge.py` takes its provider, model
and key from `.env` only; and no UI path triggers the VLM first pass or `render_views.py`. Choosing
"VLM" or "Hybrid" changes a label and nothing else.

**Where**
- [main.py:394](main.py:394) — the only runtime read of `judge.json`, and it just echoes the file
  back to the wizard in `api_registry()`.
- [5A - Subjective Pipeline/vlm_judge.py:85](5A - Subjective Pipeline/vlm_judge.py:85) — `config()`
  reads `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` from the environment; `maxTokens` from
  `judge.json` is ignored (`max_tokens` is hardcoded to 800) and so is the chosen provider.
- [frontend/index.html:689](frontend/index.html:689) — `JUDGE_TYPES` renders all three tiles as
  selectable regardless of whether any VLM credentials exist.

**Fix** — two honest options. Minimum: have the wizard read judge availability from the server and
grey out VLM/Hybrid unless `.env` is configured, so the UI stops promising what it can't do. Full:
make `vlm_judge.py` read provider/model/maxTokens from `judge.json` (env only for the secret), and
add a "run VLM first pass" action that calls `render_views.py` then `vlm_judge.py` for the current
matchups — `hybrid.py` already resolves the ballots correctly once they exist.

**Why it matters** — it is the largest box in the architecture that cannot currently be exercised,
and the wizard actively misrepresents it.

---

## 3. Concurrent judges overwrite each other, and there is no ballot audit trail

**What's wrong** — the viewer hardcodes `voter:'human'`, and `record_vote` upserts on
`(matchup_id, voter)`. Two people judging the same run therefore collapse into one ballot each,
silently replacing rather than accumulating. Re-voting also destroys the prior ballot with no
history, so there is no way to audit how a score was arrived at.

**Where**
- [5A - Subjective Pipeline/preview.html:476](5A - Subjective Pipeline/preview.html:476) — every
  ballot POSTs `voter:'human'`.
- [main.py:167](main.py:167) — `key = (record["matchup_id"], record["voter"])`, then the matching
  record is replaced in place.
- No append-only log exists anywhere; `updated.png`'s "User Management" and "Audit Logs" boxes are
  unbuilt.

**Fix** — give each judge an identity (a name prompt or a cookie-backed session id) and use it as
the `voter` key, so `aggregate.py`'s existing per-voter handling starts working and inter-rater
agreement becomes measurable. Keep the upsert for the *current* ballot but append every submission
to a separate append-only `ballot_log.jsonl` so revisions are recoverable.

**Why it matters** — the docstring on `record_vote` already claims different voters each get a
ballot; today that is not true, which quietly caps the benchmark at one judge.
