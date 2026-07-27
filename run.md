# Run Guide — Start the Server, Voting & Leaderboard

All commands run from the project root (this repo's directory).
> New here? See the **🚀 Getting Started** section in [README.md](./README.md) first.

## 1. Start the server (serves the voting + leaderboard pages)
```bash
python3 main.py --viewer-only --port 8000
```
Leave this terminal running. If you see **"address already in use"**, free the port first:
```bash
lsof -ti:8000 | xargs kill ; python3 main.py --viewer-only --port 8000
```

## 2. Open the pages in your browser
- **Voting (cast votes):**
  `http://127.0.0.1:8000/5A - Subjective Pipeline/preview.html`
- **Leaderboard:**
  `http://127.0.0.1:8000/7 - Leaderboard/leaderboard.html`

## 3. Update the leaderboard from your votes
- Easiest: click the **↻ REFRESH** button on the leaderboard page.
- Or recompute manually (reads existing votes — **no re-voting**):
```bash
python3 "5A - Subjective Pipeline/hybrid.py" && \
python3 "5A - Subjective Pipeline/aggregate.py" && \
python3 "6 - Fusion Layer/fuse.py" && \
python3 "7 - Leaderboard/build_leaderboard.py"
```
Then reload the leaderboard page. (`hybrid.py` resolves the VLM-first-pass /
human-review ballots into `final_votes.json`, which `aggregate.py` then reads.)

## First time / fresh clone (build everything once)
The leaderboard HTML and score files are generated (not committed). Build them once:
```bash
python3 "5B - Objective Pipeline/score.py"            # objective scores
python3 "5A - Subjective Pipeline/hybrid.py"          # resolve VLM/human → final_votes.json
python3 "5A - Subjective Pipeline/aggregate.py"       # subjective ELOs (needs votes)
python3 "6 - Fusion Layer/fuse.py"                          # fuse → 1300-centered score
python3 "7 - Leaderboard/build_leaderboard.py"        # render leaderboard.html
```
(Objective scores need the Layer 5 cache populated — run `python3 main.py` once if needed.)

## Notes
- Votes are saved to `5A - Subjective Pipeline/votes.json` (append-only; surviving server restarts). Don't re-vote a matchup — there's no dedup.
- The VLM judge needs an API key in `.env` (gitignored). Human voting needs no key.
