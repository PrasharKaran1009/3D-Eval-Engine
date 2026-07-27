# 3D Model Evaluation Leaderboard

A comprehensive, state-of-the-art evaluation pipeline for benchmarking 3D generative models. The system combines **Subjective (Human/LLM in the loop)** and **Objective (Automated Analysis)** metrics, aggregating them through a **Multi-Criteria Decision Analysis (MCDA) Fusion Layer** to produce a definitive, multi-dimensional leaderboard.

**🔗 Live demo (no install):** [`docs/` on GitHub Pages](https://YOUR-USERNAME.github.io/YOUR-REPO/) — the example run's leaderboard, charts, and an interactive **blind 3D pair voter** (rotate / zoom / wireframe), served fully static. *(Enable Pages → Settings → Pages → Source: `main` / `/docs`.)*

---

## 🏗️ Architecture Overview

The system processes 3D model generation tasks through an 8-stage pipeline, maximizing efficiency via lazy generation and caching while ensuring rigorous evaluation.

![3D Model Evaluation Leaderboard Architecture](./updated.png)

### Build status — BYOM complete, BYOA is an open socket
The **BYOM** path (bring your own meshes) runs end-to-end and the Setup Studio
wizard drives it live:

- **Registry** persists to `models.json` / `prompts.json` / `judge.json`
  (`POST /api/registry`) — model ids are **stable by name**, so re-saving never
  re-attaches old votes/cache to the wrong model.
- **Input** uploads land canonically in `TEST CASES/<TC> GLB/` as
  `TC<nnn>_<Name>.<ext>` (`POST /api/upload`), which is why the strict Layer 2
  gate can stay strict.
- **Scheduler** runs the real L2→L3→L4→L5 pipeline (`POST /api/schedule`).
- **Fusion** exposes the full MCDA menu — hard constraints + optional Pareto +
  **SAW / TOPSIS / PROMETHEE / VIKOR / ELECTRE** — configurable from the wizard
  (`GET`/`POST /api/settings`, stored in `6 - Fusion Layer/settings.json`).
- **Judging** supports Human, VLM, and Hybrid; `hybrid.py` resolves the
  VLM-first-pass / human-review ballots into `final_votes.json`, which
  `aggregate.py` now consumes. Blind pairs are **A/B side-randomized**
  (deterministic per matchup) and the **prompt + reference images are shown to
  the judge** so `prompt_alignment` is actually judgeable.

**BYOA** (bring your own API) is the deliberate open circuit. Register a model
with `source: "BYOA"` and the scheduler routes that cell through
`4 - Generation Prep Scheduler/providers.py::BYOAGenerator` — a documented stub
today. Implement its `resolve()` (the wiring points are spelled out inline) and
the box energizes with **no scheduler changes**. Everything else — pairing,
caching, voting, fusion, leaderboard — already flows around it.

---

## 🎬 Live demo & screenshots

The [`docs/`](./docs) folder is a **self-contained static export** of the example
run — no Python, no backend — so it deploys free to GitHub Pages:

| Page | What it shows |
| :--- | :--- |
| [`docs/index.html`](./docs/index.html) | Final leaderboard (podium + ranked table) and four charts (fused-score bars, per-parameter radar, cost-vs-quality scatter, objective-dimension breakdown), rendered from precomputed JSON. |
| [`docs/viewer.html`](./docs/viewer.html) | The **blind pairwise voter** — two 3D models side-by-side (rotate / zoom / textured·shaded·wireframe), the generation prompt + reference images, and the ballot. Ships the *Mecha Dragon* test case (all 7 models, 21 matchups). |

The 3D meshes are Draco-free **quantized + WebP-compressed** GLBs (the original
~1 GB of assets → ~65 MB), so the plain `three.js` GLTFLoader renders them with
no extra decoder. Leaderboard and charts aggregate **all five** test cases; the
in-browser voter ships one test case to keep the demo repo light — the full
five-case voter runs in the local app.

> **Add screenshots/GIF here.** For the résumé, a short screen-capture of the
> wizard → blind vote → leaderboard (or a couple of PNGs) reads better than any
> link. Drop them in `docs/` and embed: `![demo](docs/screenshot.png)`.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+**
- Install the Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  This includes **pymeshlab**, which is **required** — `cache.py` and
  `metrics_extractor.py` import it at load time, so Layers 5/6B will not run
  without it.
- *Optional:* the VLM/hybrid judge's auto-render step needs Playwright —
  `pip install playwright && playwright install chromium`. Human voting and the
  whole objective pipeline run fine without it.

### Run the whole thing (one command)
From the project root (this repo's directory):
```bash
python3 main.py
```
This runs the full pipeline — **Layer 2 input validation → Layer 3 matchups →
Layer 4/5 scheduling + metric caching** — then opens the **Setup Studio** in your
browser, starting on the **Welcome** page (pick BYOM + the judge).

### The web UIs
While the server is running, open:
| Page | URL |
| :--- | :--- |
| **Entry — Setup Studio** (starts on Welcome) | `http://127.0.0.1:8000/` |
| **Voting** (blind A/B pair viewer) | `http://127.0.0.1:8000/5A - Subjective Pipeline/preview.html` |
| **Leaderboard** | `http://127.0.0.1:8000/7 - Leaderboard/leaderboard.html` |

The root URL redirects to the wizard; every step reads live data from the running
server (registry, validation, pairs, results).

### Useful commands
| Command | What it does |
| :--- | :--- |
| `python3 main.py` | Validate → run pipeline → launch viewer |
| `python3 main.py --no-viewer` | Run the pipeline only (no browser) |
| `python3 main.py --viewer-only` | Just serve the UIs (uses existing cache) |
| `python3 main.py --skip-validation` | Bypass the Layer 2 input-validation gate |
| `python3 "2 - Input Layer/byom_validator.py"` | Run BYOM input validation on its own |
| `python3 "2 - Input Layer/byom_validator.py" --json` | Same, as machine-readable JSON |

If port 8000 is busy: `lsof -ti:8000 | xargs kill`.

### Deploy it (Docker)
The repo ships a `Dockerfile` that builds a headless, self-contained BYOM
deployment (objective pipeline + blind voting + fused leaderboard):
```bash
docker build -t 3d-eval .
docker run -p 8000:8000 3d-eval        # open http://localhost:8000
```
The mesh cache is baked at build time, so the container starts instantly and the
blind viewer has its GLBs. On a host (Render / Railway / Fly / a VM) it reads
`HOST` and `PORT` from the environment and binds `0.0.0.0` automatically; run
`python3 main.py --viewer-only --host 0.0.0.0 --no-open` if you deploy without
Docker. Note the image is large (it contains `TEST CASES/` + the extracted
cache); to slim it, drop the `RUN python3 main.py --no-viewer` line and mount
`TEST CASES/` at runtime instead.

### Deploy the demo for free (GitHub Pages)
The [`docs/`](./docs) export needs **no server** — it's the reliable, zero-cost
way to put a clickable demo on a résumé:
1. Push the repo (the compressed demo meshes in `docs/models/` are force-tracked
   past the global `*.glb` ignore — see `.gitignore`).
2. **Settings → Pages → Source: `main` branch, `/docs` folder.**
3. Share `https://<user>.github.io/<repo>/`.

The live pipeline (uploads, generation, big-mesh objective analysis) needs a real
host with RAM + a disk — see the Docker section above. The static demo is
view/interact only; it doesn't run the pipeline.

### Runs & the example benchmark
Each benchmark is a **run** — its own votes + scores + cache for one model set.
The bundled demo (105 hand-cast ballots) is frozen once as the permanent,
restorable **`runs/example/`**. When someone registers a *different* model set
(the fingerprint of names+sources changes), the app **forks**: it archives the
current run to `runs/archive_<ts>/`, then resets votes + scores + mesh cache so a
brand-new, empty `votes.json` forms for them — their voting never touches the
example, and re-saving the *same* set (editing metadata) preserves everything.
The top-bar **↺ Load example** button (or `POST /api/load-example`) restores the
demo run at any time.

### Update the leaderboard after voting
Votes are saved to `5A - Subjective Pipeline/votes.json`. To recompute the
leaderboard from existing votes, click **↻ REFRESH** on the leaderboard page, or:
```bash
python3 "5A - Subjective Pipeline/hybrid.py" && \
python3 "5A - Subjective Pipeline/aggregate.py" && \
python3 "6 - Fusion Layer/fuse.py" && \
python3 "7 - Leaderboard/build_leaderboard.py"
```

> The benchmark test cases, prompts, and models are defined in the **Registry**
> (`1 - Registry Layer/models.json` and `prompts.json`). Model assets live
> in `TEST CASES/TC00x GLB/` named `TC00x_ModelName.glb`; the Layer 2 validator
> enforces that naming + full coverage before any run.

---

## 🔍 Detailed Pipeline Description

### 1. Input Layer
*   **$N$ Models ($M_1$ to $M_N$):** Generative models evaluated under the exact same prompt and task constraints.
*   **Prompt Manager:** Standardizes the user prompts, ensures consistent distribution to all models, and generates a unique **Prompt Hash** used for downstream caching.

### 2. Round Robin Pair Generator
*   Generates all unique pairwise comparisons ($nC_2$).
*   For any number of models $n$, this results in $\frac{n(n-1)}{2}$ unique pairs (e.g., $(M_1, M_2)$, $(M_1, M_3)$, ..., $(M_{n-1}, M_n)$).

### 3. Generation Scheduler (Lazy Generation + Caching)
To avoid unnecessary computation and API costs, generations are scheduled on-demand:
*   **Logic:** For each pair, the scheduler checks if model outputs already exist in the cache. If not, it triggers generation.
*   **Efficiency Benefit (Lazy Generation):**
    *   Since outputs are cached after generation, each of the $n$ models is generated exactly once.
    *   **Result:** Only **$n$ total generations** are executed across the entire run instead of the redundant $nC_2$ generations.

### 4. Output Cache / Storage
Stores model assets and raw generation metadata:
*   **3D Files:** `GLB` / `OBJ` files.
*   **Previews:** High-quality preview images/renders.
*   **Metadata:** Mesh statistics (vertices, faces), Generation Time, Credits/Cost Used, Prompt Hash, Model Version, and custom system metadata.

---

### 5A. Subjective Pipeline (Human/LLM-in-the-Loop)
Evaluates qualitative characteristics through pairwise comparison:
*   **Pair Viewer:** Built on `Three.js` or `Babylon.js` for side-by-side 3D inspection (supporting rotate, zoom, and wireframe views).
*   **Judges:** Votes cast by Human Judges or LLM-as-a-Judge agents.
*   **Voting Parameters:**
    *   👁️ *Visual Appeal*
    *   ✨ *Surface Quality*
    *   📏 *Proportions*
    *   🔍 *Detail Level*
    *   🎨 *Style Accuracy*
*   **Bradley-Terry MLE:** Compiles pairwise comparison matrices per parameter and fits a Bradley-Terry Maximum Likelihood Estimation model to compute continuous **ELO Scores** for each parameter per model.

---

### 5B. Objective Pipeline (Automated)
Extracts quantitative, deterministic features directly from the 3D meshes:
*   **Automated Analysis Engine:**
    *   *Mesh Integrity:* Detects non-manifold geometry, self-intersections, and broken faces.
    *   *Watertightness:* Identifies holes and boundary edges.
    *   *UV Quality:* Evaluates UV overlap and texture stretching.
    *   *Symmetry Score:* Quantifies mesh reflectional symmetry.
    *   *Polygon Count:* Measures model optimization and efficiency.
*   **Feature Vectors & Normalization:** Converts raw technical metrics into a normalized Rubric Score Matrix scaled from `0` to `100`.

---

### 6. Fusion Layer (MCDA)
Consolidates subjective rankings and objective metrics using Multi-Criteria Decision Analysis (MCDA):
1.  **Step 1: Hard Constraints:** Rejects broken meshes and applies heavy penalties for non-watertight geometry or invalid topologies.
2.  **Step 2: Pareto Filter (Optional):** Screens out dominated models (models that are worse than another model across all metrics) to isolate the Pareto frontier.
3.  **Step 3: MCDA Aggregation:** Combines metrics using configurable MCDA algorithms:
    *   **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)
    *   **PROMETHEE** (Preference Ranking Organization Method for Enrichment Evaluations)
    *   **VIKOR** (Multicriteria Optimization and Compromise Solution)
    *   **ELECTRE** (Elimination and Choice Expressing Reality, with veto capabilities)
4.  **Step 4: Final Composite Score:** Outputs a single, unified score per model representing its overall quality.

---

## 🏆 Leaderboard Output & Reporting

The leaderboard outputs a unified view of all evaluated models, sorted by their final composite score:

| Rank | Model | Subjective ELO (Avg) | Objective Score (MCDA) | Final Score | Mesh Integrity | UV Quality | Poly Count | Watertight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Model-07 | 1840 | 94 | **92.6** | 98 | 92 | 78k | Yes |
| **2** | Model-02 | 1790 | 88 | **89.1** | 91 | 88 | 82k | Yes |
| **3** | Model-11 | 1750 | 82 | **83.4** | 87 | 79 | 70k | Yes |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **13** | Model-04 | 1420 | 51 | **55.2** | 60 | 55 | 40k | No (Penalty) |

### Reporting Features
*   📊 **Visualizations:** Radar charts for per-parameter strengths, cost vs. quality scatter plots, and score breakdown bar charts.
*   📤 **Data Export:** Direct exports to Google Sheets, CSV, or formatted PDF reports.
