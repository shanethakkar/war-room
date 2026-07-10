# CLAUDE.md

> Read this first, every session. For architecture and methodology see `design.md`.
> For current status and the task board see `progress.md`. **Update `progress.md`
> whenever you complete or start a task.**

## What this is

An open-source, full-stack fantasy football draft and analysis engine. It derives
its own player projections from public NFL data, wraps them in calibrated
uncertainty, and turns them into draft decisions. Three phases: (1) pre-draft
research, (2) live in-draft co-pilot, (3) season-long management. Currently in
Phase 1.

## Non-negotiable constraints

These are guardrails. Do not violate them without an explicit decision logged in
`progress.md`.

1. **Fully open source. No paid or proprietary data, ever.** All projections
   derive from the free `nflverse` ecosystem. Do not add FantasyPros, PFF, ESPN
   projections, or any gated/scraped/keyed projection source. If a task seems to
   need one, stop and flag it instead of adding it.
2. **Use `nflreadpy`, NOT `nfl_data_py`.** `nfl_data_py` was deprecated and
   archived in Sept 2025. Any tutorial referencing it is stale. `nflreadpy`
   returns **Polars** DataFrames, not pandas — write Polars-native code and only
   `.to_pandas()` at a boundary that genuinely needs it (e.g. PyMC input).
3. **The only external runtime API is Sleeper** (free, no key), used ONLY for ADP
   and live-draft sync in Phases 1–2. It never feeds projections. Everything
   projection-related must work fully offline from cached nflverse data.
4. **Ship the transparent baseline before the Bayesian model.** The pipeline must
   run end-to-end and backtest against ADP with a simple, non-Bayesian projection
   first. The PyMC layer is a swap-in that must prove it beats that baseline.

## Tech stack

- **Language:** Python-first. TypeScript/Next.js only for the frontend.
- **Env/deps:** `uv` (nflreadpy targets it; use it for the whole backend).
- **Data:** `nflreadpy` (Polars) → local Parquet cache.
- **Modeling:** Polars/NumPy for the baseline; `PyMC` for the hierarchical model.
- **Backend API:** FastAPI.
- **Frontend:** Next.js (React), dark theme. Kept thin until Phase 1 analysis is solid.

## Repo layout (target)

```
.
├── CLAUDE.md              # this file
├── design.md              # architecture + methodology (source of truth)
├── progress.md            # living status + task board
├── pyproject.toml         # uv-managed
├── data/                  # local Parquet cache (gitignored)
├── src/
│   ├── ingest/            # nflreadpy loaders → clean player-season panel
│   ├── features/          # volume/efficiency/opportunity feature builds
│   ├── projections/
│   │   ├── baseline/      # transparent, non-Bayesian (build FIRST)
│   │   └── bayesian/      # PyMC hierarchical (swap-in)
│   ├── decision/          # VOR, tiers, ADP arbitrage, draft sim
│   ├── formats/           # scoring + roster configs (redraft PPR, superflex)
│   ├── validation/        # backtest, calibration, ADP benchmark
│   └── api/               # FastAPI app
├── frontend/              # Next.js
└── tests/
```

## Commands

Keep these working; update this section if they change.

```bash
# setup
uv sync

# pull / refresh nflverse data into local cache
uv run python -m src.ingest.refresh

# build the player-season feature panel
uv run python -m src.features.build

# run projections (baseline by default; --model bayesian to swap)
uv run python -m src.projections.run --season 2025

# backtest + calibration + ADP benchmark
uv run python -m src.validation.backtest --through 2024

# tests
uv run pytest

# serve API
uv run uvicorn src.api.main:app --reload

# frontend
cd frontend && npm run dev
```

## Conventions

- Type hints everywhere; run `ruff` for lint/format and `mypy` on `src`.
- Prefer pure functions from panel → panel; keep I/O at the edges.
- No point estimate without an uncertainty interval attached once the Bayesian
  layer lands. The distribution IS the product.
- Every projection change must be re-backtested. **Beating ADP is the scoreboard**
  — accuracy (rank correlation, MAE) and calibration (interval coverage) both.
- Formats are config, not code branches. Adding a format = a config file in
  `src/formats/`, not `if superflex:` scattered around.
- Write disciplined docstrings on any modeling function explaining the *why*
  (this repo is meant to be publicly readable).

## Working style

- Before any architectural or modeling work, re-read the relevant section of
  `design.md`. It is the source of truth for methodology.
- After finishing a unit of work, move the task to Done in `progress.md` and note
  any decision or gotcha in the Decisions Log.
- When unsure about scope or a design tradeoff, ask before building.
