# progress.md

Living status. Update this as work happens: move tasks between Todo / Doing /
Done, and record every meaningful decision or gotcha in the log.

**Current phase:** Phase 1 — Pre-draft research
**Current focus:** Ingestion layer done — 10 nflverse tables cached to Parquet
(2016–2025, ~142 MB, offline-reproducible). Next up is the feature panel
(`src/features`).
**Last updated:** 2026-07-10

---

## Decisions log

Locked decisions from planning. Add new ones here with a date; don't silently
reverse these.

- **Open source, no proprietary data.** Projections derive only from nflverse.
  No FantasyPros/PFF/ESPN/gated sources.
- **`nflreadpy`, not `nfl_data_py`.** Latter deprecated + archived Sept 2025.
  nflreadpy returns Polars.
- **Derive own baseline projections** rather than ingesting consensus. Keeps the
  project fully open and unblocked by external APIs.
- **Sleeper API** is the only external runtime dependency: ADP + live draft only,
  never projections; free, no key.
- **Baseline before Bayesian.** Ship a transparent projection + full pipeline +
  ADP backtest first; PyMC model is a swap-in that must beat it.
- **Formats are config.** Redraft PPR + superflex/2QB share one projection layer;
  only the decision layer's scoring + replacement level differ.
- **Scoreboard = beat ADP** on accuracy (rank corr, MAE) AND calibration
  (interval coverage).
- **Superflex QB valuation** identified as the biggest concrete edge: correct
  replacement baseline, no special modeling.
- **Rookies:** draft-capital priors + wide uncertainty under the open constraint.
- **(2026-07-10) Project named "War Room."** Threaded into `README.md` and docs.
- **(2026-07-10) Training window = 2016–present.** Modern pass-heavy era; keeps
  shares and aging curves on-regime without old rule/scheme eras. Encoded as
  `src.config.DATA_START_SEASON`.
- **(2026-07-10) Toolchain:** `uv` + Python 3.12; `ruff` (lint+format), `mypy`
  `--strict` on `src`, `pytest`. PyMC/arviz isolated in the optional `bayesian`
  extra so the baseline stack stays light and never depends on it.
- **(2026-07-10) GitHub:** `github.com/shanethakkar/war-room`; push after each
  milestone.
- **(2026-07-10) Toolchain pinned to Python 3.12.** `requires-python >=3.12`,
  ruff `py312`, mypy `python_version 3.12`. numpy 2.5 stubs use 3.12-only `type`
  syntax, so straddling 3.11 broke mypy; we develop/run on 3.12 anyway.
- **(2026-07-10) Ingestion = a table registry.** `src/ingest/sources.py` maps a
  stable cache name → nflreadpy loader (seasoned / all-history / reference). Cache
  I/O isolated in `cache.py`; `refresh.py` orchestrates + is the CLI. Per-table
  failures are reported, not fatal. Adding a table = one registry entry.
- **(2026-07-10) End of window via `nflreadpy.get_current_season()`** (date-based,
  no network), so the window auto-extends each season. Resolved to 2016–2025 now.

### Gotchas

- **Windows stdout is cp1252.** Non-cp1252 chars (e.g. `→` U+2192) raise
  `UnicodeEncodeError` at runtime; even cp1252 chars (em-dash) render as `�` in the
  console. **Keep all runtime `print`/CLI output ASCII-only.** (Source files are
  UTF-8 and fine.)
- **`season` dtype is inconsistent across tables.** `ff_opportunity.season` is a
  **string** (`'2016'`) while `pbp.season` / `snap_counts.season` are **ints**.
  The feature layer must normalize the join key (cast to int) before merging.

## Open questions

- ~~Exact nflreadpy function names for snap counts + draft capital.~~ **RESOLVED
  (2026-07-10):** `load_snap_counts(seasons=...)` (PFR, since 2012) and
  `load_draft_picks(seasons=...)` (since 1980). Both cached.
- Route participation source: `load_snap_counts` gives snap share but not routes.
  Routes/route-participation (needed for target-share modeling, design.md §4.1)
  likely come from `load_participation` / `load_nextgen_stats` — evaluate coverage
  and recency when the feature layer needs them (not yet cached).
- When (if ever) to relax the open-data rule for college data to improve rookies.
- How thick the Next.js frontend should be in Phase 1 vs. batch reports.

---

## Task board

### Phase 1 — Pre-draft research

**Todo**
- [ ] Feature panel (`src/features`): player-season panel with volume / efficiency / opportunity features from weekly + pbp + ff_opportunity.
- [ ] Format configs (`src/formats`): redraft PPR and superflex/2QB (scoring + roster/replacement rules).
- [ ] Baseline projections (`src/projections/baseline`): top-down team environment → share allocation → regressed efficiency → expected-TD-based scoring.
- [ ] Uncertainty (baseline): empirical residual spread by role/position → intervals.
- [ ] Decision layer (`src/decision`): VOR against format replacement level; distribution-overlap tiers.
- [ ] ADP arbitrage board: pull Sleeper ADP; rank by projection-vs-ADP disagreement.
- [ ] Validation (`src/validation`): train-through-N / project-N+1 backtest; rank corr + MAE + calibration; ADP benchmark.
- [ ] Minimal API (`src/api`) + thin Next.js view for the board.
- [ ] Bayesian projections (`src/projections/bayesian`): PyMC hierarchical (offense group level + partially-pooled player role + aging curves); posterior predictive intervals.
- [ ] Prove Bayesian beats baseline on the scoreboard; make it the default only if it wins.

**Doing**
- (empty)

**Done**
- [x] Project planning: methodology, architecture, constraints, roadmap (see `design.md`).
- [x] `uv` project scaffold: `pyproject.toml` (light baseline deps + optional
  `bayesian` extra), `ruff`/`mypy`/`pytest` config, `.gitignore`, `README.md`,
  and the full `src/` skeleton per `design.md` §3 — layer packages with
  docstrings, runnable CLI entrypoints (`ingest.refresh`, `features.build`,
  `projections.run`, `validation.backtest`) as honest not-yet-implemented stubs,
  a real FastAPI app (`/health`, `/formats`), format configs (redraft PPR +
  superflex registry), and a smoke test. All gates green: ruff, mypy `--strict`,
  10/10 pytest.
- [x] Ingestion layer (`src/ingest`): verified nflreadpy API (0.1.5), built a
  table registry (`sources.py`) + Parquet cache I/O (`cache.py`) + orchestrating
  CLI (`refresh.py`) with `--start/--end-season`, `--only`, `--list` and per-table
  error handling. Pulled 10 tables for 2016–2025 into `data/cache` (~142 MB):
  pbp (484k rows), player_stats week+season, ff_opportunity, snap_counts, rosters,
  schedules, players, teams, draft_picks. 11 network-free tests; 21/21 pytest,
  ruff + mypy green. `data/` gitignored (cache reproducible from `refresh`).

### Phase 2 — Live draft co-pilot (not started)
- [ ] Live Sleeper draft sync + real-time board.
- [ ] Best-available given current roster construction.
- [ ] Positional run detection.
- [ ] Monte Carlo draft simulation / pick optimization from your slot.

### Phase 3 — Season-long management (not started)
- [ ] Weekly projections from the same engine.
- [ ] Lineup optimizer.
- [ ] FAAB / waiver valuation.
- [ ] Trade evaluator.
- [ ] Playoff odds.

---

## Changelog

- **2026-07-10** — Project scaffold landed. uv project + `src/` skeleton +
  tooling (ruff/mypy/pytest) all green; FastAPI app and format registry live.
  Repo initialized and pushed to `github.com/shanethakkar/war-room`.
- **2026-07-10** — Ingestion layer runs, cache populated. 10 nflverse tables
  cached for 2016–2025 (~142 MB) via `python -m src.ingest.refresh`; snap-count
  and draft-capital loader names resolved. Toolchain pinned to Python 3.12.
