# progress.md

Living status. Update this as work happens: move tasks between Todo / Doing /
Done, and record every meaningful decision or gotcha in the log.

**Current phase:** Phase 1 — Pre-draft research
**Current focus:** Scaffold is in place (uv project, `src/` skeleton, tooling, CI
gates green). Next up is the data ingestion layer (`src/ingest`).
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

## Open questions

- Exact nflreadpy function names for snap counts + draft capital — verify against
  docs during ingestion scaffold.
- When (if ever) to relax the open-data rule for college data to improve rookies.
- How thick the Next.js frontend should be in Phase 1 vs. batch reports.

---

## Task board

### Phase 1 — Pre-draft research

**Todo**
- [ ] Ingestion (`src/ingest`): nflreadpy loaders → Parquet cache; `refresh` entrypoint. Verify snap-count + draft-capital function names.
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
- _(add dated entries as milestones land, e.g. "2026-07-11 — ingestion layer runs, cache populated")_
