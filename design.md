# design.md

Source of truth for architecture and methodology. `CLAUDE.md` points here; keep
this current when design decisions change and log the change in `progress.md`.

---

## 1. Goal

A fantasy football draft and analysis engine that:

- derives its **own** player projections from public NFL data (no bought numbers),
- expresses every projection as a **distribution**, not a point,
- converts projections into **draft decisions** that are correct for the specific
  league format,
- works for **both** snake redraft (PPR) and **superflex / 2QB** from one engine.

The differentiator is not projection accuracy on the mean — consensus is hard to
beat there. It's the two things consensus can't give you: **calibrated
uncertainty** and a **rigorous decision layer**, plus correct superflex valuation
where the public market is soft.

## 2. Guiding principles

- **Opportunity is king.** Volume (targets, carries, routes, snaps) is far more
  stable and predictive year-over-year than efficiency or touchdowns. Project
  volume carefully; treat efficiency and TDs as regression-to-role.
- **Open and offline.** All projections come from `nflverse` via `nflreadpy`.
  Nothing projection-related depends on a live third-party API.
- **Formats are configuration.** One projection layer; the format only changes the
  decision layer (scoring weights + replacement level).
- **Baseline before Bayes.** A transparent non-Bayesian projection ships first and
  becomes the benchmark the PyMC model must beat.
- **Beat ADP or it doesn't count.** ADP (public, via Sleeper) is the real field
  consensus and the honest benchmark.

## 3. System architecture

Four layers, each consuming the one above.

```
Data ─► Projection ─► Decision ─► Interface
```

### 3.1 Data layer (`src/ingest`, `src/features`)
Pull nflverse tables via `nflreadpy`, cache to Parquet, and assemble a clean
**player-season panel** (one row per player-season) plus the weekly detail needed
to derive opportunity features. Output is deterministic and offline-reproducible.

### 3.2 Projection layer (`src/projections`)
Turns the panel into next-season projections. Two interchangeable implementations
behind one interface:
- `baseline/` — transparent regression + shares (build first).
- `bayesian/` — PyMC hierarchical model (swap-in, must beat baseline).

### 3.3 Decision layer (`src/decision`, `src/formats`)
Format-aware. Converts projections into draftable value: VOR/replacement,
statistically-real tiers, an ADP-arbitrage board, and Monte Carlo draft
simulation.

### 3.4 Interface layer (`src/api`, `frontend`)
FastAPI serves projections/decisions; Next.js renders the board. Kept thin until
Phase 1 analysis is trustworthy.

## 4. Projection methodology

Every player projection decomposes into three components, in descending order of
how much to trust them.

### 4.1 Structure: top-down, then allocate
A player's output is capped by his offense, so model the environment first.

1. **Team environment** — from `load_pbp`, project each team's plays/game, pass
   rate, and scoring context (points/drive). This sets the total volume pie.
2. **Allocate shares** — distribute team pass/rush volume to players via projected
   target share, carry share, and route participation. This is the load-bearing
   step and gets the most modeling effort.
3. **Efficiency, bottom-up** — apply yards-per-opportunity and catch rate,
   regressed toward role/positional means, shrinkage weighted by sample size.

### 4.2 The three components
1. **Volume / role** — most projectable; the core of the work.
2. **Efficiency** — YPT/YPC/YAC/catch rate; regress hard toward role means.
   Prior-year efficiency is unreliable for low-volume players.
3. **Touchdowns** — noisiest. **Never project off raw prior-year TDs.** Project
   *expected* TDs from red-zone volume and air yards, then regress. `nflreadpy`'s
   `load_ff_opportunity` (expected fantasy points given actual opportunity) is the
   backbone here.

### 4.3 Why this is a hierarchical model
The structure is naturally hierarchical and mirrors the "separate skill from
situation" approach:
- **Group level:** offensive environment (team, scheme, pace).
- **Individual level:** player role, **partially pooled** — thin-data players
  shrink toward role priors; strong-signal players pull away.
- **Aging curves:** position-specific, estimated hierarchically from history
  (RBs decline early ~27; WRs peak later). Applied as an age adjustment to the
  role/efficiency priors.

The baseline approximates this with explicit regression-to-mean and simple share
projection; the PyMC version does it properly and, critically, emits posterior
distributions that feed the uncertainty layer.

### 4.4 Rookies (the one hard case under the open constraint)
No NFL history exists, and clean college data is outside nflverse. Approach:
**draft-capital priors** — draft position is a strong predictor of opportunity —
combined with deliberately **wide uncertainty**. Fully open, weaker for first-year
players. Adding college data later would be the one sanctioned external source and
must be logged as a decision.

## 5. Uncertainty layer

Consensus gives a number; we give a distribution. This is the product.
- Baseline: attach empirical spread (historical residual variance by
  role/position) so the pipeline has intervals from day one.
- Bayesian: posterior predictive intervals per player, capturing boom/bust,
  volume risk, and rookie priors.
- **Tiers are derived from overlapping distributions:** a tier is a set of players
  whose credible intervals overlap enough that pick order barely matters — not an
  arbitrary manual break.

## 6. Formats

One projection layer; the decision layer is parameterized by a format config
(scoring weights + roster/replacement rules). Adding a format is a config file,
never scattered conditionals.

### 6.1 Redraft PPR
Standard PPR scoring, single-QB roster. Replacement levels computed for the
standard starting-lineup construction.

### 6.2 Superflex / 2QB — the biggest edge
A second startable QB is worth far more here because replacement-level QB sits well
below replacement-level flex. Public/consensus rankings default to single-QB and
systematically **under-rank QBs** for superflex. The engine gets this for free:
compute the **correct replacement baseline for a superflex roster** and elite QBs
rise to where they belong (round 1). No special modeling — just correct VOR math
against the right baseline. This is the most concrete source of edge in the whole
project.

## 7. Decision layer

- **Value over replacement (VOR):** projected points minus positional
  replacement level, where replacement level is defined by the format config.
- **Tiers:** from distribution overlap (see §5).
- **ADP arbitrage board:** rank players by the disagreement between our projection
  distribution and market ADP. This is the headline pre-draft view — the mispriced
  contracts.
- **Monte Carlo draft simulation (Phase 2):** simulate the draft from your slot to
  optimize pick strategy given positional scarcity and run risk.
- **Positional run detection (Phase 2):** live signal that a position is emptying.

## 8. Validation & benchmarking

- **Backtest protocol:** train through season N, project N+1, compare to actual
  finish. No leakage across the split.
- **Accuracy:** rank correlation and MAE vs. actual end-of-season finish.
- **Calibration:** do the 80% intervals contain the outcome ~80% of the time?
  Consensus can't even measure this; it's a core deliverable.
- **Benchmark:** **beat ADP** (public via Sleeper) at predicting finish. Every
  projection change is re-backtested against this scoreboard.

## 9. Data sources (all `nflreadpy`)

- `load_player_stats` (weekly + seasonal) — production history.
- `load_ff_opportunity` — expected points / opportunity backbone.
- `load_pbp` — red-zone touches, air yards, team pass rate, plays/game.
- `load_players` / `load_rosters` — age, position, depth, team changes.
- `load_schedules` — games and context.
- snap-count / draft-capital datasets in nflverse — confirm exact function
  signatures against nflreadpy docs when scaffolding.
- **Sleeper API (external, free, no key):** ADP + live-draft sync only. Never
  feeds projections.

## 10. Phased roadmap

**Phase 1 — Pre-draft research (current)**
Ingestion → feature panel → baseline projections → uncertainty → format-aware
VOR/tiers → ADP-arbitrage board → backtest vs ADP. Then swap in the Bayesian model
and prove it beats the baseline.

**Phase 2 — Live draft co-pilot**
Real-time board synced to a live draft (Sleeper), best-available given roster
construction, run detection, Monte Carlo pick optimization.

**Phase 3 — Season-long management**
Lineup optimizer, FAAB/waiver valuation, trade evaluator, playoff odds — all
reusing the same projection engine, weekly instead of seasonal.

## 11. Open questions / future

- Exact nflverse function names for snap counts and draft capital (verify at
  scaffold time).
- Whether/when to relax the open-data rule for college data to improve rookies.
- Frontend depth: how much of the analysis to expose interactively vs. batch
  reports.
