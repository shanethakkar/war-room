# progress.md

Living status. Update this as work happens: move tasks between Todo / Doing /
Done, and record every meaningful decision or gotcha in the log.

**Current phase:** Phase 1 — Pre-draft research
**Current focus:** DST/K + fully flexible league formats are wired in
(backend-first, user-directed), and the draft-sim was made substantially more
realistic in the process (needs-aware drafting, 15 rounds, full DST/K pools) —
which **revised the blend edge downward, honestly**: 3-way blend (ADP 0.80 /
baseline 0.10 / bayesian 0.10, DST/K pinned to market) wins **0.533 vs a 0.507
null**; earlier 0.564 was partly a sim artifact (see decisions log). UI numbers
updated to match.
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
- **(2026-07-10) ADP source = Fantasy Football Calculator, NOT Sleeper.**
  *Constraint change to CLAUDE.md #3, user-approved.* Verified Sleeper's public API
  exposes no ADP (no endpoint/field; `search_rank` is an unpopulated sentinel), and
  nflreadpy's only ADP-ish table is FantasyPros ECR (banned by #1). FFC is free,
  no-key, and has current + historical ADP for PPR and superflex (2QB) — exactly
  what the arbitrage board and the beat-ADP backtest need. ADP is a benchmark ONLY
  and never feeds projections, so the open/offline guardrail is intact. Sleeper is
  retained for Phase-2 live-draft sync (it hosts drafts).
- **(2026-07-10) The baseline BEATS ADP** (the scoreboard, design.md §8).
  Leakage-free backtest 2021–2024 on the drafted pool: our Spearman 0.515 vs ADP
  0.455 (positive every season, +0.06 mean); forward calibration coverage 0.82;
  overall accuracy Spearman 0.71 / MAE ~45. Modest but consistent, with an
  untuned transparent model — the Bayesian layer and tuning have room to extend it.
- **(2026-07-10) DRAFT-SIM: our board does NOT beat ADP in wins, despite better
  rank correlation.** New metric (`src/validation/draft_sim.py`): Monte-Carlo snake
  drafts, half the teams draft by our VOR board, half by ADP (both rank+symmetric
  noise; null verified at 0.504); rosters scored by actual optimal lineup.
  Leakage-free 2021–2024, baseline redraft: mean **win rate 0.45**, margin **-41
  pts** (2021 .49 / 2022 .51 / 2023 .47 / **2024 .33**). So the +0.06 rank-corr
  edge is illusory for drafting - rank corr weights the whole pool equally, but
  draft value lives in the top picks and positional allocation. **This is the real
  scoreboard now; improvements are gated on it, not rank corr.** (Adds `player_id`
  to the value board so it joins to actuals.)
  - **Diagnosis:** our board systematically **buries bounce-back players** (coming
    off injury/limited seasons - JT 2024 our#93/ADP#10, Kyren #98/#16, Achane
    #110/#22, Kupp) and **ascending youngsters/rookies** (Marvin Harrison Jr
    #121/#14) because the projection is anchored to last season's box score. It
    also **overrates TEs** (mean rank 72 vs ADP 101) and **underrates RBs** (97 vs
    82). The market prices in offseason info (roles, recoveries, depth charts) that
    a backward-looking model structurally lacks.
  - **Tuning does NOT fix it:** swept LOOKBACK/DECAY against the draft-sim; best
    (4, 0.9) helps 2024 (.33->.42) but overall win rate stays ~0.45. So it's ~PARITY
    with ADP most years (≈0.49 ex-2024), not an edge - and closing the gap needs
    more *signal* (depth charts, offseason moves, rookie landing spot), not tuning.
    Strategic decision pending with the user.
- **(2026-07-10) Research + decision: stop chasing "beat ADP", ship the toolkit.**
  Researched what others have found (Fantasy Football Analytics 2016; FF Insights
  2024; PFF breakout models): projection-vs-ADP is a tiny, year/position-dependent
  margin - a 2024 study found ADP *beat* projections for RBs, matching our sim.
  Real edges people cite need signal we lack (route-level *predicted* targets;
  offseason role/depth-chart/beat-reporter info). Confirmed by our own superflex
  draft-sim: **also below parity (0.40)** - superflex is QB-premium and our QB
  projections have the same backward-looking anchoring, amplified. **Confidence in
  beating ADP with open box-score data: LOW.** User chose to reframe: accept
  ~parity, build the honest decision toolkit + usable frontend. (Chasing signal
  parked as a possible future experiment.)
- **(2026-07-10) Frontend stack: Next.js (App Router) + Tailwind v3.** Honors
  CLAUDE.md's Next.js; used Tailwind v3 (stable, and the frontend-ui-dark-ts skill's
  dark tokens drop in) rather than fighting v4 config on Windows. API on a spare
  port in dev if 8000/3000 are taken; CORS allows any localhost port.
- **(2026-07-11) THE BOARD IS NOW A MARKET-ANCHORED BLEND — the first validated
  edge.** Triggered by the user challenging McBride at overall #3 (pure VOR's
  elite-TE premium) and asking whether ADP should be the spine. Tested blending
  our VOR rank with ADP rank on the draft-sim, extended to 2019–2024 (fetched
  2019/2020 FFC ADP):
  - **2-way sweep** (w = model weight): pure model 0.40, pure ADP 0.50 (null
    checks out), optimum **w=0.30 -> 0.546**; LOSO picks w=0.30 in every fold,
    **honest estimate 0.548**. Bates-Granger forecast-combination theory gives
    w*≈0.42 from rank errors — same ballpark, sim optimum lower because the
    sim penalizes our cross-position allocation.
  - **Fancier schemes all LOSE:** round-dependent weights (≈0.53–0.55),
    position-specific weights (0.52–0.53), and within-position reordering with
    ADP macro structure (0.33–0.44 — our within-position ordering, conditioned
    on ADP's structure, is *worse* than ADP's own). The blend's value is
    classic forecast combination (error averaging), not superior ordering.
    Lookback-4 model input also blends worse (0.527). Robust to draft-noise
    assumptions (0.532–0.554 at noise 3–9 picks).
  - **3-way ensemble wins: baseline 0.20 / bayesian 0.10 / ADP 0.70 -> 0.564**
    on a FRESH seed at n=500 (pre-registered confirmation; better than 2-way in
    5/6 seasons). The bayesian model ties the baseline solo but adds value as a
    diverse third forecast. (Fixed a real bug en route: bayesian fits crashed
    when no player had >=3 training seasons, e.g. projecting 2019.)
  - **Shipped:** `src/decision/blend.py` (2-way core + aux ensemble), service
    ranks boards by the blend (3-way when pymc available, else 2-way),
    `draft_sim --blend W` gates future changes on the shipped ranking. Frontend:
    default sort = Board (blended), "Arbitrage" reframed as **Tilt** (where the
    models nudge a player from market). McBride: #3 -> #20 (ADP ~29, tilt +11).
  - Superflex blend is thinner (2-way 0.51; 3-way untested) — noted, not oversold.
  - **Scoreboard now: 0.50 pure ADP -> 0.564 blended.** Win rate = share of our
    6 teams finishing top-half by actual optimal-lineup points, 12-team drafts.
- **(2026-07-11) DST/K + flexible formats shipped (backend-first, user-directed).**
  - **Data:** kicker components by distance from `player_stats_season` (fg buckets,
    PATs); DST components from pbp + schedules (sacks, INTs, fumble recoveries,
    DST/ST TDs via `td_team`, safeties, points-allowed bracket counts) — REG only
    (both sources carry playoffs; caught via games=18-21 and Broncos 65-vs-63
    sacks). Cached as `special_panel` by `features.build`.
  - **Projections:** heavily regressed per-game rates -> league mean (DST shrinks
    harder than K); PA bucket probabilities renormalized; DST plays 17. Intervals
    from the same empirical machinery (interval model v2 includes K/DST residuals).
  - **Formats are now fully config:** every scoring number is a knob (pass_td 4/6,
    PPR 0/.5/1, TE premium, K distance values, DST brackets), presets
    redraft_{ppr,half,standard} + superflex + two_qb, `customize()` for overrides,
    and `/board` accepts league-setup query params (teams, slots, scoring), cached
    per resolved config. ADP market resolves per config (`ffc_slug`); FFC `PK`/`DEF`
    map to K/DST, and FFC's city-style defense names ("LA Rams Defense") are
    canonicalized to team names for the join.
- **(2026-07-11) SIM REALISM PASS — the honest revision.** Wiring DST/K into the
  draft-sim exposed artifacts that had inflated the published blend numbers:
  1. **Empty-slot stranding:** naive ADP bots pick purely by ADP priority and can
     finish a draft without a DST/K/QB (a ~100-250-point phantom penalty). The
     needs-aware fix (both sides must fill starter slots by the endgame, like any
     human) plus full 32-DST/34-K pools (unmatched at synthetic late ADP ordered
     by OUR vor — conservative) and the correct **15 rounds** (9 starters + 6
     bench; 14 was arbitrary) produce clean nulls (0.503-0.509).
  2. On this realistic sim the old numbers do not hold: uniform blend 0.30 ->
     ~0.50; the DST/K "edge" (+12pp) was entirely artifact — measured ordering
     signal: DST spearman +0.14 (market +0.01), K +0.21 (market +0.37, better).
  3. **Recalibrated shipped config:** offense tilt only, DST/K pinned to market.
     2-way flat optimum w=0.10-0.20 (LOSO 0.512); **3-way baseline 0.10 /
     bayesian 0.10 / ADP 0.80 -> 0.533 vs 0.507 null** — the shipped default.
     Year risk explicit: 2019/2024 at or below null; 2020-2023 0.53-0.61.
  4. Every future change gates on THIS sim (`draft_sim --blend`, needs-aware,
     15 rounds, DST/K pools).
  5. **Revised human-terms numbers** (1 blend drafter vs 11 ADP, shipped 3-way
     config, 500 drafts/season): **+33 pts/season (+1.9/wk)**, avg finish 5.94
     (null 6.5), **top-half 56.7%**, top-3 31.2%, most-points 11.7% (null 8.3%).
     Per-season is the honest part: 2021-2023 strong (+76..+110, 66-74%
     top-half), 2020 flat, **2019 -19 and 2024 -64 (36% top-half)** — the edge
     is a multi-season tilt with real down-year risk, and the UI says so.
- **(2026-07-11) BACKTESTER ACCURACY AUDIT** (user asked "are we certain it's
  accurate?"). Two more inaccuracies found and FIXED in `draft_sim`:
  1. **Drafted busts now stay in the pool at 0 points** (before, an ADP-20
     player who never played vanished — erasing draft risk). Effect small in
     practice (FFC year-end ADP already drops most preseason casualties; 0-5
     busts/season) but correct.
  2. **Per-player draft noise from FFC `adp_stdev`** (clipped [2, 25]) replaces
     uniform sigma=6 — consensus firsts barely move, late fliers swing wide.
  Re-validated: null 0.497 ✓; shipped 3-way **0.531** (unchanged from 0.533) —
  the edge does not depend on either simplification.
  **Statistical honesty:** 6 seasons is the sample. Season-bootstrap 90% CI on
  the shipped win rate: **[0.476, 0.584]; P(edge > null) ~= 83%** — probably
  real, not proven. Stated on the board.
  **Known remaining limitations** (accepted or queued):
  - Season-total optimal-lineup scoring, no weekly lineups/H2H schedule/playoffs
    (weeks 15-17) — the biggest realism gap; a weekly H2H simulator from cached
    `player_stats_week` is the queued upgrade (also the only way to measure what
    the calibrated intervals are worth).
  - Actuals scored in reference PPR even for custom-scoring gates (fix when
    custom-league gating matters; components are in the panel).
  - No waivers/in-season churn (sim overweights late-round accuracy somewhat).
  - FFC ADP is mock-draft consensus (noisiest late, and the basis of our
    "market is better at K" finding).
- **(2026-07-12) Weekly H2H simulator REJECTED (user challenge upheld).**
  Simulating H2H schedules adds zero-mean matchup noise that favors no strategy
  and would only degrade the power of a 6-season sample. The valid kernel of the
  idea — **weekly lineup re-scoring + deterministic all-play expected wins**
  (captures bench/injury-cover depth value and consistency-vs-boom/bust, which
  season-total optimal lineups cannot see) — is **backlogged**, to be built only
  if a depth-related modeling question ever needs adjudicating. Validation is
  adequate and honest as-is (0.531, ~83% confidence). Next effort goes to the
  product: the draft-day core.
- **(2026-07-11) The edge, in human terms — SUPERSEDED by the realistic-sim
  numbers below the next entry.** Original naive-bot measurement (kept for the
  record; solo-user sim: ONE blend drafter vs 11 ADP drafters, 600 drafts/season,
  2019–2024, random draft slot):
  - **+51 points/season vs league average** (~3.0/week over 17 weeks; roughly
    half an extra H2H win/season under typical weekly variance).
  - **Average points-based finish 5.56 vs the 6.5 null**; top-half (playoff)
    odds **61.3% vs 50%**; top-3 **33.6% vs 25%**; most points in league
    **11.7% vs 8.3%**.
  - **Draft-capital translation:** adjacent early rounds differ by ~15–40 actual
    points, so +51 pts/season ≈ turning 2–3 mid picks into picks a round earlier.
  - **Variance honesty:** year-dependent — 2022 +130, 2021/2023 ~+81, but
    2019/2024 ~flat. A consistent tilt, not a guarantee. Solo edge (+51) ≈
    6v6 margin (+52), so the edge doesn't shrink when shared.
  - Footer on the board now states these numbers plainly.
- **(2026-07-10) The v1 Bayesian model does NOT beat the baseline; baseline stays
  default** (constraint #4). Head-to-head backtest 2021–2024 (redraft PPR):
  - Ranking: rank corr 0.708 (baseline 0.710) — a tie; beat-ADP +0.047 (baseline
    +0.060) — baseline marginally ahead.
  - MAE 42.4 (baseline 44.9) — Bayesian slightly better on magnitude.
  - **Calibration: coverage 0.65 vs baseline 0.82** — the Bayesian intervals are
    too narrow. Diagnosed cause: it models points-*per-game* and multiplies by a
    fixed games projection, so it misses **games/availability (injury) variance**,
    the dominant source of season-total spread. The baseline's empirical residuals
    capture it because they're fit on actual season totals.
  - Verdict: the scoreboard did its job — a more complex model that doesn't beat
    the transparent one does not ship as default. Bayesian is available via
    `--model bayesian`; the fix (model games variance; possibly component-level
    pooling) is queued.
  - **Modeling deps** live in the optional `bayesian` extra: `pymc` (arviz pinned
    `<1.0` — 1.x drops `InferenceData`), `nutpie` + `numba` (fast Rust NUTS).
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
- **(2026-07-10) Feature panel = one row per player-season, skill positions only**
  (QB/RB/WR/TE via `position_group`). Pure transforms in `features/panel.py`, I/O
  in `build.py`. Panel holds *current-season* actuals + context (not lagged);
  lagging/feature-engineering for prediction belongs to the projection layer.
  - Leans on nflverse-computed `target_share` / `air_yards_share` / `wopr` /
    `fantasy_points_ppr` rather than re-deriving them.
  - **Expected-TD backbone** from `ff_opportunity` (`*_touchdown_exp`), filtered to
    regular season (`week <= 18`; it carries playoff weeks 19–22). Validated: the
    expected-vs-actual TD gap flags regression candidates as intended.
  - **Snap share** joined from `snap_counts` (keyed by `pfr_player_id`) via a
    gsis<->pfr crosswalk from `players` (99.8% coverage). `offense_pct` is a 0–1
    fraction.
  - Efficiency rates use safe division (null, not 0/inf, on zero opportunity) so
    shrinkage treats them as missing.
  - Age = as of Sept 1 of the season; draft capital + `rookie_season` from
    `players`. Left joins throughout (a production row is never dropped).
  - Output cached as `feature_panel.parquet`.
- **(2026-07-10) Baseline projection = transparent, component-first, then scored**
  (`src/projections/baseline`). Confirmed approach: project component stats, then
  score per format (so superflex/custom scoring falls out correctly), not fantasy
  points directly.
  - **Volume** projected per-game, lightly shrunk toward the position mean (sticky).
  - **Efficiency** per-opportunity rates shrunk *hard* toward the positional pooled
    mean, weighted by the player's own opportunity (explicit partial pooling). `k`
    constants are in opportunity units and **TUNABLE via the backtest**.
  - **TDs** from shrunk *expected*-TD rates (ff_opportunity), never raw prior TDs.
  - **Rookies** via draft-capital priors (position x draft round, position-level
    fallback); rookie QBs get passing priors too.
  - **Scoring** lives in `src/formats/score.py` (interprets a `ScoringConfig` over
    component columns) so there's no projection->decision layer inversion.
  - **Aging curves deliberately deferred** — an unvalidated age adjustment must not
    ship before the backtest can prove it helps. First enhancement to test.
  - Validated (leakage-free, project 2024 from <=2023): overall Spearman **0.721**
    (QB .71 / RB .75 / WR .73 / TE .74; rookies .64). Honest caveat: only marginally
    beats a naive last-season carry-forward (.716) on returning players — the edge
    must come from tuning + the ADP benchmark, which is the real scoreboard.
- **(2026-07-10) Decision layer = VOR against format replacement level**
  (`src/decision`). Dedicated + FLEX + SUPERFLEX starter allocation is greedy over
  the best available; replacement level = best non-starter per position. The
  **superflex QB edge falls out for free** (validated below): same projections,
  swapping only the roster's superflex slot.
  - **Replacement convention:** best non-starter at the position (the streamer),
    so the marginal starter sits just above 0 VOR.
  - **Tiers are PROVISIONAL (gap-based)**: a new positional tier starts where the
    VOR drop to the next player exceeds `TIER_GAP`. This is a placeholder for the
    intended **distribution-overlap tiers** (design.md §5), which require the
    uncertainty layer (not built yet). Swap once intervals exist.
  - Validated on real 2025: single-QB board leads with WR/RB (first QB at overall
    #12); superflex vaults Lamar #12->#1 (VOR 84->160), Allen #14->#2. Top 4
    superflex picks are QBs.
- **(2026-07-10) Uncertainty = empirical scaled-residual quantiles, bucketed by
  projection tier** (`src/projections/uncertainty.py`). Distribution is the product
  (design.md §5).
  - Residuals collected **leakage-free** (project each past season from prior data,
    compare to actual PPR); 4,591 residuals over 2018–2025.
  - Spread modeled as quantiles of `z = residual / max(projected, FLOOR)`, split by
    position AND projection tier (top/mid/bottom third). **Bucketing was necessary,
    not optional:** a single pooled-per-position z gave elite players an absurd
    near-zero floor (Lamar 345 -> floor 3.6). Bucketed -> Lamar [122, 457]; elite
    tiers are relatively tight, fringe tiers boom/bust. In-sample coverage ~80%
    (QB 79 / RB 83 / TE 80 / WR 83). One residual model serves both formats (both
    score PPR; only the roster differs).
  - **Tiers upgraded to distribution-based** (`add_overlap_tiers`): naive interval
    overlap over-merges (season intervals are huge -> one blob). Instead: new tier
    when the anchor's median beats a player's median by > `TIER_SEP` sigma
    (sigma from the 80% width), complete-linkage on an anchor to avoid chaining.
    `TIER_SEP=0.5` (TUNABLE) gives draft-actionable elite tiers (QB t1~10, WR
    t1~14, TE t1~5). The gap-based `add_position_tiers` remains as a fallback.
  - Intervals attach in `run.py` (projection carries its distribution); the board
    consumes them.

### Gotchas

- **Windows stdout is cp1252.** Non-cp1252 chars (e.g. `→` U+2192) raise
  `UnicodeEncodeError` at runtime; even cp1252 chars (em-dash) render as `�` in the
  console. **Keep all runtime `print`/CLI output ASCII-only.** (Source files are
  UTF-8 and fine.)
- **PyMC's default NUTS is unusably slow here (no g++).** PyTensor can't compile C
  (no g++ on this Windows box), so it falls back to Python-mode gradients and a
  hierarchical fit took ~an hour. Fix: sample with **nutpie** (numba backend) -
  ~40s. `fit_model` defaults to `nuts_sampler="nutpie"`. Also cap random effects to
  established players (`MIN_PLAYER_SEASONS`) to keep the parameter space small.
- **`season` dtype is inconsistent across tables.** `ff_opportunity.season` is a
  **string** (`'2016'`) while `pbp.season` / `snap_counts.season` are **ints**.
  The feature layer must normalize the join key (cast to int) before merging.

## Open questions

- ~~Exact nflreadpy function names for snap counts + draft capital.~~ **RESOLVED
  (2026-07-10):** `load_snap_counts(seasons=...)` (PFR, since 2012) and
  `load_draft_picks(seasons=...)` (since 1980). Both cached.
- ~~ADP source (design assumed Sleeper).~~ **RESOLVED (2026-07-10):** Sleeper has
  no ADP -> moved to Fantasy Football Calculator (see decisions log). Note: FFC
  returns Error for year 2025 (gap); 2019–2024 + 2026 are available.
- **Rookie projections sit well below market ADP** (arbitrage board shows rookies
  as the biggest fades). Our draft-capital prior is conservative and ignores
  landing spot / college profile. Is that alpha (market overdrafts rookies) or a
  model weakness? The rookie-heavy pool is small in the backtest; investigate as
  part of tuning + the (future) college-data question.
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
- [ ] Aging curves: position-specific age adjustment (delta method); add + re-backtest (must beat no-aging).
- [ ] Tune shrinkage `k` constants + recency decay against the ADP backtest.
- [ ] Investigate rookie conservatism vs ADP (biggest arbitrage fades); does it help or hurt the backtest?
- [ ] Improve the Bayesian model to actually beat the baseline (it currently ties on ranking and under-covers). Priority fix: model games/availability variance in the posterior predictive (multiplying ppg by fixed games misses the injury downside -> 65% coverage). Then re-backtest.

**Doing**
- (empty)

**Done**
- [x] Project planning: methodology, architecture, constraints, roadmap (see `design.md`).
- [x] Baseline projections (`src/projections/baseline`) + per-format scoring
  (`src/formats/score.py`): recency-weighted recent form -> shrinkage regression
  (volume light, efficiency + expected-TD hard) -> component stat line -> scored
  by format. Rookies via draft-capital priors. `projections.run --season Y
  --format K` writes `projections_<model>_<fmt>_<season>.parquet` + prints top-N.
  9 network-free tests (scoring, pooled priors, shrinkage, no-leakage, rookies);
  33/33 pytest, ruff + mypy `--strict` green. Validated leakage-free (Spearman
  0.721 overall on 2024). Aging deferred to a backtested enhancement.
- [x] Decision layer (`src/decision`): replacement-level **VOR** with dedicated +
  FLEX + SUPERFLEX starter allocation (`replacement.py`), provisional gap-based
  positional tiers (`tiers.py`), and the assembled value board + CLI (`board.py`).
  `python -m src.decision.board --season Y --format K` writes
  `value_board_<fmt>_<season>.parquet` + prints top-N. 8 network-free tests
  (allocation, VOR, superflex QB edge, tier gaps, board shape); 37/37 pytest,
  ruff + mypy `--strict` green. **Superflex QB edge validated on real 2025 data.**
- [x] Draft-simulation "wins" metric (`src/validation/draft_sim.py`): Monte-Carlo
  snake drafts (our VOR board vs ADP, symmetric rank+noise, null verified ~0.50),
  rosters scored by actual optimal starting lineup; per-season win-rate + margin;
  CLI. 3 network-free tests. Revealed the baseline does NOT out-draft ADP (see
  decisions log). Added `player_id` to the value board.
- [x] DST/K + flexible formats (`src/projections/special.py`, extended
  `formats/*`, `special_panel` cache, interval model v2, `/board` league-setup
  params, FFC PK/DEF + defense-name canonicalization, DST/K in the value board
  and frontend). Sim upgraded to needs-aware drafting + 15 rounds + full DST/K
  pools; blend recalibrated on it (see decisions log). 9 new special/format
  tests + blend market-pinning test.
- [x] **Draft-day core** (frontend): draft mode with click-to-mark-taken /
  ＋-to-mark-mine, hide-drafted, undo/reset, localStorage persistence per
  (season, format) — imperative single-writer persistence (a reactive
  save-effect lost state to StrictMode double-mounts; caught by a reload test);
  my-roster panel (greedy slot fill incl. FLEX/SFLX, needs list, next-pick
  indicator with snake math); **Avail column** = P(player survives to your next
  pick) from Normal(adp, FFC per-player stdev) — `adp_stdev` added to the
  `/board` payload. All 5 format presets in the UI. Playwright-verified
  (12-pick draft simulated, persistence across reload).
- [x] Interface layer: **FastAPI `/board`** (`src/api/main.py` + `service.py`) serving
  the value board + intervals + arbitrage per season/format (cached; lazy-imports
  the pymc path), and a **polished dark Next.js draft board** (`frontend/`): season
  selector, format toggle (redraft/superflex), position filter, search, value/market/
  arbitrage sort; color-coded position tiers, floor-ceiling interval bars, VOR, ADP,
  and the arbitrage radar (targets ▲ / fades ▼) - with honest framing. Screenshot-
  verified (Playwright). 5 offline-safe API tests.
- [x] Bayesian model (`src/projections/bayesian`): hierarchical PyMC ppg model -
  partially-pooled player effects (established players only, for tractability),
  position-varying slopes, position-specific aging (age/age^2 slopes), and
  heteroscedastic Student-T; posterior-predictive intervals. Rookies reuse the
  baseline draft-capital prior. Swap-in via `--model bayesian` behind a shared
  `pipeline.scored_projection`. Sampled with **nutpie** (numba) - default PyMC NUTS
  took ~an hour on this machine (no g++ -> Python-mode gradients); nutpie fits in
  ~40s. `features` + guarded PyMC smoke tests. **Result: does NOT beat the baseline
  (see decisions log) - baseline stays default.**

- [x] Uncertainty layer (`src/projections/uncertainty.py`) + distribution-based
  tiers: leakage-free residual collection, scaled-residual quantiles bucketed by
  position x projection tier, `add_intervals`, in-sample coverage, cached
  `interval_model` (CLI: `python -m src.projections.uncertainty`). Intervals wired
  into `run.py`; overlap tiers into the board. Calibrated ~80% coverage; realistic
  elite intervals. 6 uncertainty tests + overlap-tier test; 43/43 pytest, ruff +
  mypy `--strict` green.
- [x] ADP arbitrage board + formal backtest (the scoreboard). `src/names.py`
  (name matching), `src/ingest/adp.py` (FFC ADP client + cache + CLI),
  `src/decision/arbitrage.py` (rank by projection-vs-ADP disagreement; targets +
  fades; CLI), `src/validation/backtest.py` (leakage-free accuracy + forward
  calibration + beat-ADP benchmark; CLI). **Baseline beats ADP** 2021–2024
  (0.515 vs 0.455, positive every season; coverage 0.82). 6 network-free market
  tests; 47/47 pytest, ruff + mypy `--strict` green.
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
- [x] Feature panel (`src/features`): pure transforms (`panel.py`) + orchestrating
  CLI (`build.py`, `--summary`) assembling **6,015 player-seasons x 55 features**
  for 2016–2025 → `feature_panel.parquet`. Volume/role (targets, shares, snaps),
  efficiency (safe-division rates), opportunity (expected TDs / expected pts from
  ff_opportunity), context (age, draft capital, experience). Coverage: snap_share
  99.8%, expected_pts 95.4%, age 100%. Validated against known 2024 finishes
  (Chase/Lamar/rookies correct; expected-vs-actual TD gap behaves). 8 network-free
  tests; 26/26 pytest, ruff + mypy `--strict` green.

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
- **2026-07-10** — Feature panel built and validated. `python -m src.features.build`
  produces `feature_panel.parquet` (6,015 player-seasons x 55 features, 2016–2025);
  expected-TD backbone from ff_opportunity confirmed against real finishes.
- **2026-07-10** — Baseline projection runs end-to-end. `python -m src.projections.run
  --season 2025` produces sane component projections scored per format; leakage-free
  backtest Spearman 0.721 (2024). The transparent benchmark the Bayesian model must beat.
- **2026-07-10** — Decision layer (VOR + tiers) live. `python -m src.decision.board`
  turns projections into a format-aware value board. Superflex QB edge demonstrated
  on real data: Lamar Jackson #12 (single-QB) -> #1 (superflex), same projections.
- **2026-07-10** — Uncertainty layer live. Every projection now carries a calibrated
  ~80% interval (leakage-free empirical residuals, bucketed by projection tier); the
  board's tiers come from the distributions. "The distribution IS the product."
- **2026-07-10** — **Phase-1 pipeline complete and it BEATS ADP.** ADP-arbitrage
  board (`src.decision.arbitrage`) + formal backtest (`src.validation.backtest`) on
  FFC ADP. Leakage-free 2021–2024: our rank corr 0.515 vs ADP 0.455 (every season),
  forward calibration 0.82. ADP source moved Sleeper -> FFC (user-approved).
- **2026-07-10** — Bayesian model (`src.projections.bayesian`) built as a `--model`
  swap-in (hierarchical PyMC, nutpie sampler, ~40s/fit). Head-to-head: it does NOT
  beat the baseline (ties on ranking; 0.65 vs 0.82 coverage). Baseline stays default;
  the "baseline before Bayes" discipline held.
- **2026-07-10** — Draft-simulation "wins" metric built (`src.validation.draft_sim`).
  Big finding: rank correlation was misleading — in simulated drafts our board does
  NOT beat ADP (win rate 0.45 vs 0.50 null, bad 2024). The real scoreboard; all
  future projection changes gated on it.
- **2026-07-10** — Reframed to an honest decision toolkit and shipped the interface:
  FastAPI `/board` + a polished dark Next.js draft board (calibrated intervals,
  tiers, format-aware VOR, ADP arbitrage radar). Research + the superflex draft-sim
  (0.40) confirmed beating ADP with open data is low-probability; value is the
  transparent, calibrated toolkit, not a market-beating claim.
- **2026-07-11** — **First validated edge shipped: the market-anchored blend.**
  User challenged McBride #3 + questioned pure-model ranking vs ADP; analysis
  (6-season sweep, LOSO, Bates-Granger, variant tests, fresh-seed confirmation)
  landed on ADP 0.70 / baseline 0.20 / bayesian 0.10 -> draft-sim win rate
  **0.564 vs 0.50** pure ADP. Board reranked by the blend; "Arbitrage" reframed
  as model Tilt; `draft_sim --blend` gates future changes. McBride #3 -> #20.
- **2026-07-12** — **Draft-day core shipped.** The board is now a live draft
  tool: draft mode (mark taken/mine, undo, persistent across refresh), my-roster
  panel with needs, and per-player "survives to my next pick" odds from FFC's
  pick-variance data. Weekly H2H sim rejected as metric noise (user challenge
  upheld); weekly all-play re-scoring backlogged.
- **2026-07-11** — **DST/K + flexible league formats shipped; sim made realistic;
  blend numbers honestly revised.** Every scoring/roster rule is now a knob
  (pass_td, PPR, TE premium, K distances, DST brackets, team count, 2QB), the
  board covers all startable positions, and the upgraded sim (needs-aware
  drafting, 15 rounds, full DST/K pools) exposed artifacts in the earlier
  numbers: revised edge is **0.533 vs 0.507 null** (3-way: ADP 0.80 / baseline
  0.10 / bayesian 0.10; DST/K pinned to market - their ordering signal measured
  as noise). Footer updated to the revised, smaller, honest numbers.
