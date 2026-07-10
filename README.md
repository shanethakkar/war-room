# War Room

An open-source, full-stack **fantasy football draft & analysis engine**. It
derives its *own* player projections from free public NFL data, wraps every
projection in **calibrated uncertainty** (a distribution, not a point estimate),
and turns those distributions into **format-aware draft decisions** — for both
snake redraft PPR and superflex / 2QB, from one engine.

> **The edge isn't mean accuracy** — consensus is hard to beat there. It's the two
> things consensus can't give you: honest, *calibrated* uncertainty and a rigorous
> decision layer — plus correct superflex QB valuation, where the public market is
> systematically soft.

## Principles (the non-negotiables)

- **Fully open. No paid or proprietary data, ever.** All projections derive from
  the free [`nflverse`](https://github.com/nflverse) ecosystem via
  [`nflreadpy`](https://github.com/nflverse/nflreadpy) (Polars-native; **not** the
  deprecated `nfl_data_py`).
- **The only runtime API is Sleeper** (free, no key) — used *only* for ADP and
  live-draft sync. It never feeds projections; everything projection-related runs
  fully offline from a local Parquet cache.
- **Baseline before Bayes.** A transparent, non-Bayesian projection ships first and
  becomes the benchmark the PyMC hierarchical model must beat.
- **Beating ADP is the scoreboard** — accuracy (rank correlation, MAE) *and*
  calibration (interval coverage).

## Architecture

Four layers, each consuming the one above:

```
Data (nflreadpy → Parquet)  →  Projection (baseline → Bayesian swap-in)
  →  Decision (VOR, tiers, ADP arbitrage)  →  Interface (FastAPI + Next.js)
```

See [`design.md`](design.md) for the full methodology and [`progress.md`](progress.md)
for current status and the task board.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). Python is managed by uv.

```bash
uv sync                                             # install (baseline stack)
uv sync --extra bayesian                            # + PyMC (hierarchical model)

uv run python -m src.ingest.refresh                 # pull nflverse → Parquet cache
uv run python -m src.features.build                 # build player-season panel
uv run python -m src.projections.run --season 2025  # projections (--model bayesian to swap)
uv run python -m src.validation.backtest --through 2024   # backtest vs ADP

uv run pytest                                       # tests
uv run ruff check . && uv run mypy                  # lint + types
uv run uvicorn src.api.main:app --reload            # serve API
```

## Status

**Phase 1 — pre-draft research.** Project scaffold is in place; the data ingestion
layer is next. Roadmap: ingestion → feature panel → baseline projections →
uncertainty → format-aware VOR/tiers → ADP-arbitrage board → backtest, then swap in
the Bayesian model and prove it wins. Phases 2 (live draft co-pilot) and 3
(season-long management) follow.

## License

MIT
