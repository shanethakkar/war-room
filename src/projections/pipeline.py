"""Model-agnostic projection pipeline.

Produces a scored projection with an 80% interval for a season, routing to either
model, and (for the baseline path) appending K/DST from the special panel so the
board covers every startable position. All paths return the same display schema
(identity + ``projected_points`` + ``points_low``/``points_median``/
``points_high``), so the decision layer and the backtest never branch on model.

- **baseline**: component projection -> per-format scoring -> empirical intervals.
- **bayesian**: hierarchical posterior predictive (points + intervals); K/DST
  still come from the (baseline) special projections - the MCMC model is
  offense-only.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from src.formats import FormatConfig, get_format
from src.formats.score import score_components, score_special
from src.ingest.cache import read_table
from src.projections import MODELS
from src.projections.baseline.project import project_season
from src.projections.special import project_special
from src.projections.uncertainty import (
    _all_residuals,
    add_intervals,
    fit_interval_model,
    load_or_fit_interval_model,
)

# Shared display schema every scored projection exposes downstream.
DISPLAY_COLS: tuple[str, ...] = (
    "player_id",
    "player_name",
    "position",
    "position_group",
    "team",
    "season",
    "is_rookie",
    "projected_games",
    "projected_points",
    "points_low",
    "points_median",
    "points_high",
)


def _resolve_format(fmt: FormatConfig | str) -> FormatConfig:
    return fmt if isinstance(fmt, FormatConfig) else get_format(fmt)


def _special_scored(
    season: int, fmt: FormatConfig, interval_model: pl.DataFrame
) -> pl.DataFrame | None:
    """K/DST scored projection + intervals, or None if the panel isn't cached."""
    try:
        special_panel = read_table("special_panel")
    except FileNotFoundError:
        return None
    scored = score_special(project_special(special_panel, season), fmt.scoring)
    return add_intervals(scored, interval_model).select(DISPLAY_COLS)


def scored_projection(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    season: int,
    *,
    model: str = "baseline",
    fmt_key: FormatConfig | str = "redraft_ppr",
    interval_residual_seasons: Sequence[int] | None = None,
    bayes_kwargs: dict[str, int] | None = None,
    include_special: bool = True,
) -> pl.DataFrame:
    """Scored projection + intervals for ``season`` from the chosen model.

    ``fmt_key`` accepts a preset key or a full (possibly customized)
    ``FormatConfig``. ``interval_residual_seasons`` (baseline only) pins the
    empirical interval model to specific seasons for a leakage-free backtest;
    ``None`` uses the cached model. ``include_special`` appends K/DST rows when
    the special panel is cached.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}.")
    fmt = _resolve_format(fmt_key)

    if model == "bayesian":
        # Lazy import: keeps the baseline/API path free of the optional pymc extra.
        from src.projections.bayesian.project import bayesian_project_season

        offense = (
            bayesian_project_season(panel, players, season, **(bayes_kwargs or {}))
            .with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
            .select(DISPLAY_COLS)
        )
        if not include_special:
            return offense
        interval_model = load_or_fit_interval_model(panel, players)
        special = _special_scored(season, fmt, interval_model)
        return offense if special is None else pl.concat([offense, special])

    projection = score_components(project_season(panel, players, season), fmt.scoring)
    if interval_residual_seasons is None:
        interval_model = load_or_fit_interval_model(panel, players)
    else:
        interval_model = fit_interval_model(
            _all_residuals(panel, players, list(interval_residual_seasons))
        )
    offense = add_intervals(projection, interval_model).select(DISPLAY_COLS)
    if not include_special:
        return offense
    special = _special_scored(season, fmt, interval_model)
    return offense if special is None else pl.concat([offense, special])
