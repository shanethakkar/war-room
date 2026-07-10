"""Model-agnostic projection pipeline.

Produces a scored projection with an 80% interval for a season, routing to either
model. Both return the same schema (identity + ``projected_points`` +
``points_low``/``points_median``/``points_high``), so the decision layer and the
backtest never branch on the model.

- **baseline**: component projection -> per-format scoring -> empirical intervals.
- **bayesian**: hierarchical posterior predictive (points + intervals) directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from src.formats import get_format
from src.formats.score import score_components
from src.projections import MODELS
from src.projections.baseline.project import project_season
from src.projections.bayesian.project import bayesian_project_season
from src.projections.uncertainty import (
    add_intervals,
    collect_residuals,
    fit_interval_model,
    load_or_fit_interval_model,
)


def scored_projection(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    season: int,
    *,
    model: str = "baseline",
    fmt_key: str = "redraft_ppr",
    interval_residual_seasons: Sequence[int] | None = None,
    bayes_kwargs: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Scored projection + intervals for ``season`` from the chosen model.

    ``interval_residual_seasons`` (baseline only) pins the empirical interval model
    to specific seasons for a leakage-free backtest; ``None`` uses the cached model.
    ``bayes_kwargs`` forwards sampler settings (e.g. ``draws``) to the Bayesian fit.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}.")

    if model == "bayesian":
        return bayesian_project_season(panel, players, season, **(bayes_kwargs or {}))

    projection = score_components(
        project_season(panel, players, season), get_format(fmt_key).scoring
    )
    if interval_residual_seasons is None:
        interval_model = load_or_fit_interval_model(panel, players)
    else:
        interval_model = fit_interval_model(
            collect_residuals(panel, players, interval_residual_seasons)
        )
    return add_intervals(projection, interval_model)
