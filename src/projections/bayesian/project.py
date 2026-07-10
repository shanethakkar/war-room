"""Bayesian projection for a season - the swap-in that must beat the baseline.

Returning players are projected by the hierarchical model's posterior predictive
(``model.FitResult.predict``), which gives calibrated intervals directly. Rookies
have no NFL history for the model, so they reuse the baseline's draft-capital prior
(design.md §4.4) with a deliberately wide interval. Output matches the scored
schema the board/backtest expect: point estimate + 80% interval per player.
"""

from __future__ import annotations

import polars as pl

from src.formats import get_format
from src.formats.score import score_components
from src.projections.baseline.priors import (
    rookie_priors,
    rookie_priors_by_position,
)
from src.projections.baseline.project import (
    project_rookies,
)
from src.projections.bayesian.features import projection_features, training_pairs
from src.projections.bayesian.model import fit_model

# The schema every projection (baseline or bayesian) exposes downstream.
SCORED_SCHEMA: tuple[str, ...] = (
    "player_id",
    "player_name",
    "position",
    "position_group",
    "team",
    "is_rookie",
    "projected_games",
    "projected_points",
    "points_low",
    "points_median",
    "points_high",
)

# Games projection (single prior season, lightly shrunk toward a durability prior).
_DURABILITY_GAMES = 16.0
_DURABILITY_WEIGHT = 0.15
# Rookie interval: wide multiplicative band (rookies are highly uncertain; the
# model can't posterior-predict them without NFL history).
_ROOKIE_LOW = 0.35
_ROOKIE_HIGH = 1.9

_MIN_PAIRS = 100


def _rookie_projection(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    target_season: int,
    exclude_ids: list[str],
) -> pl.DataFrame:
    """Baseline draft-capital rookie projection with a wide fixed interval."""
    train = panel.filter(pl.col("season") < target_season)
    rookies = project_rookies(
        target_season,
        players,
        rookie_priors(train),
        rookie_priors_by_position(train),
        exclude_ids,
    )
    scored = score_components(rookies, get_format("redraft_ppr").scoring)
    return scored.with_columns(
        (pl.col("projected_points") * _ROOKIE_LOW).alias("points_low"),
        pl.col("projected_points").alias("points_median"),
        (pl.col("projected_points") * _ROOKIE_HIGH).alias("points_high"),
    )


def bayesian_project_season(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    target_season: int,
    *,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """Project ``target_season`` Bayesianly (returning) + baseline rookies.

    Leakage-free: the model trains only on target seasons before ``target_season``.
    """
    pairs = training_pairs(panel, target_season)
    if pairs.height < _MIN_PAIRS:
        raise ValueError(
            f"Only {pairs.height} training pairs before {target_season}; "
            f"need >= {_MIN_PAIRS}. Project a later season."
        )
    fit = fit_model(pairs, draws=draws, tune=tune, chains=chains, seed=seed)

    features = projection_features(panel, target_season).with_columns(
        (
            (1 - _DURABILITY_WEIGHT) * pl.col("prev_games")
            + _DURABILITY_WEIGHT * _DURABILITY_GAMES
        )
        .clip(1.0, 17.0)
        .alias("projected_games")
    )
    predicted = fit.predict(features, seed=seed)
    returning = (
        features.select(
            "player_id",
            "player_name",
            "position",
            "position_group",
            "team",
            "projected_games",
        )
        .join(predicted, on="player_id", how="inner")
        .with_columns(pl.lit(False).alias("is_rookie"))
    )

    rookies = _rookie_projection(
        panel, players, target_season, returning["player_id"].to_list()
    )

    combined = pl.concat(
        [returning.select(SCORED_SCHEMA), rookies.select(SCORED_SCHEMA)],
        how="vertical",
    )
    return combined.sort("projected_points", descending=True)
