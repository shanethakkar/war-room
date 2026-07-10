"""Feature construction for the Bayesian model (Polars, leakage-free).

The Bayesian model predicts a player's next-season fantasy points-per-game (ppg)
from their most recent prior season, with player and position pooling and an aging
curve. This module builds:

- ``training_pairs`` - (prior-season features -> that player's next-season ppg)
  pairs, using only target seasons strictly before a cutoff.
- ``projection_features`` - one row per returning player: their latest prior
  season's features, carried forward to the projected season.

Opportunity features (target share, snap share, expected ppg) are the predictors,
honoring "opportunity is king". Ppg is modeled (not season totals) so games/
availability is handled separately by the projection layer.
"""

from __future__ import annotations

import polars as pl

# Predictor columns, in a fixed order (the model's design matrix depends on it).
PREDICTORS: tuple[str, ...] = (
    "prev_ppg",
    "prev_exp_ppg",
    "prev_target_share",
    "prev_snap_share",
    "age",
    "age2",
)

# Only use a prior season within this many years of the target: a player whose
# last game was long ago (retired/washed out) must not be projected off stale
# features (e.g. Andrew Luck's 2018 season resurfacing in a 2025 projection).
RECENCY_WINDOW = 2


def _prepared(panel: pl.DataFrame) -> pl.DataFrame:
    """Per player-season: points-per-game, expected ppg, and cleaned shares."""
    return panel.filter(pl.col("games") >= 1).with_columns(
        (pl.col("fantasy_points_ppr") / pl.col("games")).alias("ppg"),
        (
            pl.coalesce("expected_fantasy_points", "fantasy_points_ppr")
            / pl.col("games")
        ).alias("exp_ppg"),
        pl.col("target_share").fill_null(0.0),
        pl.col("snap_share").fill_null(0.0),
    )


def _prior_features(prepared: pl.DataFrame) -> pl.DataFrame:
    """Feature-side view of each season (renamed to prev_* for joining forward)."""
    return prepared.select(
        "player_id",
        pl.col("season").alias("feat_season"),
        pl.col("position_group"),
        pl.col("player_name"),
        pl.col("position"),
        pl.col("team"),
        pl.col("age").alias("age_at_feat"),
        pl.col("games").alias("prev_games"),
        pl.col("ppg").alias("prev_ppg"),
        pl.col("exp_ppg").alias("prev_exp_ppg"),
        pl.col("target_share").alias("prev_target_share"),
        pl.col("snap_share").alias("prev_snap_share"),
    )


def training_pairs(panel: pl.DataFrame, before_season: int) -> pl.DataFrame:
    """(prior-season features -> next-season ppg) pairs with target season < cutoff.

    For each player-season used as a target, the features come from that player's
    most recent earlier season. Only target seasons strictly before
    ``before_season`` are kept, so a model fit for projecting year Y never sees Y.
    """
    prepared = _prepared(panel)
    targets = prepared.select(
        "player_id",
        pl.col("season").alias("target_season"),
        pl.col("age"),
        pl.col("ppg").alias("target_ppg"),
    )
    features = _prior_features(prepared)

    pairs = (
        targets.join(features, on="player_id", how="inner")
        .filter(pl.col("feat_season") < pl.col("target_season"))
        .sort(["player_id", "target_season", "feat_season"])
        .group_by(["player_id", "target_season"], maintain_order=True)
        .last()  # most recent prior season per (player, target)
        .filter(pl.col("target_season") < before_season)
        .filter(pl.col("target_season") - pl.col("feat_season") <= RECENCY_WINDOW)
        .with_columns((pl.col("age") ** 2).alias("age2"))
    )
    return pairs.drop_nulls(["target_ppg", *PREDICTORS])


def projection_features(panel: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Latest prior-season features per returning player, carried to ``target_season``.

    Age is advanced to the projected season; ``prev_games`` is kept for the games
    projection. Only players with a season before ``target_season`` appear
    (rookies are handled separately by the projection layer).
    """
    prepared = _prior_features(_prepared(panel)).filter(
        pl.col("feat_season") < target_season
    )
    latest = (
        prepared.sort(["player_id", "feat_season"])
        .group_by("player_id", maintain_order=True)
        .last()
        # Drop players whose most recent season is too old to project (retired).
        .filter(pl.col("feat_season") >= target_season - RECENCY_WINDOW)
    )
    return (
        latest.with_columns(
            (pl.col("age_at_feat") + (target_season - pl.col("feat_season"))).alias(
                "age"
            )
        )
        .with_columns((pl.col("age") ** 2).alias("age2"))
        .drop_nulls(list(PREDICTORS))
    )
