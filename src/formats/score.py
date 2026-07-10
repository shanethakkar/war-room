"""Score a component stat line into fantasy points for a format.

The interpreter for a ``ScoringConfig``: given a frame with component columns
(projected or actual), compute fantasy points. Lives with the format config it
interprets so both the projection and decision layers can reuse it without a
layer inversion.

Expected component columns (as produced by the baseline projection):
``pass_yards``, ``pass_tds``, ``interceptions``, ``rush_yards``, ``rush_tds``,
``receptions``, ``rec_yards``, ``rec_tds``, ``fumbles_lost``.
"""

from __future__ import annotations

import polars as pl

from src.formats.base import FormatConfig, ScoringConfig


def score_expr(scoring: ScoringConfig) -> pl.Expr:
    """Fantasy-points expression for a component stat line under ``scoring``."""
    return (
        pl.col("pass_yards") * scoring.pass_yd
        + pl.col("pass_tds") * scoring.pass_td
        + pl.col("interceptions") * scoring.pass_int
        + pl.col("rush_yards") * scoring.rush_yd
        + pl.col("rush_tds") * scoring.rush_td
        + pl.col("receptions") * scoring.rec
        + pl.col("rec_yards") * scoring.rec_yd
        + pl.col("rec_tds") * scoring.rec_td
        + pl.col("fumbles_lost") * scoring.fumble_lost
    )


def score_components(
    df: pl.DataFrame,
    fmt: FormatConfig | ScoringConfig,
    alias: str = "projected_points",
) -> pl.DataFrame:
    """Attach a fantasy-points column scored under ``fmt`` (format or scoring)."""
    scoring = fmt.scoring if isinstance(fmt, FormatConfig) else fmt
    return df.with_columns(score_expr(scoring).alias(alias))
