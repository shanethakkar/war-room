"""Score component stat lines into fantasy points for a format.

The interpreter for a ``ScoringConfig``: given frames with component columns
(projected or actual), compute fantasy points. Position-aware - offense, kicker,
and DST each carry their own component columns and scoring rules:

- offense: ``pass_yards``/``pass_tds``/... (+ ``te_rec_bonus`` for TE rows)
- kicker:  ``fg_0_39``/``fg_40_49``/``fg_50_plus``/``fg_missed``/``pat_*``
- DST:     ``sacks``/``ints``/``fumble_recs``/``dst_tds``/``safeties`` + counts
           of games in each points-allowed bracket (``games_pa_*``)
"""

from __future__ import annotations

import polars as pl

from src.formats.base import FormatConfig, ScoringConfig

# DST points-allowed bracket component columns, paired with their scoring field.
DST_PA_BUCKETS: tuple[tuple[str, str], ...] = (
    ("games_pa_0", "dst_pa_0"),
    ("games_pa_1_6", "dst_pa_1_6"),
    ("games_pa_7_13", "dst_pa_7_13"),
    ("games_pa_14_20", "dst_pa_14_20"),
    ("games_pa_21_27", "dst_pa_21_27"),
    ("games_pa_28_34", "dst_pa_28_34"),
    ("games_pa_35_plus", "dst_pa_35_plus"),
)


def score_expr(scoring: ScoringConfig) -> pl.Expr:
    """Fantasy-points expression for an OFFENSE component stat line."""
    rec_value = (
        pl.when(pl.col("position_group") == "TE")
        .then(scoring.rec + scoring.te_rec_bonus)
        .otherwise(scoring.rec)
        if scoring.te_rec_bonus
        else pl.lit(scoring.rec)
    )
    return (
        pl.col("pass_yards") * scoring.pass_yd
        + pl.col("pass_tds") * scoring.pass_td
        + pl.col("interceptions") * scoring.pass_int
        + pl.col("rush_yards") * scoring.rush_yd
        + pl.col("rush_tds") * scoring.rush_td
        + pl.col("receptions") * rec_value
        + pl.col("rec_yards") * scoring.rec_yd
        + pl.col("rec_tds") * scoring.rec_td
        + pl.col("fumbles_lost") * scoring.fumble_lost
    )


def kicker_score_expr(scoring: ScoringConfig) -> pl.Expr:
    """Fantasy-points expression for a KICKER component stat line."""
    return (
        pl.col("fg_0_39") * scoring.fg_0_39
        + pl.col("fg_40_49") * scoring.fg_40_49
        + pl.col("fg_50_plus") * scoring.fg_50_plus
        + pl.col("fg_missed") * scoring.fg_miss
        + pl.col("pat_made") * scoring.pat_made
        + pl.col("pat_missed") * scoring.pat_miss
    )


def dst_score_expr(scoring: ScoringConfig) -> pl.Expr:
    """Fantasy-points expression for a DST component stat line."""
    expr = (
        pl.col("sacks") * scoring.dst_sack
        + pl.col("ints") * scoring.dst_int
        + pl.col("fumble_recs") * scoring.dst_fumble_rec
        + pl.col("dst_tds") * scoring.dst_td
        + pl.col("safeties") * scoring.dst_safety
    )
    for column, field in DST_PA_BUCKETS:
        expr = expr + pl.col(column) * getattr(scoring, field)
    return expr


def score_components(
    df: pl.DataFrame,
    fmt: FormatConfig | ScoringConfig,
    alias: str = "projected_points",
) -> pl.DataFrame:
    """Attach a fantasy-points column to an OFFENSE frame scored under ``fmt``."""
    scoring = fmt.scoring if isinstance(fmt, FormatConfig) else fmt
    return df.with_columns(score_expr(scoring).alias(alias))


def score_special(
    df: pl.DataFrame,
    fmt: FormatConfig | ScoringConfig,
    alias: str = "projected_points",
) -> pl.DataFrame:
    """Attach fantasy points to a K/DST frame (scored per position group)."""
    scoring = fmt.scoring if isinstance(fmt, FormatConfig) else fmt
    return df.with_columns(
        pl.when(pl.col("position_group") == "K")
        .then(kicker_score_expr(scoring))
        .otherwise(dst_score_expr(scoring))
        .alias(alias)
    )
