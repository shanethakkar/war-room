"""Priors for the transparent baseline (design.md §4.2-4.4).

Everything here is estimated from the TRAINING panel - the player-seasons
strictly before the projected season - so the backtest stays leakage-free. Two
prior sets:

- ``positional_priors``: opportunity-weighted pooled efficiency and TD rates per
  position group. These are the role means that thin-sample player rates regress
  toward (partial pooling, done explicitly).
- ``rookie_priors`` / ``rookie_priors_by_position``: pooled rookie-season per-game
  production by position (and draft round) - the draft-capital prior for players
  with no NFL history (design.md §4.4), plus a position-level fallback for thin
  draft-round buckets.

TD rates use *expected* touchdowns (from ff_opportunity), coalescing to actual
only when expected is missing - never raw prior-year TDs as the primary signal.
"""

from __future__ import annotations

import polars as pl

SKILL_POSITION_GROUPS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Undrafted players (null round) get a synthetic round below the last real one,
# so they carry their own (lower) rookie prior rather than a drafted-player mean.
UDFA_ROUND: int = 8


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """num / den, or null when den <= 0 (no opportunity -> no meaningful rate)."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def td_signal_exprs() -> list[pl.Expr]:
    """Expected-TD columns, falling back to actual TDs only where expected is null.

    Also collapses the two fumble-lost columns into one. Shared by every prior and
    by the recent-form aggregation so the TD signal is defined in exactly one place.
    """
    return [
        pl.coalesce("expected_rec_tds", "receiving_tds").alias("rec_td_signal"),
        pl.coalesce("expected_rush_tds", "rushing_tds").alias("rush_td_signal"),
        pl.coalesce("expected_pass_tds", "passing_tds").alias("pass_td_signal"),
        (
            pl.col("rushing_fumbles_lost").fill_null(0)
            + pl.col("receiving_fumbles_lost").fill_null(0)
        ).alias("fumbles_lost"),
    ]


def positional_priors(train: pl.DataFrame) -> pl.DataFrame:
    """Opportunity-weighted pooled efficiency and TD rates per position group.

    Pooled ratios (``sum(num) / sum(den)``) are dominated by high-opportunity
    player-seasons, so they represent the typical outcome per opportunity for a
    meaningful role - the right thing to shrink toward.
    """
    pooled = (
        train.with_columns(td_signal_exprs())
        .group_by("position_group")
        .agg(
            pl.col("targets").sum().alias("targets"),
            pl.col("receptions").sum().alias("receptions"),
            pl.col("receiving_yards").sum().alias("rec_yards"),
            pl.col("rec_td_signal").sum().alias("rec_td"),
            pl.col("carries").sum().alias("carries"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("rush_td_signal").sum().alias("rush_td"),
            pl.col("pass_attempts").sum().alias("pass_att"),
            pl.col("pass_completions").sum().alias("pass_comp"),
            pl.col("passing_yards").sum().alias("pass_yards"),
            pl.col("pass_td_signal").sum().alias("pass_td"),
            pl.col("interceptions").sum().alias("interceptions"),
            pl.col("fumbles_lost").sum().alias("fumbles_lost"),
            pl.col("games").sum().alias("games"),
        )
    )
    c = pl.col
    return pooled.select(
        "position_group",
        _safe_ratio(c("receptions"), c("targets")).alias("catch_rate"),
        _safe_ratio(c("rec_yards"), c("targets")).alias("yards_per_target"),
        _safe_ratio(c("rush_yards"), c("carries")).alias("yards_per_carry"),
        _safe_ratio(c("pass_yards"), c("pass_att")).alias("yards_per_pass_attempt"),
        _safe_ratio(c("pass_comp"), c("pass_att")).alias("completion_pct"),
        _safe_ratio(c("rec_td"), c("targets")).alias("rec_td_rate"),
        _safe_ratio(c("rush_td"), c("carries")).alias("rush_td_rate"),
        _safe_ratio(c("pass_td"), c("pass_att")).alias("pass_td_rate"),
        _safe_ratio(c("interceptions"), c("pass_att")).alias("interception_rate"),
        _safe_ratio(c("fumbles_lost"), c("carries") + c("targets")).alias(
            "fumble_rate"
        ),
        _safe_ratio(c("targets"), c("games")).alias("targets_per_game"),
        _safe_ratio(c("carries"), c("games")).alias("carries_per_game"),
        _safe_ratio(c("pass_att"), c("games")).alias("pass_attempts_per_game"),
    )


def _rookies_frame(train: pl.DataFrame) -> pl.DataFrame:
    """Rookie-season rows with UDFA rounds filled and TD signals attached."""
    return train.filter(pl.col("is_rookie")).with_columns(
        pl.col("draft_round").fill_null(UDFA_ROUND).alias("draft_round"),
        *td_signal_exprs(),
    )


def _rookie_rates(rookies: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Pooled rookie per-game volume + per-opportunity efficiency over ``group_cols``.

    Output uses the standard intermediate rate names the projection engine
    consumes, so drafted-round and position-level priors are interchangeable in a
    coalesce fallback.
    """
    pooled = rookies.group_by(group_cols).agg(
        pl.len().alias("n_rookies"),
        pl.col("games").sum().alias("g"),
        pl.col("targets").sum().alias("tgt"),
        pl.col("receptions").sum().alias("rec"),
        pl.col("receiving_yards").sum().alias("recyd"),
        pl.col("rec_td_signal").sum().alias("rectd"),
        pl.col("carries").sum().alias("car"),
        pl.col("rushing_yards").sum().alias("rushyd"),
        pl.col("rush_td_signal").sum().alias("rushtd"),
        pl.col("pass_attempts").sum().alias("patt"),
        pl.col("pass_completions").sum().alias("pcomp"),
        pl.col("passing_yards").sum().alias("pyd"),
        pl.col("pass_td_signal").sum().alias("ptd"),
        pl.col("interceptions").sum().alias("pint"),
    )
    c = pl.col
    return pooled.select(
        *group_cols,
        "n_rookies",
        _safe_ratio(c("g"), c("n_rookies")).alias("games_per_rookie"),
        _safe_ratio(c("tgt"), c("g")).alias("tgt_pg"),
        _safe_ratio(c("car"), c("g")).alias("car_pg"),
        _safe_ratio(c("patt"), c("g")).alias("patt_pg"),
        _safe_ratio(c("rec"), c("tgt")).alias("catch_rate"),
        _safe_ratio(c("recyd"), c("tgt")).alias("ypt"),
        _safe_ratio(c("rectd"), c("tgt")).alias("rec_td_rate"),
        _safe_ratio(c("rushyd"), c("car")).alias("ypc"),
        _safe_ratio(c("rushtd"), c("car")).alias("rush_td_rate"),
        _safe_ratio(c("pcomp"), c("patt")).alias("comp_pct"),
        _safe_ratio(c("pyd"), c("patt")).alias("ypa"),
        _safe_ratio(c("ptd"), c("patt")).alias("pass_td_rate"),
        _safe_ratio(c("pint"), c("patt")).alias("int_rate"),
    )


def rookie_priors(train: pl.DataFrame) -> pl.DataFrame:
    """Rookie per-game production by position group and draft round."""
    return _rookie_rates(_rookies_frame(train), ["position_group", "draft_round"])


def rookie_priors_by_position(train: pl.DataFrame) -> pl.DataFrame:
    """Position-level rookie prior - fallback for thin/empty draft-round buckets."""
    return _rookie_rates(_rookies_frame(train), ["position_group"])
