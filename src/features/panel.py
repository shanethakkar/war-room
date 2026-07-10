"""Player-season feature panel construction (design.md §4.1-4.2).

Pure ``panel -> panel`` transforms that assemble one row per player-season for
offensive skill players (QB/RB/WR/TE). The panel is the clean historical record
the projection layer consumes; it carries, in descending order of trust:

- **volume / role** - targets, shares, carries, snaps. The load-bearing signal.
- **efficiency** - per-opportunity rates (regressed toward role means later, by
  the projection layer; here we just expose the raw rates).
- **opportunity** - expected TDs and expected fantasy points from
  ``ff_opportunity`` (the TD backbone: never project off raw prior-year TDs).
- **context** - age, draft capital, experience.

All I/O lives in ``build.py``; every function here is deterministic. Column names
and dtypes were verified against the cached nflverse tables (nflreadpy 0.1.5).
"""

from __future__ import annotations

import polars as pl

# Offensive skill positions we project in Phase 1 (K/DST come later).
SKILL_POSITION_GROUPS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Regular season only: 17 games pre-2021, 18 from 2021. ff_opportunity carries
# playoff weeks (19-22); player_stats_season is already REG, so we align.
LAST_REGULAR_WEEK: int = 18


def _safe_div(numerator: str, denominator: str) -> pl.Expr:
    """Divide two columns, returning null (not inf/NaN) when the denominator is 0.

    Efficiency rates are meaningless without opportunity; null is the honest value
    for a zero-target or zero-carry player and lets the projection layer's
    shrinkage treat it as missing rather than a real 0.0.
    """
    num, den = pl.col(numerator), pl.col(denominator)
    return pl.when(den > 0).then(num / den).otherwise(None)


def offensive_player_seasons(player_stats_season: pl.DataFrame) -> pl.DataFrame:
    """Filter to skill players and select/rename the raw production base.

    Input is ``load_player_stats(summary_level="reg")`` - one row per
    player-season of regular-season production, with nflverse-computed shares
    (``target_share``, ``air_yards_share``, ``wopr``) and fantasy totals.
    """
    return player_stats_season.filter(
        pl.col("position_group").is_in(SKILL_POSITION_GROUPS)
    ).select(
        # identity / context
        pl.col("player_id"),
        pl.col("player_display_name").alias("player_name"),
        pl.col("position"),
        pl.col("position_group"),
        pl.col("recent_team").alias("team"),
        pl.col("season"),
        pl.col("games"),
        # receiving volume
        pl.col("targets"),
        pl.col("receptions"),
        pl.col("receiving_yards"),
        pl.col("receiving_tds"),
        pl.col("receiving_air_yards"),
        pl.col("receiving_yards_after_catch"),
        pl.col("target_share"),
        pl.col("air_yards_share"),
        pl.col("wopr"),
        # rushing volume
        pl.col("carries"),
        pl.col("rushing_yards"),
        pl.col("rushing_tds"),
        # passing volume
        pl.col("attempts").alias("pass_attempts"),
        pl.col("completions").alias("pass_completions"),
        pl.col("passing_yards"),
        pl.col("passing_tds"),
        pl.col("passing_interceptions").alias("interceptions"),
        pl.col("passing_air_yards"),
        # turnovers
        pl.col("rushing_fumbles_lost"),
        pl.col("receiving_fumbles_lost"),
        # fantasy totals (reference targets; format scoring applied downstream)
        pl.col("fantasy_points"),
        pl.col("fantasy_points_ppr"),
    )


def add_efficiency(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach per-opportunity efficiency rates and per-game production.

    These are exposed raw; the projection layer regresses them toward role means
    (design.md §4.2). Prior-year efficiency is unreliable for low-volume players,
    which is exactly why ``_safe_div`` returns null on zero opportunity.
    """
    return panel.with_columns(
        _safe_div("receptions", "targets").alias("catch_rate"),
        _safe_div("receiving_yards", "targets").alias("yards_per_target"),
        _safe_div("receiving_yards_after_catch", "receptions").alias(
            "yac_per_reception"
        ),
        _safe_div("rushing_yards", "carries").alias("yards_per_carry"),
        _safe_div("passing_yards", "pass_attempts").alias("yards_per_pass_attempt"),
        _safe_div("pass_completions", "pass_attempts").alias("completion_pct"),
        _safe_div("passing_tds", "pass_attempts").alias("pass_td_rate"),
        _safe_div("interceptions", "pass_attempts").alias("interception_rate"),
        _safe_div("fantasy_points_ppr", "games").alias("ppr_points_per_game"),
    )


def season_opportunity(ff_opportunity: pl.DataFrame) -> pl.DataFrame:
    """Aggregate weekly ff_opportunity to player-season expected values.

    The expected-TD and expected-fantasy-point columns are the backbone for
    regressing touchdowns to role (design.md §4.2). Restricted to the regular
    season to match the panel, and keyed by gsis ``player_id``. ``season`` is a
    string in this table (a known cross-table dtype gotcha) and is cast to Int32.
    """
    return (
        ff_opportunity.filter(pl.col("week") <= LAST_REGULAR_WEEK)
        .with_columns(pl.col("season").cast(pl.Int32))
        .group_by("player_id", "season")
        .agg(
            pl.col("pass_touchdown_exp").sum().alias("expected_pass_tds"),
            pl.col("rush_touchdown_exp").sum().alias("expected_rush_tds"),
            pl.col("rec_touchdown_exp").sum().alias("expected_rec_tds"),
            pl.col("receptions_exp").sum().alias("expected_receptions"),
            pl.col("rush_yards_gained_exp").sum().alias("expected_rush_yards"),
            pl.col("rec_yards_gained_exp").sum().alias("expected_rec_yards"),
            pl.col("total_fantasy_points_exp").sum().alias("expected_fantasy_points"),
        )
    )


def season_snap_share(
    snap_counts: pl.DataFrame, crosswalk: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate weekly snap counts to a player-season average snap share.

    ``snap_counts`` is keyed by ``pfr_player_id``; ``crosswalk`` maps that to the
    gsis ``player_id`` used everywhere else. ``offense_pct`` is a 0-1 fraction, so
    the mean over games is the average share of team offensive snaps.
    """
    return (
        snap_counts.join(
            crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner"
        )
        .group_by("player_id", "season")
        .agg(
            pl.col("offense_pct").mean().alias("snap_share"),
            pl.col("offense_snaps").sum().alias("offense_snaps"),
        )
    )


def player_reference(players: pl.DataFrame) -> pl.DataFrame:
    """Select the static per-player reference: birthdate and draft capital.

    Draft position is the strongest available predictor of opportunity for players
    with thin NFL history (design.md §4.4). ``birth_date`` is a string here and is
    parsed to a Date for age computation downstream.
    """
    return players.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False).alias("birth_date"),
        pl.col("draft_year"),
        pl.col("draft_round"),
        pl.col("draft_pick"),
        pl.col("rookie_season"),
    ).filter(pl.col("player_id").is_not_null())


def pfr_crosswalk(players: pl.DataFrame) -> pl.DataFrame:
    """Unique gsis_id <-> pfr_id map for joining PFR-keyed tables (snap counts)."""
    return (
        players.select(pl.col("gsis_id").alias("player_id"), pl.col("pfr_id"))
        .drop_nulls()
        .unique()
    )


def assemble_panel(
    player_stats_season: pl.DataFrame,
    ff_opportunity: pl.DataFrame,
    snap_counts: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Join the sources into the final player-season panel.

    Left joins throughout: a player-season present in production stats is never
    dropped for lacking a snap-count or opportunity match (missing → null).
    """
    base = add_efficiency(offensive_player_seasons(player_stats_season))
    opportunity = season_opportunity(ff_opportunity)
    snaps = season_snap_share(snap_counts, pfr_crosswalk(players))
    reference = player_reference(players)

    panel = (
        base.join(opportunity, on=["player_id", "season"], how="left")
        .join(snaps, on=["player_id", "season"], how="left")
        .join(reference, on="player_id", how="left")
    )

    # Age as of Sept 1 of the season; experience/rookie flag from rookie_season.
    panel = panel.with_columns(
        (
            (pl.date(pl.col("season"), 9, 1) - pl.col("birth_date")).dt.total_days()
            / 365.25
        )
        .round(1)
        .alias("age"),
        (pl.col("season") - pl.col("rookie_season")).alias("experience"),
        (pl.col("season") == pl.col("rookie_season")).alias("is_rookie"),
    )

    return panel.sort(["season", "fantasy_points_ppr"], descending=[False, True])
