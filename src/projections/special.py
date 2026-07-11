"""Kicker and team-defense (DST) projections.

Every standard league starts a K and a DST, so the board must rank them - but
their year-over-year signal is weak (DST notoriously so), so the honest model is
simple and heavily regressed: recency-weighted per-game component rates shrunk
hard toward the league mean, times projected games. Components stay unscored so
any league's scoring config applies (``formats.score.score_special``).

Component columns (union across K and DST, zero-filled so one frame serves both):
- K:   fg_0_39, fg_40_49, fg_50_plus, fg_missed, pat_made, pat_missed
- DST: sacks, ints, fumble_recs, dst_tds, safeties, games_pa_* bracket counts

DST identity: ``player_id = "DST_<abbr>"``, ``player_name = <full team name>``
(matches FFC's DEF naming for ADP joins).
"""

from __future__ import annotations

import polars as pl

K_COMPONENTS: tuple[str, ...] = (
    "fg_0_39",
    "fg_40_49",
    "fg_50_plus",
    "fg_missed",
    "pat_made",
    "pat_missed",
)
DST_COUNT_COMPONENTS: tuple[str, ...] = (
    "sacks",
    "ints",
    "fumble_recs",
    "dst_tds",
    "safeties",
)
PA_BUCKETS: tuple[str, ...] = (
    "games_pa_0",
    "games_pa_1_6",
    "games_pa_7_13",
    "games_pa_14_20",
    "games_pa_21_27",
    "games_pa_28_34",
    "games_pa_35_plus",
)
ALL_COMPONENTS: tuple[str, ...] = K_COMPONENTS + DST_COUNT_COMPONENTS + PA_BUCKETS

# Recency weighting (mirrors the offense baseline).
DECAY = 0.7
LOOKBACK = 2
# Per-game rate shrinkage toward the league mean, in games units. DST regresses
# harder than K: team defense barely persists year over year.
K_SHRINK_GAMES = 20.0
DST_SHRINK_GAMES = 34.0
KICKER_DEFAULT_GAMES = 16.0
DST_GAMES = 17.0


def _zero_fill(df: pl.DataFrame) -> pl.DataFrame:
    """Add any missing component columns as zeros (union-schema convention)."""
    missing = [c for c in ALL_COMPONENTS if c not in df.columns]
    return df.with_columns([pl.lit(0.0).alias(c) for c in missing]).with_columns(
        [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in ALL_COMPONENTS]
    )


def kicker_actuals(player_stats_season: pl.DataFrame) -> pl.DataFrame:
    """Per kicker-season component stats from the cached player stats."""
    kickers = player_stats_season.filter(pl.col("position") == "K").select(
        pl.col("player_id"),
        pl.col("player_display_name").alias("player_name"),
        pl.lit("K").alias("position"),
        pl.lit("K").alias("position_group"),
        pl.col("recent_team").alias("team"),
        pl.col("season"),
        pl.col("games"),
        (
            pl.col("fg_made_0_19").fill_null(0)
            + pl.col("fg_made_20_29").fill_null(0)
            + pl.col("fg_made_30_39").fill_null(0)
        ).alias("fg_0_39"),
        pl.col("fg_made_40_49").fill_null(0).alias("fg_40_49"),
        (
            pl.col("fg_made_50_59").fill_null(0) + pl.col("fg_made_60_").fill_null(0)
        ).alias("fg_50_plus"),
        pl.col("fg_missed").fill_null(0).alias("fg_missed"),
        pl.col("pat_made").fill_null(0).alias("pat_made"),
        pl.col("pat_missed").fill_null(0).alias("pat_missed"),
    )
    return _zero_fill(kickers)


def dst_actuals(
    pbp: pl.DataFrame, schedules: pl.DataFrame, teams: pl.DataFrame
) -> pl.DataFrame:
    """Per team-season DST component stats from play-by-play + schedules.

    Regular season only (both sources carry playoffs), matching the offense panel.
    """
    pbp = pbp.filter(pl.col("season_type") == "REG")
    schedules = schedules.filter(pl.col("game_type") == "REG")
    # Defensive counting stats, credited to the defense on the play.
    defense = (
        pbp.group_by("defteam", "season")
        .agg(
            pl.col("sack").sum().alias("sacks"),
            pl.col("interception").sum().alias("ints"),
            pl.col("fumble_lost").sum().alias("fumble_recs"),
            pl.col("safety").sum().alias("safeties"),
        )
        .filter(pl.col("defteam").is_not_null())
    )
    # TDs for the fantasy DST: any TD scored without possession (pick-six,
    # fumble return) plus punt/kick return TDs (special teams count for DST).
    dst_td = (
        pbp.filter(
            (pl.col("touchdown") == 1)
            & (
                (pl.col("td_team") != pl.col("posteam"))
                | pl.col("play_type").is_in(["punt", "kickoff"])
            )
        )
        .group_by(pl.col("td_team").alias("defteam"), "season")
        .agg(pl.len().alias("dst_tds"))
    )
    # Points allowed per game from final scores -> bracket counts per season.
    home = schedules.select(
        pl.col("season"),
        pl.col("home_team").alias("team"),
        pl.col("away_score").alias("pa"),
    )
    away = schedules.select(
        pl.col("season"),
        pl.col("away_team").alias("team"),
        pl.col("home_score").alias("pa"),
    )
    games = pl.concat([home, away]).drop_nulls("pa")
    pa = games.group_by("team", "season").agg(
        pl.len().alias("games"),
        (pl.col("pa") == 0).sum().alias("games_pa_0"),
        pl.col("pa").is_between(1, 6).sum().alias("games_pa_1_6"),
        pl.col("pa").is_between(7, 13).sum().alias("games_pa_7_13"),
        pl.col("pa").is_between(14, 20).sum().alias("games_pa_14_20"),
        pl.col("pa").is_between(21, 27).sum().alias("games_pa_21_27"),
        pl.col("pa").is_between(28, 34).sum().alias("games_pa_28_34"),
        (pl.col("pa") >= 35).sum().alias("games_pa_35_plus"),
    )
    names = teams.select(
        pl.col("team_abbr").alias("team"), pl.col("team_name").alias("player_name")
    ).unique(subset="team")

    combined = (
        pa.join(
            defense,
            left_on=["team", "season"],
            right_on=["defteam", "season"],
            how="left",
        )
        .join(
            dst_td,
            left_on=["team", "season"],
            right_on=["defteam", "season"],
            how="left",
        )
        .join(names, on="team", how="left")
        .with_columns(
            ("DST_" + pl.col("team")).alias("player_id"),
            pl.coalesce("player_name", pl.col("team")).alias("player_name"),
            pl.lit("DST").alias("position"),
            pl.lit("DST").alias("position_group"),
        )
        .select(
            "player_id",
            "player_name",
            "position",
            "position_group",
            "team",
            "season",
            "games",
            *DST_COUNT_COMPONENTS,
            *PA_BUCKETS,
        )
    )
    return _zero_fill(combined)


def build_special_panel(
    player_stats_season: pl.DataFrame,
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    teams: pl.DataFrame,
) -> pl.DataFrame:
    """K + DST actuals in one union-schema frame (one row per entity-season)."""
    k = kicker_actuals(player_stats_season).with_columns(pl.col("games").cast(pl.Int32))
    d = dst_actuals(pbp, schedules, teams).with_columns(pl.col("games").cast(pl.Int32))
    return pl.concat([k.select(d.columns), d], how="vertical").sort(
        ["season", "position_group", "player_id"]
    )


def project_special(special_panel: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Project K/DST components for ``target_season`` (leakage-free).

    Recency-weighted per-game rates, shrunk toward the position's league-mean
    per-game rate; PA bucket rates are renormalized to sum to one game before
    scaling, so bracket points can't drift. DST plays all 17 games; kickers get
    their recent games shrunk toward a durability prior.
    """
    window = special_panel.filter(
        (pl.col("season") < target_season)
        & (pl.col("season") >= target_season - LOOKBACK)
    ).with_columns((DECAY ** (target_season - 1 - pl.col("season"))).alias("w"))
    if window.height == 0:
        raise ValueError(f"No special-teams history before {target_season}.")

    weighted = window.group_by(
        "player_id", "player_name", "position", "position_group"
    ).agg(
        pl.col("team").last(),
        (pl.col("w") * pl.col("games")).sum().alias("w_games"),
        pl.col("w").sum().alias("w_total"),
        *[(pl.col("w") * pl.col(c)).sum().alias(f"w_{c}") for c in ALL_COMPONENTS],
    )
    # League per-game mean rates by position, from the same window (no leakage).
    league = window.group_by("position_group").agg(
        *[
            (pl.col(c).sum() / pl.col("games").sum()).alias(f"lg_{c}")
            for c in ALL_COMPONENTS
        ]
    )
    shrink = (
        pl.when(pl.col("position_group") == "DST")
        .then(DST_SHRINK_GAMES)
        .otherwise(K_SHRINK_GAMES)
    )
    rates = weighted.join(league, on="position_group", how="left").with_columns(
        [
            (
                (pl.col(f"w_{c}") + shrink * pl.col(f"lg_{c}"))
                / (pl.col("w_games") + shrink)
            ).alias(f"r_{c}")
            for c in ALL_COMPONENTS
        ]
    )
    # Renormalize PA bucket rates to one game's worth of probability (DST only).
    bucket_sum = sum((pl.col(f"r_{c}") for c in PA_BUCKETS), pl.lit(0.0))
    rates = rates.with_columns(
        [
            pl.when(pl.col("position_group") == "DST")
            .then(pl.col(f"r_{c}") / bucket_sum)
            .otherwise(pl.col(f"r_{c}"))
            .alias(f"r_{c}")
            for c in PA_BUCKETS
        ]
    )
    projected_games = (
        pl.when(pl.col("position_group") == "DST")
        .then(DST_GAMES)
        .otherwise(
            (
                0.85 * (pl.col("w_games") / pl.col("w_total"))
                + 0.15 * KICKER_DEFAULT_GAMES
            ).clip(1.0, 17.0)
        )
    )
    return (
        rates.with_columns(projected_games.alias("projected_games"))
        .with_columns(
            [
                (pl.col(f"r_{c}") * pl.col("projected_games")).alias(c)
                for c in ALL_COMPONENTS
            ]
        )
        .with_columns(
            pl.lit(target_season).cast(pl.Int32).alias("season"),
            pl.lit(False).alias("is_rookie"),
        )
        .select(
            "player_id",
            "player_name",
            "position",
            "position_group",
            "team",
            "season",
            "is_rookie",
            "projected_games",
            *ALL_COMPONENTS,
        )
    )
