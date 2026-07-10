"""Transparent baseline projection engine (design.md §4.1-4.4).

Projects a component stat line for every skill player for a target season, using
only data from prior seasons (leakage-free). The method is deliberately readable:

1. **Recent form** - exponentially recency-weighted sums of each player's last
   ``LOOKBACK`` seasons.
2. **Volume** - project per-game targets / carries / pass attempts, lightly
   shrunk toward the position mean (volume is sticky).
3. **Efficiency** - per-opportunity rates shrunk *hard* toward the position role
   mean, weighted by the player's own opportunity sample (partial pooling).
4. **Touchdowns** - projected from shrunk *expected*-TD rates, never raw
   prior-year TDs.
5. **Rookies** - draft-capital priors (position x draft round, with a
   position-level fallback), since they have no NFL history.

Component totals = projected per-game volume x projected games x projected rate.
Scoring into fantasy points per format happens downstream (``formats.score``); the
projection itself is format-agnostic.

Regression constants are in opportunity units and are TUNABLE - the backtest is
the scoreboard. Aging curves are deliberately NOT applied yet: an unvalidated age
adjustment must not ship before the backtest can prove it helps.
"""

from __future__ import annotations

import polars as pl

from src.projections.baseline.priors import (
    SKILL_POSITION_GROUPS,
    UDFA_ROUND,
    positional_priors,
    rookie_priors,
    rookie_priors_by_position,
    td_signal_exprs,
)

# Recency weighting for a player's own history.
DECAY: float = 0.7
LOOKBACK: int = 2

# Efficiency shrinkage constants, in opportunity units. Larger = more regression
# to the role mean. TD rates and QB rates are noisier, so they regress harder.
K_CATCH_RATE: float = 45.0
K_YARDS_PER_TARGET: float = 45.0
K_YARDS_PER_CARRY: float = 55.0
K_YARDS_PER_ATT: float = 150.0
K_COMPLETION: float = 150.0
K_REC_TD: float = 55.0
K_RUSH_TD: float = 70.0
K_PASS_TD: float = 180.0
K_INT: float = 160.0
K_FUMBLE: float = 120.0
K_VOLUME: float = 3.0  # per-game volume shrink, in games units (light)

# Games projection: own recent average shrunk gently toward a durability prior.
DURABILITY_GAMES: float = 16.0
DURABILITY_WEIGHT: float = 0.15
ROOKIE_DEFAULT_GAMES: float = 14.0

# Standard intermediate rate/volume columns both projection paths produce before
# component totals are computed.
_RATE_COLS: tuple[str, ...] = (
    "tgt_pg",
    "car_pg",
    "patt_pg",
    "catch_rate",
    "ypt",
    "rec_td_rate",
    "ypc",
    "rush_td_rate",
    "comp_pct",
    "ypa",
    "pass_td_rate",
    "int_rate",
    "fumble_rate",
)

OUTPUT_COLS: tuple[str, ...] = (
    "player_id",
    "player_name",
    "position",
    "position_group",
    "team",
    "season",
    "is_rookie",
    "age",
    "projected_games",
    "pass_attempts",
    "pass_completions",
    "pass_yards",
    "pass_tds",
    "interceptions",
    "carries",
    "rush_yards",
    "rush_tds",
    "targets",
    "receptions",
    "rec_yards",
    "rec_tds",
    "fumbles_lost",
)


def _shrunk(w_num: str, prior: str, w_den: str, k: float) -> pl.Expr:
    """Partial-pooling estimate: (weighted_num + k*prior_rate) / (weighted_den + k).

    Equivalent to blending the player's own opportunity-weighted rate with the
    role mean, where ``k`` opportunities of the prior are added. High-sample
    players barely move; thin-sample players collapse toward the role mean.
    """
    return (pl.col(w_num) + k * pl.col(prior)) / (pl.col(w_den) + k)


def recent_form(panel: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Recency-weighted sums of each player's last ``LOOKBACK`` seasons before Y."""
    window = panel.filter(
        (pl.col("season") < target_season)
        & (pl.col("season") >= target_season - LOOKBACK)
    ).with_columns(
        (DECAY ** (target_season - 1 - pl.col("season"))).alias("w"),
        *td_signal_exprs(),
    )

    def ws(column: str) -> pl.Expr:
        return (pl.col("w") * pl.col(column)).sum()

    weighted = window.group_by("player_id").agg(
        pl.col("w").sum().alias("w_total"),
        ws("games").alias("w_games"),
        ws("targets").alias("w_targets"),
        ws("receptions").alias("w_receptions"),
        ws("receiving_yards").alias("w_rec_yards"),
        ws("rec_td_signal").alias("w_rec_td"),
        ws("carries").alias("w_carries"),
        ws("rushing_yards").alias("w_rush_yards"),
        ws("rush_td_signal").alias("w_rush_td"),
        ws("pass_attempts").alias("w_pass_att"),
        ws("pass_completions").alias("w_pass_comp"),
        ws("passing_yards").alias("w_pass_yards"),
        ws("pass_td_signal").alias("w_pass_td"),
        ws("interceptions").alias("w_int"),
        ws("fumbles_lost").alias("w_fum"),
    )
    latest = (
        window.sort(["player_id", "season"])
        .group_by("player_id", maintain_order=True)
        .last()
        .select(
            "player_id",
            "player_name",
            "position",
            "position_group",
            "team",
            pl.col("season").alias("last_season"),
            "age",
        )
    )
    return weighted.join(latest, on="player_id", how="left")


def _fill_rates(df: pl.DataFrame) -> pl.DataFrame:
    """Null rate -> 0 contribution (no signal for that component means zero)."""
    return df.with_columns(pl.col(c).fill_null(0.0) for c in _RATE_COLS)


def _apply_component_totals(df: pl.DataFrame) -> pl.DataFrame:
    """Per-game volume x games x rate -> projected season component totals."""
    df = df.with_columns(
        (pl.col("tgt_pg") * pl.col("proj_games")).clip(0.0).alias("targets"),
        (pl.col("car_pg") * pl.col("proj_games")).clip(0.0).alias("carries"),
        (pl.col("patt_pg") * pl.col("proj_games")).clip(0.0).alias("pass_attempts"),
    )
    return df.with_columns(
        (pl.col("targets") * pl.col("catch_rate")).alias("receptions"),
        (pl.col("targets") * pl.col("ypt")).alias("rec_yards"),
        (pl.col("targets") * pl.col("rec_td_rate")).alias("rec_tds"),
        (pl.col("carries") * pl.col("ypc")).alias("rush_yards"),
        (pl.col("carries") * pl.col("rush_td_rate")).alias("rush_tds"),
        (pl.col("pass_attempts") * pl.col("comp_pct")).alias("pass_completions"),
        (pl.col("pass_attempts") * pl.col("ypa")).alias("pass_yards"),
        (pl.col("pass_attempts") * pl.col("pass_td_rate")).alias("pass_tds"),
        (pl.col("pass_attempts") * pl.col("int_rate")).alias("interceptions"),
        ((pl.col("carries") + pl.col("targets")) * pl.col("fumble_rate")).alias(
            "fumbles_lost"
        ),
    )


def _finalize(df: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Stamp the season, rename projected games, and select the output schema."""
    return (
        df.with_columns(pl.lit(target_season).cast(pl.Int32).alias("season"))
        .rename({"proj_games": "projected_games"})
        .select(OUTPUT_COLS)
    )


def project_returning(
    form: pl.DataFrame, priors: pl.DataFrame, target_season: int
) -> pl.DataFrame:
    """Project returning players from recent form shrunk toward positional priors."""
    df = form.join(priors, on="position_group", how="left").with_columns(
        # per-game volume, lightly shrunk (w_x already equals w_games * own_pg)
        (
            (pl.col("w_targets") + K_VOLUME * pl.col("targets_per_game"))
            / (pl.col("w_games") + K_VOLUME)
        ).alias("tgt_pg"),
        (
            (pl.col("w_carries") + K_VOLUME * pl.col("carries_per_game"))
            / (pl.col("w_games") + K_VOLUME)
        ).alias("car_pg"),
        (
            (pl.col("w_pass_att") + K_VOLUME * pl.col("pass_attempts_per_game"))
            / (pl.col("w_games") + K_VOLUME)
        ).alias("patt_pg"),
        # efficiency, shrunk hard toward the role mean
        _shrunk("w_receptions", "catch_rate", "w_targets", K_CATCH_RATE).alias(
            "catch_rate"
        ),
        _shrunk(
            "w_rec_yards", "yards_per_target", "w_targets", K_YARDS_PER_TARGET
        ).alias("ypt"),
        _shrunk(
            "w_rush_yards", "yards_per_carry", "w_carries", K_YARDS_PER_CARRY
        ).alias("ypc"),
        _shrunk(
            "w_pass_yards", "yards_per_pass_attempt", "w_pass_att", K_YARDS_PER_ATT
        ).alias("ypa"),
        _shrunk("w_pass_comp", "completion_pct", "w_pass_att", K_COMPLETION).alias(
            "comp_pct"
        ),
        _shrunk("w_rec_td", "rec_td_rate", "w_targets", K_REC_TD).alias("rec_td_rate"),
        _shrunk("w_rush_td", "rush_td_rate", "w_carries", K_RUSH_TD).alias(
            "rush_td_rate"
        ),
        _shrunk("w_pass_td", "pass_td_rate", "w_pass_att", K_PASS_TD).alias(
            "pass_td_rate"
        ),
        _shrunk("w_int", "interception_rate", "w_pass_att", K_INT).alias("int_rate"),
        (
            (pl.col("w_fum") + K_FUMBLE * pl.col("fumble_rate"))
            / (pl.col("w_carries") + pl.col("w_targets") + K_FUMBLE)
        ).alias("fumble_rate"),
        # projected games: own recent average, shrunk toward the durability prior
        (
            (1 - DURABILITY_WEIGHT) * (pl.col("w_games") / pl.col("w_total"))
            + DURABILITY_WEIGHT * DURABILITY_GAMES
        )
        .clip(1.0, 17.0)
        .alias("proj_games"),
        # projected age = last observed age carried forward to the target season
        (pl.col("age") + (target_season - pl.col("last_season"))).alias("age"),
        pl.lit(False).alias("is_rookie"),
    )
    df = _apply_component_totals(_fill_rates(df))
    return _finalize(df, target_season)


def project_rookies(
    target_season: int,
    players: pl.DataFrame,
    round_priors: pl.DataFrame,
    position_priors: pl.DataFrame,
    exclude_ids: list[str],
) -> pl.DataFrame:
    """Project incoming rookies from draft-capital priors (design.md §4.4).

    Rookies are identified from the static ``players`` reference (leakage-free:
    draft round is known pre-season). Each gets the position x draft-round rookie
    prior, falling back to the position-level prior for thin/empty buckets.
    """
    standard = ["games_per_rookie", *_RATE_COLS]
    # rookie priors don't model fumbles; rookies get a 0 fumble rate (minor).
    round_std = [c for c in standard if c != "fumble_rate"]

    rookies = (
        players.filter(
            (pl.col("rookie_season") == target_season)
            & (pl.col("position_group").is_in(SKILL_POSITION_GROUPS))
        )
        .filter(~pl.col("gsis_id").is_in(exclude_ids))
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player_name"),
            pl.col("position"),
            pl.col("position_group"),
            pl.col("draft_round").fill_null(UDFA_ROUND).alias("draft_round"),
        )
    )
    joined = rookies.join(
        round_priors.select(["position_group", "draft_round", *round_std]),
        on=["position_group", "draft_round"],
        how="left",
    ).join(
        position_priors.select(["position_group", *round_std]),
        on="position_group",
        how="left",
        suffix="_pos",
    )
    # Prefer the draft-round bucket; fall back to the position-level prior.
    coalesced = joined.with_columns(
        pl.coalesce(c, f"{c}_pos").alias(c) for c in round_std
    ).with_columns(
        pl.coalesce("games_per_rookie", pl.lit(ROOKIE_DEFAULT_GAMES))
        .clip(1.0, 17.0)
        .alias("proj_games"),
        pl.lit(None, dtype=pl.Float64).alias("age"),
        pl.lit(True).alias("is_rookie"),
        pl.lit(0.0).alias("fumble_rate"),
        pl.lit(None, dtype=pl.String).alias("team"),
    )
    result = _apply_component_totals(_fill_rates(coalesced))
    return _finalize(result, target_season)


def project_season(
    panel: pl.DataFrame, players: pl.DataFrame, target_season: int
) -> pl.DataFrame:
    """Full baseline projection for ``target_season``: returning players + rookies.

    Uses only seasons strictly before ``target_season`` for all learned priors and
    recent form (leakage-free per design.md §8).
    """
    train = panel.filter(pl.col("season") < target_season)
    if train.height == 0:
        raise ValueError(
            f"No training data before season {target_season}; cannot project."
        )

    returning = project_returning(
        recent_form(panel, target_season), positional_priors(train), target_season
    )
    rookies = project_rookies(
        target_season,
        players,
        rookie_priors(train),
        rookie_priors_by_position(train),
        returning["player_id"].to_list(),
    )
    return pl.concat([returning, rookies], how="vertical").sort(["season", "player_id"])
