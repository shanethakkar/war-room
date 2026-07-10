"""Replacement levels and Value Over Replacement (design.md §7, §6.2).

VOR = projected points minus the positional **replacement level**, where the
replacement level is set by the format's roster. The subtlety is FLEX (RB/WR/TE)
and SUPERFLEX (QB/RB/WR/TE): those slots are filled greedily by the best players
still available, which is what dynamically raises the number of startable QBs in
superflex and pushes replacement-level QB far down - so elite QBs earn the value
they should. No special-casing; just correct allocation against the right roster.

Replacement level for a position is the best NON-starter at that position (the
player you would stream), so the marginal starter sits just above zero VOR.
"""

from __future__ import annotations

import polars as pl

from src.formats.base import RosterConfig

SKILL: tuple[str, ...] = ("QB", "RB", "WR", "TE")
FLEX_ELIGIBLE: tuple[str, ...] = ("RB", "WR", "TE")
SUPERFLEX_ELIGIBLE: tuple[str, ...] = ("QB", "RB", "WR", "TE")


def _dedicated_counts(roster: RosterConfig) -> pl.DataFrame:
    """League-wide dedicated starter count per position (teams x per-team slots)."""
    t = roster.teams
    return pl.DataFrame(
        {
            "position_group": list(SKILL),
            "_base": [t * roster.qb, t * roster.rb, t * roster.wr, t * roster.te],
        }
    )


def _fill_slots(
    df: pl.DataFrame, eligible: tuple[str, ...], taken: str, n_slots: int, out: str
) -> pl.DataFrame:
    """Flag the top ``n_slots`` still-available eligible players as starters.

    ``taken`` is a boolean column marking players already used by an earlier slot
    type; only eligible, not-yet-taken players compete for these slots.
    """
    candidate_points = (
        pl.when(pl.col("position_group").is_in(eligible) & ~pl.col(taken))
        .then(pl.col("projected_points"))
        .otherwise(None)
    )
    rank = candidate_points.rank("ordinal", descending=True)
    return df.with_columns((rank <= n_slots).fill_null(False).alias(out))


def add_starter_flags(scored: pl.DataFrame, roster: RosterConfig) -> pl.DataFrame:
    """Mark each skill player as a league-wide starter via dedicated+flex+superflex."""
    df = scored.filter(pl.col("position_group").is_in(SKILL)).join(
        _dedicated_counts(roster), on="position_group", how="left"
    )
    df = df.with_columns(
        pl.col("projected_points")
        .rank("ordinal", descending=True)
        .over("position_group")
        .alias("_pos_rank")
    ).with_columns((pl.col("_pos_rank") <= pl.col("_base")).alias("_dedicated"))

    df = _fill_slots(
        df, FLEX_ELIGIBLE, "_dedicated", roster.teams * roster.flex, "_flex"
    )
    df = df.with_columns((pl.col("_dedicated") | pl.col("_flex")).alias("_thru_flex"))
    df = _fill_slots(
        df,
        SUPERFLEX_ELIGIBLE,
        "_thru_flex",
        roster.teams * roster.superflex,
        "_sflex",
    )
    return df.with_columns(
        (pl.col("_dedicated") | pl.col("_flex") | pl.col("_sflex")).alias("is_starter")
    )


def add_vor(df: pl.DataFrame) -> pl.DataFrame:
    """Attach replacement level (best non-starter per position) and VOR."""
    replacement = (
        pl.when(~pl.col("is_starter"))
        .then(pl.col("projected_points"))
        .otherwise(None)
        .max()
        .over("position_group")
    )
    return df.with_columns(replacement.alias("replacement_level")).with_columns(
        (pl.col("projected_points") - pl.col("replacement_level")).alias("vor")
    )
