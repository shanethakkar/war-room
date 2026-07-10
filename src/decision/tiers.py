"""Positional tiers (design.md §5).

Two methods:

- ``add_overlap_tiers`` (preferred, used by the board): tiers from the prediction
  distributions. Naive interval overlap over-merges (season-long fantasy intervals
  are huge, so everyone touches everyone). Instead we ask the useful question -
  *is the next player meaningfully worse than this tier's best?* - by comparing the
  median gap to the spread. Walking a position from best to worst, a new tier
  starts when the anchor's median exceeds the player's median by more than
  ``TIER_SEP`` standard deviations of the anchor (complete-linkage on an anchor, so
  tiers don't chain). ``sigma`` is derived from the 80% interval width.

- ``add_position_tiers`` (fallback): gap-based on VOR, for the no-uncertainty path.
"""

from __future__ import annotations

import polars as pl

# VOR points; a bigger gap than this to the next player starts a new tier.
TIER_GAP: float = 10.0

# New tier when the anchor's median beats a player's median by > TIER_SEP sigma.
# TUNABLE: smaller = more, tighter tiers. 0.5 gives draft-actionable elite tiers
# (a handful per position) rather than one broad blob; validate against the board.
TIER_SEP: float = 0.5
# 10th-90th percentile spans ~2.563 std for a normal; converts width -> sigma.
_WIDTH_TO_STD: float = 2.563
# Floor on sigma so degenerate (near-zero-spread) players don't over-fragment.
_MIN_SIGMA: float = 1.0


def _tier_column(medians: list[float], sigmas: list[float]) -> pl.Series:
    """Anchor-based (complete-linkage) tier numbers over median-descending rows."""
    tiers: list[int] = []
    anchor_median = anchor_sigma = 0.0
    current = 0
    for i, (median, sigma) in enumerate(zip(medians, sigmas, strict=True)):
        if i == 0 or anchor_median - median > TIER_SEP * anchor_sigma:
            current += 1
            anchor_median, anchor_sigma = median, sigma
        tiers.append(current)
    return pl.Series("position_tier", tiers, dtype=pl.Int32)


def add_overlap_tiers(df: pl.DataFrame) -> pl.DataFrame:
    """Assign within-position tiers from the prediction distributions (design.md §5).

    Requires ``points_low`` / ``points_median`` / ``points_high`` (uncertainty layer).
    """
    with_sigma = df.with_columns(
        ((pl.col("points_high") - pl.col("points_low")) / _WIDTH_TO_STD)
        .clip(_MIN_SIGMA)
        .alias("_sigma")
    )
    parts: list[pl.DataFrame] = []
    for _key, group in with_sigma.group_by("position_group"):
        ordered = group.sort("points_median", descending=True)
        parts.append(
            ordered.with_columns(
                _tier_column(
                    ordered["points_median"].to_list(), ordered["_sigma"].to_list()
                )
            )
        )
    return pl.concat(parts).drop("_sigma")


def add_position_tiers(df: pl.DataFrame, gap: float = TIER_GAP) -> pl.DataFrame:
    """Assign a within-position tier number (1 = best) from VOR gaps."""
    ordered = df.sort(["position_group", "vor"], descending=[False, True])
    ordered = ordered.with_columns(
        (pl.col("vor").shift(1).over("position_group") - pl.col("vor")).alias("_gap")
    )
    return ordered.with_columns(
        (
            (pl.col("_gap") > gap)
            .fill_null(False)
            .cast(pl.Int32)
            .cum_sum()
            .over("position_group")
            + 1
        ).alias("position_tier")
    )
