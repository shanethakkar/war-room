"""Positional tiers (design.md §5).

PROVISIONAL v1: gap-based. Within a position, sorted by VOR, a new tier starts
where the drop to the next player exceeds ``TIER_GAP`` points. This is a
transparent stand-in.

The intended method is **distribution-overlap tiers**: a tier is a set of players
whose credible intervals overlap enough that pick order barely matters. That needs
the uncertainty layer (per-player intervals), which is not built yet - so this
gap heuristic is explicitly a placeholder to be swapped once intervals exist.
"""

from __future__ import annotations

import polars as pl

# VOR points; a bigger gap than this to the next player starts a new tier.
TIER_GAP: float = 10.0


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
