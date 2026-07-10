"""Board service for the API - assembles what the frontend renders.

For a (season, format) it builds the baseline value board (VOR, tiers, projected
points + 80% interval) and merges ADP to expose the arbitrage delta (where we and
the market disagree). Results are cached per (season, format); everything is
computed offline from the cache except the one-time ADP fetch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl

from src.decision.board import build_value_board
from src.formats import get_format
from src.ingest.adp import load_adp
from src.ingest.cache import read_table
from src.names import norm_name_expr
from src.projections.pipeline import scored_projection

DEFAULT_MODEL = "baseline"

_OUTPUT_COLS = (
    "overall_rank",
    "position_rank",
    "position_tier",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "team",
    "is_rookie",
    "projected_games",
    "projected_points",
    "points_low",
    "points_high",
    "vor",
    "adp",
    "adp_market_rank",
    "arbitrage_delta",
)


def _with_arbitrage(board: pl.DataFrame, season: int, fmt_key: str) -> pl.DataFrame:
    """Attach adp / market rank / arbitrage delta to the board (best-effort)."""
    try:
        adp = load_adp(season, fmt_key).select("norm_name", "position", "adp")
    except (RuntimeError, OSError):
        return board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("adp"),
            pl.lit(None, dtype=pl.Int64).alias("adp_market_rank"),
            pl.lit(None, dtype=pl.Int64).alias("arbitrage_delta"),
        )

    merged = board.with_columns(norm_name_expr("player_name")).join(
        adp,
        left_on=["norm_name", "position_group"],
        right_on=["norm_name", "position"],
        how="left",
    )
    drafted = (
        merged.filter(pl.col("adp").is_not_null())
        .with_columns(
            pl.col("vor")
            .rank("ordinal", descending=True)
            .cast(pl.Int64)
            .alias("our_value_rank"),
            pl.col("adp").rank("ordinal").cast(pl.Int64).alias("adp_market_rank"),
        )
        .with_columns(
            (pl.col("adp_market_rank") - pl.col("our_value_rank")).alias(
                "arbitrage_delta"
            )
        )
        .select("player_id", "adp", "adp_market_rank", "arbitrage_delta")
    )
    return board.join(drafted, on="player_id", how="left")


@lru_cache(maxsize=32)
def get_board(season: int, fmt_key: str) -> list[dict[str, Any]]:
    """The full board (VOR + tiers + intervals + arbitrage) as JSON-ready rows."""
    panel = read_table("feature_panel")
    players = read_table("players")
    scored = scored_projection(
        panel, players, season, model=DEFAULT_MODEL, fmt_key=fmt_key
    )
    board = _with_arbitrage(
        build_value_board(scored, get_format(fmt_key)), season, fmt_key
    )
    present = [c for c in _OUTPUT_COLS if c in board.columns]
    return board.select(present).with_columns(pl.col(pl.Float64).round(1)).to_dicts()
