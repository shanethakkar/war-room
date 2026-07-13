"""Board service for the API - assembles what the frontend renders.

For a (season, format) it builds the baseline value board (VOR, tiers, projected
points + 80% interval) and ranks it with the **market-anchored blend**
(`decision.blend`): ADP anchor + validated model tilt, the configuration that
actually wins simulated drafts. The pure-model order stays available as
``model_rank`` so the UI can show both views. Results are cached per
(season, format); everything is offline except the one-time ADP fetch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl

from src.decision.blend import (
    BASE_WEIGHT,
    BAYES_WEIGHT,
    blend_with_market,
)
from src.decision.board import build_value_board
from src.formats import FormatConfig
from src.ingest.adp import load_adp
from src.ingest.cache import read_table
from src.projections.pipeline import scored_projection

DEFAULT_MODEL = "baseline"

_OUTPUT_COLS = (
    "board_rank",
    "model_rank",
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
    "adp_stdev",
    "adp_market_rank",
    "model_tilt",
)


def _bayes_aux(
    panel: pl.DataFrame, players: pl.DataFrame, season: int, fmt: FormatConfig
) -> pl.DataFrame | None:
    """Bayesian board scores for the 3-way ensemble; None if pymc is unavailable."""
    try:
        scored = scored_projection(
            panel, players, season, model="bayesian", fmt_key=fmt
        )
        board = build_value_board(scored, fmt)
        return board.select("player_id", pl.col("vor").alias("aux_vor"))
    except ImportError:
        return None


def _ranked_board(
    board: pl.DataFrame,
    aux: pl.DataFrame | None,
    season: int,
    fmt: FormatConfig,
) -> pl.DataFrame:
    """Blend the board with market ADP; degrade gracefully if ADP is unreachable.

    With the bayesian aux available, uses the 3-way ensemble (0.20 baseline /
    0.10 bayesian / 0.70 ADP - the fresh-seed-confirmed winner); otherwise the
    LOSO-validated 2-way (0.30 / 0.70).
    """
    board = board.rename({"overall_rank": "model_rank"})
    try:
        adp = load_adp(season, fmt, teams=fmt.roster.teams)
    except (RuntimeError, OSError):
        return board.with_columns(
            pl.col("model_rank").cast(pl.Int64).alias("board_rank"),
            pl.lit(None, dtype=pl.Float64).alias("adp"),
            pl.lit(None, dtype=pl.Int64).alias("adp_market_rank"),
            pl.lit(None, dtype=pl.Int64).alias("model_tilt"),
        )
    if aux is not None:
        return blend_with_market(
            board, adp, model_weight=BASE_WEIGHT, aux=aux, aux_weight=BAYES_WEIGHT
        )
    return blend_with_market(board, adp)


@lru_cache(maxsize=64)
def get_board(season: int, fmt: FormatConfig) -> list[dict[str, Any]]:
    """The blended board for a (season, resolved format) - JSON-ready rows.

    ``fmt`` is the fully-resolved (possibly customized) config; being a frozen
    dataclass it doubles as the cache key, so every distinct league setup gets
    its own cached board.
    """
    panel = read_table("feature_panel")
    players = read_table("players")
    scored = scored_projection(panel, players, season, model=DEFAULT_MODEL, fmt_key=fmt)
    aux = _bayes_aux(panel, players, season, fmt)
    board = _ranked_board(build_value_board(scored, fmt), aux, season, fmt)
    present = [c for c in _OUTPUT_COLS if c in board.columns]
    return (
        board.select(present)
        .sort("board_rank")
        .with_columns(pl.col(pl.Float64).round(1))
        .to_dicts()
    )
