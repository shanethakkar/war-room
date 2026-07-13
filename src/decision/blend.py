"""Market-anchored board blending - the validated ranking.

The draft-sim showed the pure model board under-drafts ADP (win rate ~0.44 vs a
0.50 null), while **ADP anchored with a minority model tilt beats both** pure ADP
and the pure model. Classic forecast combination - error-prone rankings average
out each other's noise; fancier schemes (round-dependent, position-specific,
within-position reordering) all tested WORSE, so this stays deliberately simple:

    blend = W_MODEL * rank_model + W_AUX * rank_aux_model + (rest) * rank_adp

Sim-validated on 2019-2024 with the REALISTIC sim (DST/K slots, needs-aware
drafting for both sides, 15 rounds; see progress.md decisions log):
- 2-way (offense-only tilt): flat optimum w=0.10-0.20, LOSO 0.512 vs 0.507 null.
- 3-way (baseline 0.10 / bayesian 0.10 / ADP 0.80): 0.533 - the shipped default
  when the bayesian extra is available.
- DST/K rank purely by market: their model-ordering signal measured as noise
  (DST) or worse than market (K).
The edge is real but modest and year-dependent (2019/2024 at or below null).

Players without an ADP (deep rookies / free agents) are appended after the
matched pool, ordered by model VOR.
"""

from __future__ import annotations

import polars as pl

from src.names import norm_name_expr

# 2-way fallback weight (middle of the flat 0.10-0.20 optimum on the realistic
# sim; LOSO honest estimate 0.512 vs the 0.507 null).
MODEL_WEIGHT: float = 0.15
# 3-way ensemble weights (best on the realistic sim: 0.533; see progress.md).
BASE_WEIGHT: float = 0.10
BAYES_WEIGHT: float = 0.10
# Positions ranked purely by market: their model-ordering signal is noise
# (DST) or worse than market (K) - measured, not assumed. See progress.md.
MARKET_ONLY_POSITIONS: tuple[str, ...] = ("DST", "K")


def blend_with_market(
    board: pl.DataFrame,
    adp: pl.DataFrame,
    model_weight: float = MODEL_WEIGHT,
    aux: pl.DataFrame | None = None,
    aux_weight: float = 0.0,
) -> pl.DataFrame:
    """Rank the value board anchored to market ADP with a model tilt.

    ``board`` is a value board (needs ``player_id``/``player_name``/
    ``position_group``/``vor``); ``adp`` needs ``norm_name``/``position``/``adp``.
    ``aux`` optionally supplies a second model's scores (``player_id`` +
    ``aux_vor``) for the 3-way ensemble; players missing from ``aux`` fall back
    to the primary model's rank. Adds ``adp``, ``adp_market_rank``,
    ``board_rank`` (blended order, 1 = best) and ``model_tilt`` (market rank -
    board rank; positive = the models moved the player up from market).
    """
    adp_cols = ["norm_name", "position", "adp"] + (
        ["adp_stdev"] if "adp_stdev" in adp.columns else []
    )
    with_adp = board.with_columns(norm_name_expr("player_name")).join(
        adp.select(adp_cols),
        left_on=["norm_name", "position_group"],
        right_on=["norm_name", "position"],
        how="left",
    )
    if aux is not None:
        with_adp = with_adp.join(
            aux.select("player_id", "aux_vor"), on="player_id", how="left"
        )

    matched = with_adp.filter(pl.col("adp").is_not_null()).with_columns(
        pl.col("vor").rank("ordinal", descending=True).alias("_vr"),
        pl.col("adp").rank("ordinal").cast(pl.Int64).alias("adp_market_rank"),
    )
    if aux is not None and aux_weight > 0:
        matched = matched.with_columns(
            pl.col("aux_vor")
            .rank("ordinal", descending=True)
            .fill_null(pl.col("_vr"))
            .alias("_vr_aux")
        )
        blend_expr = (
            model_weight * pl.col("_vr")
            + aux_weight * pl.col("_vr_aux")
            + (1 - model_weight - aux_weight) * pl.col("adp_market_rank")
        )
    else:
        blend_expr = model_weight * pl.col("_vr") + (1 - model_weight) * pl.col(
            "adp_market_rank"
        )
    # DST/K rank purely by market; the model's ordering there is noise.
    blend_expr = (
        pl.when(pl.col("position_group").is_in(list(MARKET_ONLY_POSITIONS)))
        .then(pl.col("adp_market_rank").cast(pl.Float64))
        .otherwise(blend_expr)
    )

    matched = (
        matched.with_columns(blend_expr.alias("_blend"))
        .sort("_blend")
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("board_rank"))
        .with_columns(
            (pl.col("adp_market_rank") - pl.col("board_rank")).alias("model_tilt")
        )
        .drop(["_vr", "_blend", "_vr_aux"], strict=False)
    )

    unmatched = (
        with_adp.filter(pl.col("adp").is_null())
        .sort("vor", descending=True)
        .with_columns(
            (pl.int_range(0, pl.len(), dtype=pl.Int64) + matched.height + 1).alias(
                "board_rank"
            ),
            pl.lit(None, dtype=pl.Int64).alias("adp_market_rank"),
            pl.lit(None, dtype=pl.Int64).alias("model_tilt"),
        )
    )

    return pl.concat([matched, unmatched.select(matched.columns)], how="vertical").sort(
        "board_rank"
    )
