"""Empirical uncertainty layer (design.md §5).

Consensus gives a number; we give a distribution - this is the product. The
baseline attaches an empirical prediction interval to every projection, estimated
from **leakage-free historical residuals**:

1. Project each past season Y from data strictly before Y, score it, and compare
   to what actually happened (residual = actual - projected).
2. Model the spread as quantiles of a **projection-scaled residual**
   ``z = residual / max(projected, FLOOR)``, bucketed by position AND projection
   tier. Scaling captures heteroscedasticity; the projection-tier buckets capture
   that an elite player is *relatively* more predictable than a fringe one (a
   single pooled z gives elite players an absurd near-zero floor). Empirical
   quantiles capture skew (fantasy is boom/bust).
3. For a new projection ``p`` in position g, look up its tier and set the interval
   to ``p + z_q * max(p, FLOOR)``.

The fitted model is a small per-(position, projection-tier) table, cached to
``interval_model``. Forward calibration is measured by the backtest (design.md
§8); here we report only in-sample coverage as a plumbing check.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import polars as pl

from src.config import DATA_START_SEASON
from src.formats import get_format
from src.formats.base import ScoringConfig
from src.formats.score import score_components, score_special
from src.ingest.cache import read_table, write_table
from src.projections.baseline.project import LOOKBACK, project_season
from src.projections.special import project_special

# v2: includes K/DST residuals (bumped so a stale offense-only cache is not reused).
MODEL_NAME = "interval_model_v2"

# Softening floor (PPR points): below this projection, spread is scaled by FLOOR
# rather than the (tiny, noisy) projection itself.
FLOOR: float = 20.0
# 80% central interval by default.
LOW_Q: float = 0.10
HIGH_Q: float = 0.90
# Projection-tier cut points within a position (top / middle / bottom third).
_TIER_HI_Q: float = 2 / 3
_TIER_LO_Q: float = 1 / 3

# Residuals are scored in PPR; superflex uses the same scoring (only its roster
# differs), so one residual model serves both formats.
_PPR_SCORING: ScoringConfig = get_format("redraft_ppr").scoring


def _scale(points: str = "projected_points") -> pl.Expr:
    """max(projection, FLOOR) - the denominator that makes spread proportional."""
    return pl.max_horizontal(pl.col(points), pl.lit(FLOOR))


def _proj_tier(hi: str = "c_hi", lo: str = "c_lo") -> pl.Expr:
    """Projection tier 1/2/3 (top/middle/bottom third by projected points)."""
    return (
        pl.when(pl.col("projected_points") >= pl.col(hi))
        .then(1)
        .when(pl.col("projected_points") >= pl.col(lo))
        .then(2)
        .otherwise(3)
        .alias("proj_tier")
    )


def default_seasons(panel: pl.DataFrame) -> list[int]:
    """Seasons projectable leakage-free AND having actuals (skip the first LOOKBACK)."""
    available = sorted(panel["season"].unique().to_list())
    return [y for y in available if y >= DATA_START_SEASON + LOOKBACK]


def collect_residuals(
    panel: pl.DataFrame, players: pl.DataFrame, seasons: Sequence[int]
) -> pl.DataFrame:
    """Leakage-free (projected, actual) pairs across ``seasons``.

    Each season is projected from data strictly before it, scored in PPR, and
    joined to that season's actual PPR points for players who actually played.
    """
    frames: list[pl.DataFrame] = []
    for year in seasons:
        projection = score_components(
            project_season(panel, players, year), _PPR_SCORING
        ).select("player_id", "position_group", "projected_points")
        actual = panel.filter(
            (pl.col("season") == year) & (pl.col("games") >= 1)
        ).select("player_id", pl.col("fantasy_points_ppr").alias("actual_points"))
        frames.append(
            projection.join(actual, on="player_id", how="inner").with_columns(
                pl.lit(year).cast(pl.Int32).alias("season"),
                (pl.col("actual_points") - pl.col("projected_points")).alias(
                    "residual"
                ),
            )
        )
    return pl.concat(frames, how="vertical")


def collect_special_residuals(
    special_panel: pl.DataFrame, seasons: Sequence[int]
) -> pl.DataFrame:
    """Leakage-free (projected, actual) pairs for K/DST across ``seasons``.

    K/DST scoring is PPR-agnostic, so the default scoring config serves as the
    reference (mirroring the PPR reference for offense).
    """
    frames: list[pl.DataFrame] = []
    for year in seasons:
        projection = score_special(
            project_special(special_panel, year), _PPR_SCORING
        ).select("player_id", "position_group", "projected_points")
        actual = score_special(
            special_panel.filter(pl.col("season") == year), _PPR_SCORING, "actual"
        ).select("player_id", "actual")
        frames.append(
            projection.join(actual, on="player_id", how="inner")
            .with_columns(
                pl.lit(year).cast(pl.Int32).alias("season"),
                (pl.col("actual") - pl.col("projected_points")).alias("residual"),
            )
            .drop("actual")
        )
    return pl.concat(frames, how="vertical")


def _position_cuts(residuals: pl.DataFrame) -> pl.DataFrame:
    """Per-position projected-points thresholds separating the three tiers."""
    return residuals.group_by("position_group").agg(
        pl.col("projected_points").quantile(_TIER_HI_Q).alias("c_hi"),
        pl.col("projected_points").quantile(_TIER_LO_Q).alias("c_lo"),
    )


def fit_interval_model(
    residuals: pl.DataFrame, low_q: float = LOW_Q, high_q: float = HIGH_Q
) -> pl.DataFrame:
    """Scaled-residual quantiles per (position, projection tier), with the cuts.

    The cut points (``c_hi`` / ``c_lo``) are stored so a new projection can be
    assigned to the same tier at inference time.
    """
    cuts = _position_cuts(residuals)
    z = residuals.join(cuts, on="position_group").with_columns(
        (pl.col("residual") / _scale()).alias("z"), _proj_tier()
    )
    return z.group_by("position_group", "proj_tier").agg(
        pl.col("z").quantile(low_q).alias("z_low"),
        pl.col("z").quantile(0.5).alias("z_median"),
        pl.col("z").quantile(high_q).alias("z_high"),
        pl.col("c_hi").first().alias("c_hi"),
        pl.col("c_lo").first().alias("c_lo"),
        pl.len().alias("n_residuals"),
    )


def add_intervals(scored: pl.DataFrame, model: pl.DataFrame) -> pl.DataFrame:
    """Attach ``points_low`` / ``points_median`` / ``points_high`` to a projection."""
    cuts = model.group_by("position_group").agg(
        pl.col("c_hi").first(), pl.col("c_lo").first()
    )
    scale = _scale()
    return (
        scored.join(cuts, on="position_group", how="left")
        .with_columns(_proj_tier())
        .join(
            model.select("position_group", "proj_tier", "z_low", "z_median", "z_high"),
            on=["position_group", "proj_tier"],
            how="left",
        )
        .with_columns(
            (pl.col("projected_points") + pl.col("z_low") * scale)
            .clip(0.0)
            .alias("points_low"),
            (pl.col("projected_points") + pl.col("z_median") * scale).alias(
                "points_median"
            ),
            (pl.col("projected_points") + pl.col("z_high") * scale).alias(
                "points_high"
            ),
        )
        .drop("c_hi", "c_lo", "proj_tier", "z_low", "z_median", "z_high")
    )


def in_sample_coverage(residuals: pl.DataFrame, model: pl.DataFrame) -> pl.DataFrame:
    """Fraction of actuals inside the interval, per position (should ~= high-low)."""
    scored = add_intervals(
        residuals.with_columns(
            (pl.col("projected_points") + pl.col("residual")).alias("actual")
        ),
        model,
    )
    covered = scored.with_columns(
        (
            (pl.col("actual") >= pl.col("points_low"))
            & (pl.col("actual") <= pl.col("points_high"))
        ).alias("covered")
    )
    return covered.group_by("position_group").agg(
        pl.col("covered").mean().alias("coverage"), pl.len().alias("n")
    )


def load_or_fit_interval_model(
    panel: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    """Return the cached interval model, fitting and caching it on first use."""
    try:
        return read_table(MODEL_NAME)
    except FileNotFoundError:
        seasons = default_seasons(panel)
        residuals = _all_residuals(panel, players, seasons)
        model = fit_interval_model(residuals)
        write_table(MODEL_NAME, model)
        return model


def _all_residuals(
    panel: pl.DataFrame, players: pl.DataFrame, seasons: list[int]
) -> pl.DataFrame:
    """Offense + K/DST residuals, aligned to the fit's input schema."""
    offense = collect_residuals(panel, players, seasons).select(
        "player_id", "position_group", "projected_points", "season", "residual"
    )
    try:
        special_panel = read_table("special_panel")
    except FileNotFoundError:  # cache predates DST/K; offense-only intervals
        return offense
    special = collect_special_residuals(special_panel, seasons).select(
        "player_id", "position_group", "projected_points", "season", "residual"
    )
    return pl.concat([offense, special], how="vertical")


def main() -> None:
    argparse.ArgumentParser(
        description="Fit the empirical interval model from historical residuals."
    ).parse_args()

    panel = read_table("feature_panel")
    players = read_table("players")
    seasons = default_seasons(panel)
    print(f"[uncertainty] collecting residuals for seasons {seasons[0]}-{seasons[-1]}")
    residuals = _all_residuals(panel, players, seasons)
    model = fit_interval_model(residuals)
    write_table(MODEL_NAME, model)

    pct = int(round((HIGH_Q - LOW_Q) * 100))
    print(
        f"[uncertainty] fit on {residuals.height:,} residuals; {pct}% interval "
        f"z-quantiles by position x projection-tier (1=elite):"
    )
    for row in model.sort(["position_group", "proj_tier"]).to_dicts():
        print(
            f"    {row['position_group']:3} t{row['proj_tier']}  "
            f"z_low={row['z_low']:+.2f}  z_high={row['z_high']:+.2f}  "
            f"(n={row['n_residuals']:,})"
        )
    print(f"[uncertainty] in-sample coverage (target ~{pct}%):")
    for row in in_sample_coverage(residuals, model).sort("position_group").to_dicts():
        print(f"    {row['position_group']:3}  {row['coverage']:.1%}  (n={row['n']:,})")


if __name__ == "__main__":
    main()
