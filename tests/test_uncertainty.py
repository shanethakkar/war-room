"""Uncertainty-layer tests - network-free, synthetic residuals.

Covers the fitted per-(position, projection-tier) scaled-residual quantiles,
interval attachment (scaling with projection + floor + clip), and in-sample
coverage. The real fit over historical projections is exercised by
``python -m src.projections.uncertainty``.
"""

from __future__ import annotations

import polars as pl
import pytest
from src.projections.uncertainty import (
    FLOOR,
    add_intervals,
    fit_interval_model,
    in_sample_coverage,
)


def _residuals(position: str, projected: float, zs: list[float]) -> pl.DataFrame:
    """Residuals whose z = residual / max(projected, FLOOR) equals ``zs`` exactly."""
    scale = max(projected, FLOOR)
    return pl.DataFrame(
        {
            "position_group": [position] * len(zs),
            "projected_points": [projected] * len(zs),
            "residual": [z * scale for z in zs],
        }
    )


def _single_tier_model(
    z_low: float, z_median: float, z_high: float, position: str = "WR"
) -> pl.DataFrame:
    """A one-bucket model (cuts at 0 -> every projection lands in tier 1)."""
    return pl.DataFrame(
        {
            "position_group": [position],
            "proj_tier": pl.Series([1], dtype=pl.Int32),
            "z_low": [z_low],
            "z_median": [z_median],
            "z_high": [z_high],
            "c_hi": [0.0],
            "c_lo": [0.0],
        }
    )


def test_fit_quantiles_ordered_and_carry_cuts() -> None:
    zs = [x / 10 for x in range(-5, 6)]  # -0.5 .. 0.5
    # Varied projections so tier buckets actually form.
    residuals = pl.concat([_residuals("WR", p, zs) for p in (50.0, 120.0, 260.0)])
    model = fit_interval_model(residuals)
    assert {"proj_tier", "c_hi", "c_lo"} <= set(model.columns)
    for row in model.iter_rows(named=True):
        assert row["z_low"] < row["z_median"] < row["z_high"]


def test_add_intervals_scales_with_projection() -> None:
    model = _single_tier_model(-0.5, 0.0, 0.5)
    scored = pl.DataFrame(
        {"position_group": ["WR", "WR"], "projected_points": [100.0, 300.0]}
    )
    out = add_intervals(scored, model)
    width = dict(
        zip(
            out["projected_points"], out["points_high"] - out["points_low"], strict=True
        )
    )
    assert width[100.0] == pytest.approx(100.0)
    assert width[300.0] == pytest.approx(300.0)  # wider for bigger projection


def test_add_intervals_floor_prevents_tiny_scale() -> None:
    model = _single_tier_model(-0.5, 0.0, 0.5)
    scored = pl.DataFrame({"position_group": ["WR"], "projected_points": [15.0]})
    out = add_intervals(scored, model)
    width = out["points_high"][0] - out["points_low"][0]
    assert width == pytest.approx(FLOOR)  # scale floored at 20, not 15


def test_points_low_clipped_at_zero() -> None:
    model = _single_tier_model(-2.0, 0.0, 0.5)
    scored = pl.DataFrame({"position_group": ["WR"], "projected_points": [30.0]})
    out = add_intervals(scored, model)
    assert out["points_low"][0] == 0.0  # 30 + (-2)*30 = -30 -> clipped


def test_in_sample_coverage_near_target() -> None:
    zs = [x / 10 for x in range(-5, 6)]
    residuals = pl.concat([_residuals("WR", p, zs) for p in (60.0, 140.0, 260.0)])
    model = fit_interval_model(residuals)
    cov = in_sample_coverage(residuals, model).row(0, named=True)
    assert 0.6 <= cov["coverage"] <= 1.0  # empirical 80% quantiles on the same data
