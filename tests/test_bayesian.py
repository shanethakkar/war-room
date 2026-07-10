"""Bayesian model tests.

Feature construction is pure/fast; the PyMC fit+predict smoke is guarded by an
import skip (only runs where the optional ``bayesian`` extra is installed) and uses
tiny sampler settings.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from src.projections.bayesian.features import (
    PREDICTORS,
    projection_features,
    training_pairs,
)


def _panel_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "player_id": "a",
        "player_name": "Player A",
        "position": "WR",
        "position_group": "WR",
        "team": "SF",
        "season": 2020,
        "games": 16,
        "age": 24.0,
        "target_share": 0.2,
        "snap_share": 0.8,
        "fantasy_points_ppr": 200.0,
        "expected_fantasy_points": 190.0,
    }
    row.update(over)
    return row


def _panel(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))


def test_training_pairs_uses_prior_season_and_respects_cutoff() -> None:
    panel = _panel(
        [
            _panel_row(season=2020, fantasy_points_ppr=200.0, games=16),  # ppg 12.5
            _panel_row(season=2021, fantasy_points_ppr=240.0, games=16, age=25.0),
            _panel_row(season=2022, fantasy_points_ppr=180.0, games=15, age=26.0),
        ]
    )
    pairs = training_pairs(panel, before_season=2023)
    # Targets 2021 (prev 2020) and 2022 (prev 2021); 2020 has no prior -> excluded.
    assert set(pairs["target_season"].to_list()) == {2021, 2022}
    p21 = pairs.filter(pl.col("target_season") == 2021).row(0, named=True)
    assert p21["prev_ppg"] == pytest.approx(12.5)  # 200/16
    assert p21["target_ppg"] == pytest.approx(15.0)  # 240/16
    assert set(PREDICTORS) <= set(pairs.columns)


def test_training_pairs_cutoff_excludes_target_season() -> None:
    panel = _panel(
        [
            _panel_row(season=2020),
            _panel_row(season=2021, age=25.0),
        ]
    )
    # before_season 2021 -> only target seasons < 2021, but 2020 has no prior -> empty.
    assert training_pairs(panel, before_season=2021).height == 0


def test_projection_features_advances_age_and_takes_latest() -> None:
    panel = _panel(
        [
            _panel_row(season=2021, fantasy_points_ppr=240.0, games=16, age=25.0),
            _panel_row(season=2022, fantasy_points_ppr=180.0, games=15, age=26.0),
        ]
    )
    feats = projection_features(panel, target_season=2023)
    row = feats.row(0, named=True)
    assert row["prev_ppg"] == pytest.approx(12.0)  # 180/15, the latest prior season
    assert row["age"] == pytest.approx(27.0)  # 26 at 2022, advanced to 2023
    assert row["age2"] == pytest.approx(27.0**2)


def test_bayesian_fit_predict_smoke() -> None:
    pytest.importorskip("pymc")
    import numpy as np
    from src.projections.bayesian.model import fit_model

    rng = np.random.default_rng(0)
    n = 160
    prev_ppg = rng.uniform(2.0, 20.0, n)
    pairs = pl.DataFrame(
        {
            "player_id": [f"p{i % 40}" for i in range(n)],
            "position_group": rng.choice(["QB", "RB", "WR", "TE"], n),
            "target_ppg": prev_ppg * 0.8 + 3.0 + rng.normal(0, 2.0, n),
            "prev_ppg": prev_ppg,
            "prev_exp_ppg": prev_ppg + rng.normal(0, 1.0, n),
            "prev_target_share": rng.uniform(0, 0.3, n),
            "prev_snap_share": rng.uniform(0, 1.0, n),
            "age": rng.uniform(22.0, 32.0, n),
        }
    ).with_columns((pl.col("age") ** 2).alias("age2"))

    fit = fit_model(pairs, draws=40, tune=40, chains=1, seed=0)
    feats = pl.DataFrame(
        {
            "player_id": ["p0", "brand_new"],  # one known, one unseen player
            "position_group": ["WR", "RB"],
            "prev_ppg": [15.0, 8.0],
            "prev_exp_ppg": [14.0, 8.0],
            "prev_target_share": [0.25, 0.1],
            "prev_snap_share": [0.85, 0.5],
            "age": [26.0, 24.0],
            "projected_games": [16.0, 16.0],
        }
    ).with_columns((pl.col("age") ** 2).alias("age2"))

    pred = fit.predict(feats, seed=0)
    assert pred.height == 2
    for r in pred.iter_rows(named=True):
        assert r["points_low"] <= r["points_median"] <= r["points_high"]
        assert r["projected_points"] >= 0.0
        assert r["points_high"] > r["points_low"]  # non-degenerate interval
