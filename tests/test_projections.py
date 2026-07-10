"""Baseline projection + scoring tests - network-free, synthetic frames.

Covers the scoring interpreter, pooled priors, the shrinkage formula, the
leakage-free split, and rookie inclusion. The real projection is exercised by
``python -m src.projections.run``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from src.formats import get_format
from src.formats.score import score_components
from src.projections.baseline.priors import positional_priors
from src.projections.baseline.project import _shrunk, project_season


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_components_matches_hand_calc() -> None:
    df = pl.DataFrame(
        {
            "pass_yards": [300.0],
            "pass_tds": [2.0],
            "interceptions": [1.0],
            "rush_yards": [10.0],
            "rush_tds": [0.0],
            "receptions": [5.0],
            "rec_yards": [60.0],
            "rec_tds": [1.0],
            "fumbles_lost": [0.0],
        }
    )
    scored = score_components(df, get_format("redraft_ppr"))
    # 0.04*300 + 4*2 - 2*1 + 0.1*10 + 6*0 + 1*5 + 0.1*60 + 6*1 = 36.0
    assert scored["projected_points"][0] == pytest.approx(36.0)


def test_score_half_ppr_reception_effect() -> None:
    df = pl.DataFrame(
        {
            "pass_yards": [0.0],
            "pass_tds": [0.0],
            "interceptions": [0.0],
            "rush_yards": [0.0],
            "rush_tds": [0.0],
            "receptions": [10.0],
            "rec_yards": [0.0],
            "rec_tds": [0.0],
            "fumbles_lost": [0.0],
        }
    )
    ppr = score_components(df, get_format("redraft_ppr"))["projected_points"][0]
    assert ppr == pytest.approx(10.0)  # full PPR: 1.0 per reception


# --------------------------------------------------------------------------- #
# shrinkage formula
# --------------------------------------------------------------------------- #
def test_shrunk_blends_between_own_and_prior() -> None:
    df = pl.DataFrame({"num": [30.0], "prior": [0.6], "den": [45.0]}).with_columns(
        _shrunk("num", "prior", "den", 45.0).alias("out")
    )
    # own rate = 30/45 = 0.667; prior = 0.6; k = own opportunity -> midpoint-ish
    assert df["out"][0] == pytest.approx((30 + 45 * 0.6) / (45 + 45))
    assert 0.6 < df["out"][0] < 30 / 45


def test_shrunk_high_sample_stays_near_own() -> None:
    small = pl.DataFrame({"num": [4.0], "prior": [0.6], "den": [5.0]}).with_columns(
        _shrunk("num", "prior", "den", 45.0).alias("out")
    )["out"][0]
    big = pl.DataFrame({"num": [400.0], "prior": [0.6], "den": [500.0]}).with_columns(
        _shrunk("num", "prior", "den", 45.0).alias("out")
    )["out"][0]
    # Both have own rate 0.8; the larger sample regresses less toward 0.6.
    assert small < big
    assert abs(big - 0.8) < abs(small - 0.8)


# --------------------------------------------------------------------------- #
# synthetic panel / players builders
# --------------------------------------------------------------------------- #
def _row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "season": 2022,
        "player_id": "a",
        "player_name": "Vet Player",
        "position": "WR",
        "position_group": "WR",
        "team": "SF",
        "games": 16,
        "age": 25.0,
        "is_rookie": False,
        "draft_round": 1,
        "targets": 0.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "expected_rec_tds": 0.0,
        "carries": 0.0,
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
        "expected_rush_tds": 0.0,
        "pass_attempts": 0.0,
        "pass_completions": 0.0,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "expected_pass_tds": 0.0,
        "interceptions": 0.0,
        "rushing_fumbles_lost": 0.0,
        "receiving_fumbles_lost": 0.0,
    }
    row.update(over)
    return row


def _panel(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("draft_round").cast(pl.Int32)
    )


def _players(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("draft_round").cast(pl.Int32),
        pl.col("rookie_season").cast(pl.Int32),
    )


def test_positional_priors_pooled_ratio() -> None:
    train = _panel(
        [
            _row(player_id="w1", targets=100.0, receptions=70.0, receiving_yards=900.0),
            _row(player_id="w2", targets=100.0, receptions=50.0, receiving_yards=700.0),
        ]
    )
    priors = positional_priors(train).filter(pl.col("position_group") == "WR")
    # pooled catch_rate = (70+50)/(100+100) = 0.6; ypt = (900+700)/200 = 8.0
    assert priors["catch_rate"][0] == pytest.approx(0.6)
    assert priors["yards_per_target"][0] == pytest.approx(8.0)


def test_project_season_is_leakage_free() -> None:
    base = [
        _row(
            season=2022,
            player_id="a",
            targets=120.0,
            receptions=80.0,
            receiving_yards=1000.0,
            games=16,
        ),
        _row(
            season=2023,
            player_id="a",
            targets=130.0,
            receptions=88.0,
            receiving_yards=1100.0,
            games=16,
        ),
    ]
    players = _players(
        [
            {
                "gsis_id": "a",
                "display_name": "Vet",
                "position": "WR",
                "position_group": "WR",
                "draft_round": 1,
                "draft_pick": 5,
                "rookie_season": 2019,
            }
        ]
    )

    proj1 = project_season(_panel(base), players, 2024)
    # Add an absurd target-season row; it must not change the 2024 projection.
    contaminated = _panel(
        base + [_row(season=2024, player_id="a", targets=9999.0, receptions=9999.0)]
    )
    proj2 = project_season(contaminated, players, 2024)

    a1 = proj1.filter(pl.col("player_id") == "a").row(0, named=True)
    a2 = proj2.filter(pl.col("player_id") == "a").row(0, named=True)
    assert a1["targets"] == pytest.approx(a2["targets"])
    # And no duplicate row for the player.
    assert proj2.filter(pl.col("player_id") == "a").height == 1


def test_project_season_includes_rookie_from_draft_prior() -> None:
    # Training rookies (WRs) so a position-level rookie prior exists.
    train = _panel(
        [
            _row(
                season=2022,
                player_id="r_old",
                is_rookie=True,
                draft_round=1,
                games=16,
                targets=110.0,
                receptions=75.0,
                receiving_yards=950.0,
                expected_rec_tds=6.0,
            ),
            _row(
                season=2023,
                player_id="vet",
                is_rookie=False,
                games=16,
                targets=100.0,
                receptions=65.0,
                receiving_yards=850.0,
            ),
        ]
    )
    players = _players(
        [
            {
                "gsis_id": "vet",
                "display_name": "Vet",
                "position": "WR",
                "position_group": "WR",
                "draft_round": 2,
                "draft_pick": 40,
                "rookie_season": 2021,
            },
            {
                "gsis_id": "rook",
                "display_name": "Rook",
                "position": "WR",
                "position_group": "WR",
                "draft_round": 1,
                "draft_pick": 6,
                "rookie_season": 2024,
            },
        ]
    )
    proj = project_season(train, players, 2024)
    rook = proj.filter(pl.col("player_id") == "rook")
    assert rook.height == 1
    assert rook["is_rookie"][0] is True
    assert rook["targets"][0] > 0.0  # got a draft-capital opportunity prior
