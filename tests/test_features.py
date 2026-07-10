"""Feature-panel tests - network-free, using synthetic frames.

Each frame includes only the columns the transforms touch, with dtypes matching
the real cached tables (notably ``season`` as Int32 in stats/snaps but String in
ff_opportunity). The real build is exercised by ``python -m src.features.build``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from src.features.panel import (
    add_efficiency,
    assemble_panel,
    offensive_player_seasons,
    season_opportunity,
    season_snap_share,
)


def _pss_row(**over: Any) -> dict[str, Any]:
    """One player_stats_season row with sensible zero defaults."""
    row: dict[str, Any] = {
        "player_id": "00-1",
        "player_display_name": "Player One",
        "position": "WR",
        "position_group": "WR",
        "recent_team": "SF",
        "season": 2020,
        "games": 16,
        "targets": 0,
        "receptions": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "receiving_air_yards": 0,
        "receiving_yards_after_catch": 0,
        "target_share": 0.0,
        "air_yards_share": 0.0,
        "wopr": 0.0,
        "carries": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "attempts": 0,
        "completions": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "passing_interceptions": 0,
        "passing_air_yards": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "fantasy_points": 0.0,
        "fantasy_points_ppr": 0.0,
    }
    row.update(over)
    return row


def _pss_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))


def test_offensive_filter_keeps_only_skill_positions() -> None:
    df = _pss_frame(
        [
            _pss_row(player_id="a", position_group="WR"),
            _pss_row(player_id="b", position_group="QB"),
            _pss_row(player_id="c", position_group="DB"),
            _pss_row(player_id="d", position_group="OL"),
        ]
    )
    out = offensive_player_seasons(df)
    assert set(out["player_id"]) == {"a", "b"}


def test_efficiency_rates_and_zero_opportunity_is_null() -> None:
    df = _pss_frame(
        [
            _pss_row(player_id="wr", targets=5, receptions=3, receiving_yards=45),
            _pss_row(player_id="zero", targets=0, receptions=0),
        ]
    )
    out = add_efficiency(offensive_player_seasons(df)).sort("player_id")
    wr = out.filter(pl.col("player_id") == "wr").row(0, named=True)
    zero = out.filter(pl.col("player_id") == "zero").row(0, named=True)
    assert wr["catch_rate"] == 0.6
    assert wr["yards_per_target"] == 9.0
    # Zero targets -> null, not a divide-by-zero 0.0 or inf.
    assert zero["catch_rate"] is None
    assert zero["yards_per_target"] is None


def test_season_opportunity_excludes_playoffs_and_casts_season() -> None:
    ffo = pl.DataFrame(
        {
            "player_id": ["x", "x", "x"],
            "season": ["2020", "2020", "2020"],  # string in this table
            "week": [1.0, 2.0, 20.0],  # week 20 is a playoff week -> dropped
            "pass_touchdown_exp": [0.0, 0.0, 0.0],
            "rush_touchdown_exp": [0.0, 0.0, 0.0],
            "rec_touchdown_exp": [0.5, 0.5, 5.0],
            "receptions_exp": [4.0, 4.0, 9.0],
            "rush_yards_gained_exp": [0.0, 0.0, 0.0],
            "rec_yards_gained_exp": [40.0, 40.0, 100.0],
            "total_fantasy_points_exp": [10.0, 10.0, 30.0],
        }
    )
    out = season_opportunity(ffo)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["season"] == 2020
    assert out.schema["season"] == pl.Int32
    # Only the two regular-season weeks summed (playoff week 20 excluded).
    assert row["expected_rec_tds"] == 1.0
    assert row["expected_receptions"] == 8.0
    assert row["expected_fantasy_points"] == 20.0


def test_snap_share_joins_via_pfr_crosswalk() -> None:
    snaps = pl.DataFrame(
        {
            "pfr_player_id": ["PfrA", "PfrA", "PfrZ"],
            "season": pl.Series([2020, 2020, 2020], dtype=pl.Int32),
            "offense_pct": [0.8, 0.6, 0.5],
            "offense_snaps": [50.0, 40.0, 30.0],
        }
    )
    crosswalk = pl.DataFrame({"player_id": ["a"], "pfr_id": ["PfrA"]})
    out = season_snap_share(snaps, crosswalk)
    # PfrZ has no crosswalk entry -> inner join drops it.
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["player_id"] == "a"
    assert abs(row["snap_share"] - 0.7) < 1e-9
    assert row["offense_snaps"] == 90.0


def _players_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["a", "b"],
            "pfr_id": ["PfrA", "PfrB"],
            "birth_date": ["1995-09-01", "1990-01-15"],
            "draft_year": pl.Series([2017, 2012], dtype=pl.Int32),
            "draft_round": pl.Series([1, 3], dtype=pl.Int32),
            "draft_pick": pl.Series([5, 80], dtype=pl.Int32),
            "rookie_season": pl.Series([2017, 2012], dtype=pl.Int32),
        }
    )


def test_assemble_panel_shape_age_and_left_joins() -> None:
    pss = _pss_frame(
        [
            _pss_row(player_id="a", season=2020, targets=100, fantasy_points_ppr=250.0),
            _pss_row(player_id="b", season=2020, targets=50, fantasy_points_ppr=120.0),
        ]
    )
    ffo = pl.DataFrame(
        {
            "player_id": ["a"],
            "season": ["2020"],
            "week": [1.0],
            "pass_touchdown_exp": [0.0],
            "rush_touchdown_exp": [0.0],
            "rec_touchdown_exp": [1.0],
            "receptions_exp": [6.0],
            "rush_yards_gained_exp": [0.0],
            "rec_yards_gained_exp": [70.0],
            "total_fantasy_points_exp": [15.0],
        }
    )
    snaps = pl.DataFrame(
        {
            "pfr_player_id": ["PfrA"],
            "season": pl.Series([2020], dtype=pl.Int32),
            "offense_pct": [0.9],
            "offense_snaps": [60.0],
        }
    )
    panel = assemble_panel(pss, ffo, snaps, _players_frame())

    assert panel.height == 2
    assert panel.select(["player_id", "season"]).n_unique() == 2

    a = panel.filter(pl.col("player_id") == "a").row(0, named=True)
    b = panel.filter(pl.col("player_id") == "b").row(0, named=True)

    # Age as of Sept 1, 2020 for a 1995-09-01 birthdate.
    assert a["age"] == 25.0
    assert a["is_rookie"] is False
    assert a["experience"] == 3
    assert a["snap_share"] == 0.9
    assert a["expected_rec_tds"] == 1.0

    # b has no ff_opp or snap row -> left join leaves nulls, row still present.
    assert b["snap_share"] is None
    assert b["expected_fantasy_points"] is None

    # Sorted by season asc, then ppr points desc.
    assert panel["player_id"].to_list() == ["a", "b"]
