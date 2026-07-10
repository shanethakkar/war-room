"""Decision-layer tests - network-free.

Covers replacement-level VOR, the superflex QB edge (the headline differentiator),
tier methods, and the assembled board shape.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from src.decision.board import _BOARD_COLS, build_value_board
from src.decision.replacement import add_starter_flags, add_vor
from src.decision.tiers import add_overlap_tiers, add_position_tiers
from src.formats import get_format
from src.formats.base import RosterConfig


def _scored(players: list[tuple[str, float]]) -> pl.DataFrame:
    """Build a scored projection (with placeholder intervals) from (pos, pts) pairs."""
    rows: list[dict[str, Any]] = []
    for i, (pos, pts) in enumerate(players):
        rows.append(
            {
                "player_id": f"p{i}",
                "player_name": f"{pos}{i}",
                "position": pos,
                "position_group": pos,
                "team": "SF",
                "is_rookie": False,
                "projected_games": 16.0,
                "projected_points": pts,
                "points_low": pts * 0.6,
                "points_median": pts,
                "points_high": pts * 1.4,
            }
        )
    return pl.DataFrame(rows)


def _vor_of(df: pl.DataFrame, name: str) -> float:
    return float(df.filter(pl.col("player_name") == name)["vor"][0])


def test_replacement_vor_single_qb_format() -> None:
    roster = RosterConfig(
        teams=1, qb=1, rb=1, wr=1, te=1, flex=0, superflex=0, dst=0, k=0, bench=0
    )
    scored = _scored(
        [
            ("QB", 300.0),
            ("QB", 250.0),
            ("RB", 200.0),
            ("RB", 150.0),
            ("WR", 180.0),
            ("WR", 120.0),
            ("TE", 100.0),
            ("TE", 60.0),
        ]
    )
    out = add_vor(add_starter_flags(scored, roster))
    # 1 team, 1 QB starter -> replacement QB = best non-starter QB = 250.
    assert _vor_of(out, "QB0") == 300.0 - 250.0
    assert _vor_of(out, "QB1") == 0.0  # the replacement player itself


def test_superflex_lifts_elite_qb_vor() -> None:
    # Identical rosters except the superflex slot.
    common = dict(teams=1, rb=2, wr=2, te=1, flex=1, dst=0, k=0, bench=0)
    redraft = RosterConfig(qb=1, superflex=0, **common)
    superflex = RosterConfig(qb=1, superflex=1, **common)
    scored = _scored(
        [
            ("QB", 320.0),
            ("QB", 300.0),
            ("QB", 280.0),
            ("RB", 200.0),
            ("RB", 190.0),
            ("RB", 150.0),
            ("RB", 140.0),
            ("WR", 210.0),
            ("WR", 180.0),
            ("WR", 160.0),
            ("WR", 120.0),
            ("TE", 130.0),
            ("TE", 90.0),
        ]
    )
    vor_redraft = _vor_of(add_vor(add_starter_flags(scored, redraft)), "QB0")
    vor_superflex = _vor_of(add_vor(add_starter_flags(scored, superflex)), "QB0")
    # Superflex makes a 2nd QB startable -> lower QB replacement -> higher elite VOR.
    assert vor_superflex > vor_redraft


def test_position_tiers_break_on_gap() -> None:
    df = pl.DataFrame(
        {
            "position_group": ["RB", "RB", "RB", "RB"],
            "vor": [100.0, 95.0, 60.0, 58.0],
        }
    )
    out = add_position_tiers(df, gap=10.0).sort("vor", descending=True)
    # 100,95 together; big gap to 60 starts tier 2; 58 stays tier 2.
    assert out["position_tier"].to_list() == [1, 1, 2, 2]


def test_overlap_tiers_split_on_sigma_scaled_gap() -> None:
    # Interval width 128 -> sigma ~= 128/2.563 ~= 50; TIER_SEP=1 -> ~50-pt threshold.
    medians = [300.0, 290.0, 150.0, 140.0]
    df = pl.DataFrame(
        {
            "position_group": ["WR"] * 4,
            "projected_points": medians,
            "points_median": medians,
            "points_low": [m - 64.0 for m in medians],
            "points_high": [m + 64.0 for m in medians],
        }
    )
    out = add_overlap_tiers(df).sort("points_median", descending=True)
    # 300 & 290 within ~1 sigma -> tier 1; 150 is >1 sigma below -> tier 2; 140 joins.
    assert out["position_tier"].to_list() == [1, 1, 2, 2]


def test_build_value_board_shape_and_ranking() -> None:
    scored = _scored(
        [
            ("QB", 300.0),
            ("QB", 250.0),
            ("RB", 280.0),
            ("RB", 120.0),
            ("WR", 260.0),
            ("WR", 110.0),
            ("TE", 150.0),
            ("TE", 70.0),
        ]
    )
    board = build_value_board(scored, get_format("redraft_ppr"))
    assert board.columns == list(_BOARD_COLS)
    assert board["overall_rank"].to_list() == list(range(1, board.height + 1))
    # Sorted by VOR descending.
    assert board["vor"].is_sorted(descending=True)
