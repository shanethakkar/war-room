"""Draft-simulation tests - network-free, synthetic pools.

The optimal-lineup scorer is checked by hand; the simulation is validated by the
property that a *perfect* board (VOR = actual finish) must out-draft a random ADP.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from src.formats import get_format
from src.validation.draft_sim import (
    _lineup_slots,
    _lineup_value,
    evaluate_pool,
)


def test_lineup_value_optimal_fill() -> None:
    roster = get_format("redraft_ppr").roster  # qb1 rb2 wr2 te1 flex1
    slots = _lineup_slots(roster)
    # codes: QB=0, RB=1, WR=2, TE=3
    codes = [0, 1, 1, 1, 2, 2, 3]
    points = [20.0, 15.0, 10.0, 5.0, 12.0, 8.0, 6.0]
    # QB 20 + RB 15,10 + WR 12,8 + TE 6 + FLEX best-remaining (RB 5) = 76
    assert _lineup_value(codes, points, slots) == 76.0


def _perfect_vs_random_pool(seed: int = 0) -> pl.DataFrame:
    """Pool where VOR = actual (perfect board) and ADP is random noise."""
    rng = np.random.default_rng(seed)
    positions = ["QB"] * 30 + ["RB"] * 60 + ["WR"] * 70 + ["TE"] * 40
    n = len(positions)
    actual = rng.uniform(20.0, 320.0, n)
    return pl.DataFrame(
        {
            "position_group": positions,
            "vor": actual,  # perfect foresight
            "actual_points": actual,
            "adp": rng.permutation(n).astype(float),  # uncorrelated with actual
        }
    )


def test_perfect_board_out_drafts_random_adp() -> None:
    pool = _perfect_vs_random_pool()
    roster = get_format("redraft_ppr").roster
    metrics = evaluate_pool(pool, roster, n_sims=40, seed=1)
    # A board that knows the future must draft better teams than random ADP.
    assert metrics["margin"] > 0.0
    assert metrics["win_rate"] > 0.5
    assert metrics["our_value"] > metrics["adp_value"]


def test_two_random_boards_have_no_edge() -> None:
    # Both rankings random wrt actual (reshuffled independently each iteration) ->
    # no systematic edge to either strategy; win rate averages to ~0.5.
    roster = get_format("redraft_ppr").roster
    positions = ["QB"] * 30 + ["RB"] * 60 + ["WR"] * 70 + ["TE"] * 40
    n = len(positions)
    win_rates = []
    for s in range(15):
        rng = np.random.default_rng(s)
        pool = pl.DataFrame(
            {
                "position_group": positions,
                "actual_points": rng.uniform(20.0, 320.0, n),
                "vor": rng.permutation(n).astype(float),
                "adp": rng.permutation(n).astype(float),
            }
        )
        win_rates.append(evaluate_pool(pool, roster, n_sims=30, seed=1)["win_rate"])
    assert 0.40 <= float(np.mean(win_rates)) <= 0.60
