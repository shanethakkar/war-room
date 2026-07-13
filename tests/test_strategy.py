"""Strategy-math tests - hand-computable cases, network-free."""

from __future__ import annotations

import numpy as np
import pytest
from src.decision.strategy import (
    conditional_survival,
    expected_best_available,
    expected_pick_value,
    market_anchored_values,
    optimal_plan,
    positional_outlooks,
)


def test_conditional_survival_truncation() -> None:
    # A faller: ADP 15, still available at pick 30. Unconditionally he'd be
    # ~gone (<1%); conditionally he's very much alive. (Deep in the normal
    # tail the hazard is steep, so ~0.36 to survive ONE more pick is correct -
    # fallers get snapped up fast - but it's orders of magnitude above the
    # unconditional number.)
    adp = np.array([15.0])
    sd = np.array([4.0])
    uncond = conditional_survival(adp, sd, current_pick=1, future_pick=31)
    cond = conditional_survival(adp, sd, current_pick=30, future_pick=31)
    assert uncond[0] < 0.01
    assert cond[0] > 0.25
    assert cond[0] > 20 * uncond[0]


def test_conditional_survival_monotone_in_future_pick() -> None:
    adp = np.array([24.0, 24.0])
    sd = np.array([6.0, 6.0])
    s_near = conditional_survival(adp, sd, 13, 19)
    s_far = conditional_survival(adp, sd, 13, 36)
    assert np.all(s_far < s_near)


def test_expected_best_available_hand_calc() -> None:
    # Two players: 100 pts surviving with p=0.5; 80 pts surviving surely.
    # E = 0.5*100 + 0.5*(1.0*80) = 90; tail term is 0 (second survives surely).
    points = np.array([100.0, 80.0])
    survival = np.array([0.5, 1.0])
    assert expected_best_available(points, survival) == pytest.approx(90.0)


def test_expected_best_available_tail_floor() -> None:
    # Both likely gone -> expectation approaches the worst player's level
    # (replacement stream), not zero.
    points = np.array([100.0, 80.0])
    survival = np.array([0.0, 0.0])
    assert expected_best_available(points, survival) == pytest.approx(80.0)


def test_outlooks_rank_cliff_position_first() -> None:
    # QB: elite now (324), nothing behind -> big cliff. RB: flat tier -> tiny cliff.
    positions = ["QB", "QB", "RB", "RB", "RB"]
    names = ["EliteQB", "BadQB", "RB1", "RB2", "RB3"]
    points = np.array([324.0, 250.0, 270.0, 268.0, 266.0])
    adp = np.array([22.0, 60.0, 21.0, 24.0, 26.0])
    stdev = np.array([4.0, 8.0, 4.0, 4.0, 4.0])
    available = np.ones(5, dtype=bool)
    out = positional_outlooks(
        positions,
        names,
        points,
        adp,
        stdev,
        available,
        current_pick=18,
        my_next_pick=30,
        my_pick_after=43,
    )
    assert out[0].position == "QB"  # the user's exact scenario: QB cliff wins
    qb, rb = out[0], next(o for o in out if o.position == "RB")
    assert qb.marginal_value > rb.marginal_value
    assert qb.best_now_name == "EliteQB"
    # Cliffs decay with pick distance.
    assert qb.expected_at_next >= qb.expected_after_next


def test_expected_pick_value_follows_given_order() -> None:
    # Blend prefers player B (index 1) even though A has more points; the
    # expectation must walk B first.
    points = np.array([100.0, 90.0])
    survival = np.array([1.0, 1.0])
    blend_order = np.array([1, 0])  # take B first
    assert expected_pick_value(points, survival, blend_order) == pytest.approx(90.0)


def test_optimal_plan_schedules_slow_decay_last() -> None:
    # Two picks, need 1 WR + 1 K. WR decays steeply (200 -> 150); K barely
    # (120 -> 118). The plan must take WR now, K later - the greedy-VONA
    # failure mode (kicker in round 7) reversed.
    positions = ["WR", "K"]
    v = np.zeros((2, 2, 3))
    v[0, 0, 0], v[0, 1, 0] = 200.0, 150.0  # WR now / later
    v[1, 0, 0], v[1, 1, 0] = 120.0, 118.0  # K now / later
    plan = optimal_plan(
        v,
        positions,
        {"WR": 1, "K": 1},
        flex_need=0,
        superflex_need=0,
        my_picks=[10, 34],
    )
    assert plan.positions == ["WR", "K"]
    assert plan.expected_total == pytest.approx(200.0 + 118.0)


def test_market_anchored_values_suppress_model_outliers() -> None:
    # A model-loved player at market rank 20 gets pulled toward what the
    # market pays at rank 20, keeping only a minority of the model's opinion.
    rng = np.random.default_rng(0)
    n = 60
    points = np.linspace(300.0, 80.0, n) + rng.normal(0, 3, n)
    adp_rank = np.arange(1.0, n + 1)
    points[19] = 320.0  # model outlier at market rank 20
    values = market_anchored_values(points, adp_rank)
    assert values[19] < 0.5 * (points[19] + points[18])  # pulled well below model
    assert values[19] > points[21]  # but the tilt still lifts him above rank peers


def test_outlooks_respect_availability_and_missing_adp() -> None:
    positions = ["WR", "WR"]
    names = ["TakenGuy", "NoAdpGuy"]
    points = np.array([200.0, 150.0])
    adp = np.array([10.0, np.nan])
    stdev = np.array([3.0, np.nan])
    available = np.array([False, True])  # the 200-pt WR is already drafted
    out = positional_outlooks(
        positions,
        names,
        points,
        adp,
        stdev,
        available,
        current_pick=25,
        my_next_pick=36,
        my_pick_after=49,
    )
    wr = out[0]
    assert wr.best_now == 150.0  # taken player excluded
    # No-ADP players are assumed to survive -> waiting costs ~nothing.
    assert wr.marginal_value == pytest.approx(0.0, abs=1e-9)
