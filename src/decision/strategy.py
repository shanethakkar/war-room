"""Draft strategy math: positional cliffs and marginal pick value (Phase A).

The value of a pick is not the player's points - it's the points minus what you
could still get at that position at your NEXT pick (the industry's "VONA",
rarely computed properly because it needs opponents' behavior). We have that
behavior: each player's draft position is modeled as Normal(adp, per-player
stdev from FFC's observed pick variance).

Core quantities (all conditional on the CURRENT board state):

- ``conditional_survival``: P(player still available at future pick n | he is
  available now at pick c) - the truncated-normal correction matters mid-draft:
  a faller sitting on the board at pick 30 with ADP 15 is not "0% available".
- ``expected_best_available``: E[max projected points among a position's
  survivors at pick n] = sum_k p_k * s_k * prod_{j<k} (1 - s_j) over players
  ordered by points (independent-survival approximation; opponents' picks are
  not truly independent, but errors are second-order and re-solving after every
  real pick self-corrects).
- ``positional_cliffs``: the E[best available] curve per position at your next
  few picks - the "is the cliff coming?" view.
- ``marginal_values``: points(best now) - E[best at my next pick] per position -
  what waiting actually costs.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

MIN_STDEV = 2.0
DEFAULT_STDEV = 8.0


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return np.asarray(0.5 * (1.0 + erf(z / math.sqrt(2.0))), dtype=float)


def conditional_survival(
    adp: np.ndarray, stdev: np.ndarray, current_pick: int, future_pick: int
) -> np.ndarray:
    """P(available at ``future_pick`` | available at ``current_pick``), per player.

    Draft position X ~ Normal(adp, sd); availability at pick n means X >= n.
    P(X >= n | X >= c) = Phi((adp - n)/sd) / Phi((adp - c)/sd), clipped for
    numerical safety. Players with no ADP get DEFAULT_STDEV around a late slot
    upstream; here nulls are the caller's job.
    """
    sd = np.clip(np.nan_to_num(stdev, nan=DEFAULT_STDEV), MIN_STDEV, None)
    p_future = _norm_cdf((adp - future_pick) / sd)
    p_now = np.clip(_norm_cdf((adp - current_pick) / sd), 1e-9, None)
    return np.asarray(np.clip(p_future / p_now, 0.0, 1.0), dtype=float)


def expected_best_available(points: np.ndarray, survival: np.ndarray) -> float:
    """E[max points among survivors], ordered-inclusion over points-desc order.

    E = sum_k points_k * s_k * prod_{j<k}(1 - s_j); if everyone is gone the
    contribution tends to the worst player's tail, which we close by treating
    the last player as a floor (someone at the position is always signable).
    """
    if len(points) == 0:
        return 0.0
    order = np.argsort(-points)
    p = points[order]
    s = np.clip(survival[order], 0.0, 1.0)
    none_better = np.concatenate(([1.0], np.cumprod(1.0 - s)[:-1]))
    expected = float(np.sum(p * s * none_better))
    # Tail closure: with prob prod(1-s) all listed players are gone; assume the
    # worst listed player's level is still attainable (replacement stream).
    expected += float(np.prod(1.0 - s)) * float(p[-1])
    return expected


def expected_pick_value(
    points: np.ndarray, survival: np.ndarray, order: np.ndarray
) -> float:
    """E[points of the first-available player in ``order``] (selection-consistent).

    Unlike ``expected_best_available`` (max by points), this walks the given
    order - e.g. the board's blend order, the player you'd ACTUALLY take - so
    plan values and picks agree. Tail closes at the last player's level.
    """
    if len(points) == 0:
        return 0.0
    p = points[order]
    s = np.clip(survival[order], 0.0, 1.0)
    none_better = np.concatenate(([1.0], np.cumprod(1.0 - s)[:-1]))
    expected = float(np.sum(p * s * none_better))
    expected += float(np.prod(1.0 - s)) * float(p[-1])
    return expected


def _expected_at(
    pts: np.ndarray,
    adp: np.ndarray,
    stdev: np.ndarray,
    current_pick: int,
    target: int | None,
) -> float:
    """E[best available points] at ``target`` (0 when there is no future pick).

    Players without an ADP are assumed to survive - nobody in an ADP-driven
    room is racing you to them.
    """
    if target is None:
        return 0.0
    survival = np.ones(len(pts))
    has_adp = ~np.isnan(adp)
    if has_adp.any():
        survival[has_adp] = conditional_survival(
            adp[has_adp], stdev[has_adp], current_pick, target
        )
    return expected_best_available(pts, survival)


@dataclass(frozen=True)
class PositionOutlook:
    """A position's cliff curve and marginal value at the current decision."""

    position: str
    best_now: float
    best_now_name: str
    expected_at_next: float
    expected_after_next: float
    marginal_value: float  # best_now - expected_at_next: cost of waiting


def positional_outlooks(
    positions: list[str],
    names: list[str],
    points: np.ndarray,
    adp: np.ndarray,
    stdev: np.ndarray,
    available: np.ndarray,
    current_pick: int,
    my_next_pick: int | None,
    my_pick_after: int | None,
) -> list[PositionOutlook]:
    """Cliff curve + marginal value per position, given the current board state.

    ``available`` masks players still on the board; ADP-less players are treated
    as survivors (nobody in an ADP-driven room is racing you to them).
    """
    pos_arr = np.asarray(positions)
    name_arr = np.asarray(names)
    outlooks: list[PositionOutlook] = []
    for group in dict.fromkeys(positions):  # preserve first-seen order
        mask = (pos_arr == group) & available
        if not mask.any():
            continue
        pts = points[mask]
        group_adp = adp[mask]
        group_sd = stdev[mask]

        best_idx = int(np.argmax(pts))
        e_next = _expected_at(pts, group_adp, group_sd, current_pick, my_next_pick)
        e_after = _expected_at(pts, group_adp, group_sd, current_pick, my_pick_after)
        outlooks.append(
            PositionOutlook(
                position=str(group),
                best_now=float(pts[best_idx]),
                best_now_name=str(name_arr[mask][best_idx]),
                expected_at_next=e_next,
                expected_after_next=e_after,
                marginal_value=float(pts[best_idx]) - e_next,
            )
        )
    return sorted(outlooks, key=lambda o: o.marginal_value, reverse=True)


# --------------------------------------------------------------------------- #
# Full-plan DP: assign positions to ALL my remaining picks jointly.
#
# One-step marginal value ("VONA") is provably myopic: it drafted a kicker in
# round 7 in validation, because K's value barely decays pick-to-pick while the
# real cost - the round-7 WR you'll never get back - is invisible one step out.
# The DP sees the whole assignment: slow-decay positions (K/DST) naturally land
# in the last picks, steep-cliff positions get taken while they exist.
# --------------------------------------------------------------------------- #

FLEX_SET = frozenset({"RB", "WR", "TE"})
SUPERFLEX_SET = frozenset({"QB", "RB", "WR", "TE"})
MAX_SAME_POS = 3  # plan at most this many starters per position
# Value function anchoring: cliffs priced ~80% by the market, 20% by the model
# (the validated proportions). A pure-model value function re-leverages exactly
# the positional biases the blend suppresses (it drafted Kelce over Tyreek).
MARKET_VALUE_WEIGHT = 0.8


def market_anchored_values(
    points: np.ndarray, adp_rank: np.ndarray, window: int = 9
) -> np.ndarray:
    """Per-player value = 0.8 * market-implied points + 0.2 * model points.

    Market-implied points: a monotone-decreasing smoothing of projected points
    over ADP rank - the market's own price curve, with model level-scale.
    Players without an ADP rank (NaN) keep a discounted model value.
    """
    values = np.asarray(points, dtype=float).copy()
    has_rank = ~np.isnan(adp_rank)
    if has_rank.sum() >= window:
        idx = np.where(has_rank)[0]
        order = idx[np.argsort(adp_rank[idx])]
        pts = points[order]
        kernel = np.ones(window) / window
        smoothed = np.convolve(pts, kernel, mode="same")
        # Fix convolution edge shrinkage, then enforce monotone decreasing.
        smoothed[: window // 2] = smoothed[window // 2]
        smoothed[-(window // 2) :] = smoothed[-(window // 2) - 1]
        smoothed = np.minimum.accumulate(smoothed)
        market = np.full(len(points), np.nan)
        market[order] = smoothed
        values[has_rank] = (
            MARKET_VALUE_WEIGHT * market[has_rank]
            + (1 - MARKET_VALUE_WEIGHT) * points[has_rank]
        )
    # ADP-less players: model points, discounted (the market has no bid).
    values[~has_rank] = 0.5 * points[~has_rank]
    return values


@dataclass(frozen=True)
class DraftPlan:
    """The optimal position sequence for my remaining picks."""

    picks: list[int]  # my upcoming overall pick numbers
    positions: list[str | None]  # planned position per pick (None = bench value)
    expected_total: float


def plan_value_matrix(
    positions: list[str],
    points: np.ndarray,
    blend_rank: np.ndarray,
    adp: np.ndarray,
    stdev: np.ndarray,
    available: np.ndarray,
    current_pick: int,
    my_picks: list[int],
    position_order: list[str],
) -> np.ndarray:
    """V[p, k, j]: E[value of my (j+1)-th selection at position p at pick k].

    Selection-consistent: expectation over blend-order first-available, with the
    j best-by-blend current players at the position excluded (my own earlier
    planned picks consume them - an approximation that ignores which specific
    player survives, second-order for planning).
    """
    pos_arr = np.asarray(positions)
    v = np.zeros((len(position_order), len(my_picks), MAX_SAME_POS))
    for pi, group in enumerate(position_order):
        mask = (pos_arr == group) & available
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        order = idx[np.argsort(blend_rank[idx])]
        pts = points[order]
        group_adp = adp[order]
        group_sd = stdev[order]
        has_adp = ~np.isnan(group_adp)
        for ki, pick in enumerate(my_picks):
            surv = np.ones(len(order))
            if has_adp.any():
                surv[has_adp] = conditional_survival(
                    group_adp[has_adp], group_sd[has_adp], current_pick, pick
                )
            for j in range(min(MAX_SAME_POS, len(order))):
                # Exclude my j earlier planned selections at this position.
                sub = slice(j, None)
                v[pi, ki, j] = expected_pick_value(
                    pts[sub], surv[sub], np.arange(len(pts[sub]))
                )
    return v


def optimal_plan(
    v: np.ndarray,
    position_order: list[str],
    dedicated_needs: dict[str, int],
    flex_need: int,
    superflex_need: int,
    my_picks: list[int],
) -> DraftPlan:
    """Maximize total expected starter value over position-to-pick assignments.

    State: (pick index, remaining dedicated needs, flex, superflex, my per-
    position take counts). Bench picks are worth 0 in-plan, which schedules
    every starter need at its value-maximizing pick and leaves the rest late.
    """
    n_pos = len(position_order)
    pos_names = position_order

    @functools.cache
    def solve(
        k: int, ded: tuple[int, ...], flex: int, sf: int, taken: tuple[int, ...]
    ) -> tuple[float, int]:
        """Returns (best value from pick k on, best choice: -1 bench or pos idx)."""
        if k >= len(my_picks):
            return 0.0, -1
        best_val, best_choice = solve(k + 1, ded, flex, sf, taken)[0], -1  # bench
        for pi in range(n_pos):
            name = pos_names[pi]
            j = taken[pi]
            if j >= MAX_SAME_POS or v[pi, k, j] <= 0:
                continue
            uses_ded = ded[pi] > 0
            uses_flex = not uses_ded and flex > 0 and name in FLEX_SET
            uses_sf = (
                not uses_ded and not uses_flex and sf > 0 and name in SUPERFLEX_SET
            )
            if not (uses_ded or uses_flex or uses_sf):
                continue
            next_ded = tuple(
                d - 1 if (i == pi and uses_ded) else d for i, d in enumerate(ded)
            )
            next_taken = tuple(t + 1 if i == pi else t for i, t in enumerate(taken))
            val = (
                v[pi, k, j]
                + solve(
                    k + 1,
                    next_ded,
                    flex - 1 if uses_flex else flex,
                    sf - 1 if uses_sf else sf,
                    next_taken,
                )[0]
            )
            if val > best_val:
                best_val, best_choice = val, pi
        return best_val, best_choice

    ded0 = tuple(dedicated_needs.get(name, 0) for name in pos_names)
    taken0 = tuple(0 for _ in pos_names)
    total, _ = solve(0, ded0, flex_need, superflex_need, taken0)

    # Walk the plan forward.
    plan: list[str | None] = []
    ded, flex, sf, taken = ded0, flex_need, superflex_need, taken0
    for k in range(len(my_picks)):
        _, choice = solve(k, ded, flex, sf, taken)
        if choice < 0:
            plan.append(None)
            continue
        name = pos_names[choice]
        plan.append(name)
        if ded[choice] > 0:
            ded = tuple(d - 1 if i == choice else d for i, d in enumerate(ded))
        elif flex > 0 and name in FLEX_SET:
            flex -= 1
        elif sf > 0 and name in SUPERFLEX_SET:
            sf -= 1
        taken = tuple(t + 1 if i == choice else t for i, t in enumerate(taken))
    return DraftPlan(picks=list(my_picks), positions=plan, expected_total=total)
