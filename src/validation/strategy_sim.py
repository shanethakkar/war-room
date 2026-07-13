"""Strategy validation: does position-sequencing beat board-following? NO.

    uv run python -m src.validation.strategy_sim [--start 2019] [--through 2024]

One user-team vs 11 ADP bots, evaluated PAIRED (identical opponents, noise, and
slot per simulation) so the comparison isolates sequencing skill. Policies:

- ``board``: strictly follow the blended board (caps + endgame needs).
- ``greedy``: one-step marginal value (industry "VONA"). **Falsified**: it
  drafts kickers in round 7 (slow-decay positions look free one step out);
  -45 pts/season vs board over 2019-2024.
- ``dp``: full-plan receding-horizon DP over all remaining picks with a
  market-anchored value curve. Beats its own *projected* objective (+40..70
  projected pts) but **loses on actuals** (-10 pts/season mean, wildly
  season-dependent): sequencing trades on tier-difference estimates whose
  signal-to-noise is worse than player levels, and in a market-driven room,
  needs-aware best-available IS near-optimal sequencing (you capture every
  faller; a position plan forfeits fallers elsewhere). Four value-function
  designs (model points, 80/20 market-anchored, pure market curve, historical
  draft-capital curves) all failed to produce a consistent edge.

Verdict (see progress.md): the board stays the policy; cliffs ship as
descriptive insight only. Selection layer is held constant across policies
(2-way blend, DST/K market-pinned) so gaps are attributable to sequencing.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from src.decision.strategy import (
    conditional_survival,
    expected_best_available,
    market_anchored_values,
    optimal_plan,
    plan_value_matrix,
)
from src.formats import get_format
from src.formats.base import RosterConfig
from src.ingest.cache import read_table
from src.validation.draft_sim import (
    _POS,
    _POS_CODE,
    ROUNDS,
    _caps,
    _lineup_slots,
    _lineup_value,
    _needs,
    _rank,
    _snake_order,
    build_pool,
)

BLEND_W = 0.15  # 2-way selection layer, held constant across policies


def _prepare(pool: pl.DataFrame) -> dict[str, np.ndarray]:
    vr = pool["vor"].rank(descending=True).to_numpy().astype(float)
    ar = pool["adp"].rank().to_numpy().astype(float)
    spec = np.isin(np.array(pool["position_group"].to_list()), ["DST", "K"])
    blend = np.where(spec, ar, BLEND_W * vr + (1 - BLEND_W) * ar)
    points = pool["projected_points"].to_numpy().astype(float)
    adp_rank = _rank(pool["adp"].to_numpy().astype(float), descending=False)
    return {
        "pos": np.array([_POS_CODE[p] for p in pool["position_group"].to_list()]),
        "points": points,
        "value": market_anchored_values(points, adp_rank),
        "actual": pool["actual_points"].to_numpy().astype(float),
        "adp": pool["adp"].to_numpy().astype(float),
        "stdev": pool["adp_stdev"].fill_null(8.0).to_numpy().astype(float),
        "blend_rank": _rank(-blend, descending=True),
        "adp_rank": adp_rank,
    }


def _my_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    return [
        r * teams + slot if r % 2 == 0 else (r + 1) * teams - slot + 1
        for r in range(rounds)
    ]


def _board_choice(d: dict[str, np.ndarray], eligible: np.ndarray) -> int:
    return int(np.argmin(np.where(eligible, d["blend_rank"], np.inf)))


def _strategy_choice(
    d: dict[str, np.ndarray],
    eligible: np.ndarray,
    pos_ok: np.ndarray,
    current_pick: int,
    my_next: int | None,
) -> int:
    """Index of the player to take under the marginal-value policy.

    Among positions in ``pos_ok``, choose the one with the highest marginal
    value (board's top candidate's points minus E[best at the position at my
    next pick]); take that candidate. No future pick -> pure points.
    """
    best_margin, best_player = -np.inf, -1
    for code in range(len(_POS)):
        if not pos_ok[code]:
            continue
        sel = eligible & (d["pos"] == code)
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        e_next = 0.0
        if my_next is not None:
            surv = conditional_survival(
                d["adp"][idx], d["stdev"][idx], current_pick, my_next
            )
            e_next = expected_best_available(d["points"][idx], surv)
        candidate = int(idx[int(np.argmin(d["blend_rank"][idx]))])
        margin = float(d["points"][candidate]) - e_next
        if margin > best_margin:
            best_margin, best_player = margin, candidate
    return best_player if best_player >= 0 else _board_choice(d, eligible)


def _dp_choice(
    d: dict[str, np.ndarray],
    eligible: np.ndarray,
    counts_row: np.ndarray,
    roster: RosterConfig,
    overall: int,
    my_picks: list[int],
) -> int:
    """Receding-horizon DP: plan positions for all my remaining picks, execute
    the first assignment (bench plan -> board choice)."""
    remaining = [p for p in my_picks if p >= overall]
    if not remaining:
        return _board_choice(d, eligible)
    dedicated = np.array(
        [roster.qb, roster.rb, roster.wr, roster.te, roster.dst, roster.k]
    )
    ded_need = np.maximum(dedicated - counts_row, 0)
    surplus = np.maximum(counts_row - dedicated, 0)
    flex_used = min(int(surplus[1] + surplus[2] + surplus[3]), roster.flex)
    flex_need = roster.flex - flex_used
    sf_surplus = int(surplus[0] + surplus[1] + surplus[2] + surplus[3]) - flex_used
    sf_need = max(roster.superflex - max(sf_surplus, 0), 0)

    pos_names = [str(_POS[c]) for c in d["pos"]]
    v = plan_value_matrix(
        pos_names,
        d["value"],  # market-anchored, NOT raw model points
        d["blend_rank"],
        d["adp"],
        d["stdev"],
        eligible,
        overall,
        remaining,
        list(_POS),
    )
    plan = optimal_plan(
        v,
        list(_POS),
        {name: int(ded_need[i]) for i, name in enumerate(_POS)},
        flex_need,
        sf_need,
        remaining,
    )
    target = plan.positions[0]
    if target is None:
        return _board_choice(d, eligible)
    sel = eligible & (d["pos"] == _POS_CODE[target])
    if not sel.any():
        return _board_choice(d, eligible)
    idx = np.where(sel)[0]
    return int(idx[int(np.argmin(d["blend_rank"][idx]))])


def run_paired_draft(
    d: dict[str, np.ndarray],
    roster: RosterConfig,
    my_slot: int,
    opp_noise: np.ndarray,
    my_policy: str,
    teams: int = 12,
    rounds: int = ROUNDS,
) -> tuple[float, np.ndarray]:
    """One draft; returns (my team's lineup value, all teams' values)."""
    n = len(d["pos"])
    caps = _caps(roster)
    slots = _lineup_slots(roster)
    order = _snake_order(teams, rounds)
    my_team = my_slot - 1
    my_picks = _my_pick_numbers(my_slot, teams, rounds)

    available = np.ones(n, dtype=bool)
    counts = np.zeros((teams, len(_POS)), dtype=int)
    picks_left = np.full(teams, rounds)
    rosters_c: list[list[int]] = [[] for _ in range(teams)]
    rosters_p: list[list[float]] = [[] for _ in range(teams)]
    adp_pri = d["adp_rank"] + opp_noise

    for overall, team in enumerate(order, start=1):
        cap_ok = counts[team][d["pos"]] < caps[d["pos"]]
        eligible = available & cap_ok
        need_total, helps = _needs(counts[team], roster)
        if need_total >= picks_left[team]:
            forced = eligible & helps[d["pos"]]
            if forced.any():
                eligible = forced
        if not eligible.any():
            picks_left[team] -= 1
            continue

        if team == my_team:
            if my_policy == "board":
                pick = _board_choice(d, eligible)
            elif my_policy == "greedy":
                need_mask = helps if need_total > 0 else np.ones(len(_POS), bool)
                pos_ok = np.array(
                    [
                        bool(need_mask[c] and (eligible & (d["pos"] == c)).any())
                        for c in range(len(_POS))
                    ]
                )
                if need_total > 0 and pos_ok.any():
                    my_next = next((p for p in my_picks if p > overall), None)
                    pick = _strategy_choice(d, eligible, pos_ok, overall, my_next)
                else:
                    pick = _board_choice(d, eligible)
            else:  # dp: full-plan receding horizon
                pick = _dp_choice(d, eligible, counts[team], roster, overall, my_picks)
        else:
            pick = int(np.argmin(np.where(eligible, adp_pri, np.inf)))

        available[pick] = False
        counts[team][d["pos"][pick]] += 1
        picks_left[team] -= 1
        rosters_c[team].append(int(d["pos"][pick]))
        rosters_p[team].append(float(d["actual"][pick]))

    values = np.array(
        [_lineup_value(rosters_c[t], rosters_p[t], slots) for t in range(teams)]
    )
    return float(values[my_team]), values


def evaluate_strategy(
    pool: pl.DataFrame,
    roster: RosterConfig,
    n_sims: int = 400,
    seed: int = 7,
) -> dict[str, float]:
    """Paired evaluation of board / greedy / dp policies on one season's pool."""
    d = _prepare(pool)
    rng = np.random.default_rng(seed)
    n = len(d["pos"])
    policies = ("board", "greedy", "dp")
    values_by: dict[str, list[float]] = {p: [] for p in policies}
    rows: dict[str, list[tuple[float, int]]] = {p: [] for p in policies}
    for _ in range(n_sims):
        slot = int(rng.integers(1, 13))
        noise = rng.normal(0.0, np.clip(d["stdev"], 2.0, 25.0), n)
        for policy in policies:
            mine, values = run_paired_draft(d, roster, slot, noise, policy)
            others = np.delete(values, slot - 1)
            finish = 1 + int((others > mine).sum())
            values_by[policy].append(mine)
            rows[policy].append((mine - float(others.mean()), finish))

    out: dict[str, float] = {}
    for policy in policies:
        out[f"{policy}_edge"] = float(np.mean([r[0] for r in rows[policy]]))
        out[f"{policy}_tophalf"] = float(np.mean([r[1] <= 6 for r in rows[policy]]))
        out[f"{policy}_title"] = float(np.mean([r[1] == 1 for r in rows[policy]]))
    for policy in ("greedy", "dp"):
        deltas = np.array(values_by[policy]) - np.array(values_by["board"])
        out[f"{policy}_delta"] = float(deltas.mean())
        out[f"{policy}_delta_se"] = float(deltas.std() / np.sqrt(len(deltas)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy vs board, paired 1v11.")
    parser.add_argument("--start", type=int, default=2019)
    parser.add_argument("--through", type=int, default=2024)
    parser.add_argument("--sims", type=int, default=400)
    args = parser.parse_args()

    panel = read_table("feature_panel")
    players = read_table("players")
    roster = get_format("redraft_ppr").roster
    seasons = list(range(args.start, args.through + 1))
    print(f"[strategy-sim] paired 1v11, seasons {seasons[0]}-{seasons[-1]}")
    all_rows = []
    for y in seasons:
        pool = build_pool(panel, players, y, "baseline", "redraft_ppr")
        m = evaluate_strategy(pool, roster, n_sims=args.sims)
        all_rows.append(m)
        print(
            f"  {y}: board th {m['board_tophalf']:.1%} ti {m['board_title']:.1%} | "
            f"greedy delta {m['greedy_delta']:+6.1f} | "
            f"dp th {m['dp_tophalf']:.1%} ti {m['dp_title']:.1%} "
            f"delta {m['dp_delta']:+6.1f} (se {m['dp_delta_se']:.1f})"
        )
    mean = {k: float(np.mean([r[k] for r in all_rows])) for k in all_rows[0]}
    print(
        f"[strategy-sim] MEAN: board th {mean['board_tophalf']:.1%} "
        f"ti {mean['board_title']:.1%} | greedy delta {mean['greedy_delta']:+.1f} | "
        f"DP th {mean['dp_tophalf']:.1%} ti {mean['dp_title']:.1%} "
        f"delta {mean['dp_delta']:+.1f} pts/season"
    )


if __name__ == "__main__":
    main()
