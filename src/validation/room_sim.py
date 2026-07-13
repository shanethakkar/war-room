"""Stage 1 of the room-bias research: can room-awareness beat the board?

    uv run python -m src.validation.room_sim [--sims 150]

Rooms deviate from market timing; FFC mocks bound how much (Stage 0). This sim
answers, paired (same seeds/slots/noise per comparison), across 2019-2024:

A. BIAS SWEEP - bots draft with positional timing shifts (grounded: QB -4/-8,
   K/DST -12/-24, RB -8; plus a labeled beyond-observed "folklore" QB -16).
   Policies: board (baseline) / oracle-DP (knows the true room bias) /
   estimator-DP (infers bias online from observed picks, shrunk). Kill-gates,
   pre-registered: oracle gain < ~15 pts/season at plausible bias -> direction
   dies; estimator < 1/3 of oracle -> display-only at most.

B. RUN QUESTION - herding bots (contagion calibrated to Stage-0 dispersion
   ceilings): when a positional run starts, is it better to JOIN (take that
   position now, the VONA instinct) or stay on the board (implicitly fading
   the run and harvesting fallers)?

Pre-registered expectations (2026-07-12, before running): board improves
absolutely in biased rooms (reaches create fallers it harvests); oracle gain
small at plausible bias, meaningful only in folklore regimes; estimator captures
a minority of oracle; JOIN loses to board except possibly for QB/TE runs when a
starter slot is still open.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from src.decision.strategy import optimal_plan, plan_value_matrix
from src.formats import get_format
from src.formats.base import RosterConfig
from src.ingest.cache import read_table
from src.validation.draft_sim import (
    _POS,
    ROUNDS,
    _caps,
    _lineup_slots,
    _lineup_value,
    _needs,
    _rank,
    _snake_order,
    build_pool,
)
from src.validation.strategy_sim import _board_choice, _my_pick_numbers, _prepare

# Estimator shrinkage: bias estimate = mean residual * n / (n + K_SHRINK).
K_SHRINK = 8.0
# Run detection: >= RUN_COUNT picks at one position within the last RUN_WINDOW.
RUN_WINDOW = 5
RUN_COUNT = 3
# JOIN guard: only join a run if the position's best is within this many blend
# ranks of the overall best available (don't reach absurdly).
JOIN_MAX_REACH = 24
# Adaptive gating: switch from board to room-aware DP only when the estimated
# positional bias is big AND confidently observed (the control-room DP cost of
# ~-12/season must not be paid in normal rooms). Gate on the UNSHRUNK mean:
# thresholding the shrunk estimate double-counts caution (a true bias of 8
# never crosses a shrunk threshold of 6 until ~24 observations - too late).
ADAPT_MIN_SHIFT = 5.0
ADAPT_MIN_OBS = 8

BIAS_SCENARIOS: dict[str, dict[str, float]] = {
    "control": {},
    "qb_early_4": {"QB": -4.0},
    "qb_early_8": {"QB": -8.0},
    "rb_early_8": {"RB": -8.0},
    "kdst_early_12": {"K": -12.0, "DST": -12.0},
    "kdst_early_24": {"K": -24.0, "DST": -24.0},
    "folklore_qb_16": {"QB": -16.0},
}


def _bias_shift(d: dict[str, np.ndarray], bias: dict[str, float]) -> np.ndarray:
    shift = np.zeros(len(d["pos"]))
    for pos_name, delta in bias.items():
        shift[d["pos"] == _POS.index(pos_name)] = delta
    return shift


def _dp_pick(
    d: dict[str, np.ndarray],
    eff_adp: np.ndarray,
    eligible: np.ndarray,
    counts_row: np.ndarray,
    roster: RosterConfig,
    overall: int,
    my_picks: list[int],
) -> int:
    """DP choice with survival driven by ``eff_adp`` (oracle or estimated)."""
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
        d["value"],
        d["blend_rank"],
        eff_adp,
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
    sel = eligible & (d["pos"] == _POS.index(target))
    if not sel.any():
        return _board_choice(d, eligible)
    idx = np.where(sel)[0]
    return int(idx[int(np.argmin(d["blend_rank"][idx]))])


def run_room_draft(
    d: dict[str, np.ndarray],
    roster: RosterConfig,
    my_slot: int,
    opp_noise: np.ndarray,
    my_policy: str,
    bias: dict[str, float],
    herd_boost: float = 0.0,
    noise_scale: float = 1.0,
    teams: int = 12,
    rounds: int = ROUNDS,
    record_picks: list[tuple[int, float]] | None = None,
) -> tuple[float, np.ndarray]:
    """One draft in a (possibly biased/herding) room.

    Policies: board | oracle (DP, knows true bias) | estimate (DP, online
    shrunk bias estimate) | join (board + join detected runs).
    ``record_picks`` collects (player_index, overall) for dispersion calibration.
    """
    n = len(d["pos"])
    caps = _caps(roster)
    slots = _lineup_slots(roster)
    order = _snake_order(teams, rounds)
    my_team = my_slot - 1
    my_picks = _my_pick_numbers(my_slot, teams, rounds)

    shift = _bias_shift(d, bias)
    room_adp = d["adp"] + shift  # the room's true effective ADP
    bot_base = _rank(np.nan_to_num(room_adp, nan=400.0), descending=False)
    bot_pri = bot_base + opp_noise * noise_scale

    available = np.ones(n, dtype=bool)
    counts = np.zeros((teams, len(_POS)), dtype=int)
    picks_left = np.full(teams, rounds)
    rosters_c: list[list[int]] = [[] for _ in range(teams)]
    rosters_p: list[list[float]] = [[] for _ in range(teams)]
    recent: deque[int] = deque(maxlen=RUN_WINDOW)
    # Online bias estimator state (per position): sum of residuals, count.
    est_sum = np.zeros(len(_POS))
    est_n = np.zeros(len(_POS))

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
            elif my_policy == "oracle":
                pick = _dp_pick(
                    d, room_adp, eligible, counts[team], roster, overall, my_picks
                )
            elif my_policy in ("estimate", "adaptive"):
                est_shift = np.where(
                    est_n > 0,
                    est_sum / np.maximum(est_n, 1) * est_n / (est_n + K_SHRINK),
                    0.0,
                )
                raw_mean = est_sum[:4] / np.maximum(est_n[:4], 1)
                triggered = bool(
                    np.any(
                        (np.abs(raw_mean) >= ADAPT_MIN_SHIFT)
                        & (est_n[:4] >= ADAPT_MIN_OBS)
                    )
                )
                if my_policy == "adaptive" and not triggered:
                    pick = _board_choice(d, eligible)  # normal room: stay on board
                else:
                    eff = d["adp"] + est_shift[d["pos"]]
                    pick = _dp_pick(
                        d, eff, eligible, counts[team], roster, overall, my_picks
                    )
            else:  # join
                pick = _board_choice(d, eligible)
                run_pos = [
                    c
                    for c in range(len(_POS))
                    if list(recent).count(c) >= RUN_COUNT and helps[c]
                ]
                if run_pos:
                    c = run_pos[0]
                    sel = eligible & (d["pos"] == c)
                    if sel.any():
                        idx = np.where(sel)[0]
                        cand = int(idx[int(np.argmin(d["blend_rank"][idx]))])
                        overall_best = _board_choice(d, eligible)
                        if (
                            d["blend_rank"][cand] - d["blend_rank"][overall_best]
                            <= JOIN_MAX_REACH
                        ):
                            pick = cand
        else:
            pri = bot_pri.copy()
            if herd_boost > 0 and len(recent) >= 2:
                for c in set(recent):
                    if list(recent).count(c) >= 2:
                        pri[d["pos"] == c] -= herd_boost
            pick = int(np.argmin(np.where(eligible, pri, np.inf)))
            # Feed the estimator with observed bot behavior.
            if not np.isnan(d["adp"][pick]):
                code = int(d["pos"][pick])
                est_sum[code] += overall - d["adp"][pick]
                est_n[code] += 1

        available[pick] = False
        counts[team][d["pos"][pick]] += 1
        picks_left[team] -= 1
        recent.append(int(d["pos"][pick]))
        rosters_c[team].append(int(d["pos"][pick]))
        rosters_p[team].append(float(d["actual"][pick]))
        if record_picks is not None:
            record_picks.append((pick, float(overall)))

    values = np.array(
        [_lineup_value(rosters_c[t], rosters_p[t], slots) for t in range(teams)]
    )
    return float(values[my_team]), values


def evaluate_scenario(
    pools: dict[int, dict[str, np.ndarray]],
    roster: RosterConfig,
    policies: tuple[str, ...],
    bias: dict[str, float],
    herd_boost: float = 0.0,
    noise_scale: float = 1.0,
    n_sims: int = 150,
    seed: int = 11,
) -> dict[str, float]:
    """Mean paired delta vs 'board' per policy, pooled over seasons."""
    deltas: dict[str, list[float]] = {p: [] for p in policies if p != "board"}
    for d in pools.values():
        rng = np.random.default_rng(seed)
        n = len(d["pos"])
        for _ in range(n_sims):
            slot = int(rng.integers(1, 13))
            noise = rng.normal(0.0, np.clip(d["stdev"], 2.0, 25.0), n)
            base, _ = run_room_draft(
                d, roster, slot, noise, "board", bias, herd_boost, noise_scale
            )
            for policy in deltas:
                mine, _ = run_room_draft(
                    d, roster, slot, noise, policy, bias, herd_boost, noise_scale
                )
                deltas[policy].append(mine - base)
    out: dict[str, float] = {}
    for policy, values in deltas.items():
        arr = np.array(values)
        out[f"{policy}_delta"] = float(arr.mean())
        out[f"{policy}_se"] = float(arr.std() / np.sqrt(len(arr)))
    return out


def calibrate_dispersion(
    d: dict[str, np.ndarray],
    roster: RosterConfig,
    herd_boost: float,
    noise_scale: float,
    n_sims: int = 60,
    seed: int = 5,
) -> dict[str, float]:
    """Realized per-player pick stdev by phase for a (herding, noise) setting."""
    rng = np.random.default_rng(seed)
    n = len(d["pos"])
    picks_by_player: dict[int, list[float]] = {}
    for _ in range(n_sims):
        rec: list[tuple[int, float]] = []
        noise = rng.normal(0.0, np.clip(d["stdev"], 2.0, 25.0), n)
        run_room_draft(
            d,
            roster,
            int(rng.integers(1, 13)),
            noise,
            "board",
            {},
            herd_boost,
            noise_scale,
            record_picks=rec,
        )
        for player, overall in rec:
            picks_by_player.setdefault(player, []).append(overall)
    rows = [
        (float(np.mean(v)), float(np.std(v)))
        for v in picks_by_player.values()
        if len(v) >= 20
    ]
    out = {}
    for name, lo, hi in (("R1-3", 1, 36), ("R4-8", 37, 96), ("R9-15", 97, 180)):
        phase = [sd for mean, sd in rows if lo <= mean <= hi]
        out[name] = float(np.median(phase)) if phase else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: room-bias experiments.")
    parser.add_argument("--start", type=int, default=2019)
    parser.add_argument("--through", type=int, default=2024)
    parser.add_argument("--sims", type=int, default=150)
    args = parser.parse_args()

    panel = read_table("feature_panel")
    players = read_table("players")
    roster = get_format("redraft_ppr").roster
    seasons = list(range(args.start, args.through + 1))
    pools = {
        y: _prepare(build_pool(panel, players, y, "baseline", "redraft_ppr"))
        for y in seasons
    }

    print("[stage1] A. bias sweep (paired deltas vs board, pts/season)")
    for name, bias in BIAS_SCENARIOS.items():
        m = evaluate_scenario(
            pools, roster, ("board", "oracle", "estimate"), bias, n_sims=args.sims
        )
        print(
            f"  {name:16} oracle {m['oracle_delta']:+7.1f} (se {m['oracle_se']:.1f})"
            f" | estimator {m['estimate_delta']:+7.1f} (se {m['estimate_se']:.1f})"
        )

    print("[stage1] B. herding calibration (player pick stdev by phase)")
    sample = pools[seasons[-1]]
    for herd, scale in ((0.0, 1.0), (6.0, 0.85), (12.0, 0.7)):
        disp = calibrate_dispersion(sample, roster, herd, scale)
        print(f"  herd={herd:4.1f} noise x{scale:.2f}: {disp}")

    print("[stage1] B. run question: JOIN vs board (paired deltas, pts/season)")
    for herd, scale in ((6.0, 0.85), (12.0, 0.7)):
        m = evaluate_scenario(
            pools, roster, ("board", "join"), {}, herd, scale, n_sims=args.sims
        )
        print(
            f"  herd={herd:4.1f}: join {m['join_delta']:+7.1f} (se {m['join_se']:.1f})"
        )


if __name__ == "__main__":
    main()
