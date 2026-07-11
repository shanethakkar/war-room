"""Draft-simulation "wins" metric - what our edge is actually worth.

Rank correlation is a proxy; the real question is: does drafting off our board beat
drafting off ADP? For each season (leakage-free) we build the draftable pool (our
VOR board joined to ADP and to the actual season outcome), then run many Monte
Carlo snake drafts. Half the teams draft by **our board** (best VOR available);
half draft by **ADP + noise** (FFC's ``adp_stdev`` gives realistic draft variance).
Each roster is then scored by its *actual* optimal starting lineup. If our board is
better, our teams win more.

This is the honest scoreboard for baseline changes: tuning, aging, rookies - keep a
change only if it drafts better teams, not just a higher rank correlation.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from src.decision.board import build_value_board
from src.formats import FORMATS, get_format
from src.formats.base import RosterConfig
from src.ingest.adp import load_adp
from src.ingest.cache import read_table
from src.names import norm_name_expr
from src.projections import MODELS
from src.projections.pipeline import scored_projection

_POS = ("QB", "RB", "WR", "TE")
_POS_CODE = {p: i for i, p in enumerate(_POS)}
# Slot eligibility by position code: dedicated, FLEX (RB/WR/TE), SUPERFLEX (all).
_FLEX = frozenset({1, 2, 3})
_SUPERFLEX = frozenset({0, 1, 2, 3})

ROUNDS = 14
N_TEAMS = 12
N_SIMS = 200
# Draft variance, in picks, applied SYMMETRICALLY to both strategies (rank + noise)
# so the only thing that differs is the ranking quality - not a collision artifact
# from one bloc drafting deterministically while the other has noise.
NOISE_PICKS = 6.0


def _rank(values: np.ndarray, *, descending: bool) -> np.ndarray:
    """Dense 0-based rank of each element (rank 0 = best)."""
    order = np.argsort(-values if descending else values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _caps(roster: RosterConfig) -> np.ndarray:
    """Per-position draft caps (starters + headroom) so rosters stay sane."""
    return np.array(
        [
            roster.qb + roster.superflex + 1,
            roster.rb + roster.flex + roster.superflex + 3,
            roster.wr + roster.flex + roster.superflex + 3,
            roster.te + 2,
        ]
    )


def _lineup_slots(roster: RosterConfig) -> list[frozenset[int]]:
    """Starting-lineup slots (fill order), each an eligible-position-code set."""
    slots: list[frozenset[int]] = []
    slots += [frozenset({0})] * roster.qb
    slots += [frozenset({1})] * roster.rb
    slots += [frozenset({2})] * roster.wr
    slots += [frozenset({3})] * roster.te
    slots += [_FLEX] * roster.flex
    slots += [_SUPERFLEX] * roster.superflex
    return slots


def _lineup_value(
    roster_codes: list[int], roster_points: list[float], slots: list[frozenset[int]]
) -> float:
    """Greedy optimal starting lineup value from a roster's actual points.

    Slots are filled most-restrictive first (dedicated, then flex, then superflex);
    since eligibility is nested, greedy is optimal here.
    """
    used = [False] * len(roster_codes)
    total = 0.0
    for eligible in slots:
        best_i, best_pts = -1, float("-inf")
        for i, code in enumerate(roster_codes):
            if not used[i] and code in eligible and roster_points[i] > best_pts:
                best_i, best_pts = i, roster_points[i]
        if best_i >= 0:
            used[best_i] = True
            total += best_pts
    return total


def _snake_order(n_teams: int, rounds: int) -> list[int]:
    order: list[int] = []
    for r in range(rounds):
        order.extend(range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1))
    return order


def simulate_draft(
    pos_codes: np.ndarray,
    actual: np.ndarray,
    our_priority: np.ndarray,
    adp_priority: np.ndarray,
    our_mask: np.ndarray,
    caps: np.ndarray,
    slots: list[frozenset[int]],
    order: list[int],
) -> np.ndarray:
    """One draft (lower priority = drafted earlier): each team's lineup value.

    ``our_priority`` / ``adp_priority`` are already rank+noise arrays; the pick is
    the eligible player with the lowest priority for the team's strategy.
    """
    n_teams = len(our_mask)
    n_players = len(pos_codes)
    available = np.ones(n_players, dtype=bool)
    counts = np.zeros((n_teams, len(_POS)), dtype=int)
    rosters_c: list[list[int]] = [[] for _ in range(n_teams)]
    rosters_p: list[list[float]] = [[] for _ in range(n_teams)]

    for team in order:
        cap_ok = counts[team][pos_codes] < caps[pos_codes]
        eligible = available & cap_ok
        if not eligible.any():
            continue
        priority = our_priority if our_mask[team] else adp_priority
        pick = int(np.argmin(np.where(eligible, priority, np.inf)))
        available[pick] = False
        counts[team][pos_codes[pick]] += 1
        rosters_c[team].append(int(pos_codes[pick]))
        rosters_p[team].append(float(actual[pick]))

    return np.array(
        [_lineup_value(rosters_c[t], rosters_p[t], slots) for t in range(n_teams)]
    )


def evaluate_pool(
    pool: pl.DataFrame,
    roster: RosterConfig,
    *,
    n_sims: int = N_SIMS,
    n_teams: int = N_TEAMS,
    rounds: int = ROUNDS,
    seed: int = 0,
) -> dict[str, float]:
    """Run ``n_sims`` drafts (half our-board, half ADP) and aggregate the edge."""
    pos_codes = np.array([_POS_CODE[p] for p in pool["position_group"].to_list()])
    actual = pool["actual_points"].to_numpy().astype(float)
    our_rank = _rank(pool["vor"].to_numpy().astype(float), descending=True)
    adp_rank = _rank(pool["adp"].to_numpy().astype(float), descending=False)
    n_players = len(pos_codes)
    caps = _caps(roster)
    slots = _lineup_slots(roster)
    order = _snake_order(n_teams, rounds)
    half = n_teams // 2

    rng = np.random.default_rng(seed)
    margins, our_avg, adp_avg, win_rates = [], [], [], []
    for _ in range(n_sims):
        our_mask = np.zeros(n_teams, dtype=bool)
        our_mask[rng.choice(n_teams, half, replace=False)] = True
        our_pri = our_rank + rng.normal(0.0, NOISE_PICKS, n_players)
        adp_pri = adp_rank + rng.normal(0.0, NOISE_PICKS, n_players)
        values = simulate_draft(
            pos_codes, actual, our_pri, adp_pri, our_mask, caps, slots, order
        )
        ours = values[our_mask]
        theirs = values[~our_mask]
        margins.append(float(ours.mean() - theirs.mean()))
        our_avg.append(float(ours.mean()))
        adp_avg.append(float(theirs.mean()))
        top_half = set(np.argsort(values)[::-1][:half].tolist())
        our_teams = np.where(our_mask)[0]
        win_rates.append(float(np.mean([t in top_half for t in our_teams])))

    return {
        "margin": float(np.mean(margins)),
        "our_value": float(np.mean(our_avg)),
        "adp_value": float(np.mean(adp_avg)),
        "win_rate": float(np.mean(win_rates)),
    }


def build_pool(
    panel: pl.DataFrame, players: pl.DataFrame, season: int, model: str, fmt_key: str
) -> pl.DataFrame:
    """Draftable pool for ``season``: our VOR board ∩ ADP ∩ actual outcome."""
    scored = scored_projection(panel, players, season, model=model, fmt_key=fmt_key)
    board = build_value_board(scored, get_format(fmt_key)).with_columns(
        norm_name_expr("player_name")
    )
    actual = panel.filter((pl.col("season") == season) & (pl.col("games") >= 1)).select(
        "player_id", pl.col("fantasy_points_ppr").alias("actual_points")
    )
    adp = load_adp(season, fmt_key).select("norm_name", "position", "adp")
    return (
        board.join(actual, on="player_id", how="inner")
        .join(
            adp,
            left_on=["norm_name", "position_group"],
            right_on=["norm_name", "position"],
            how="inner",
        )
        .filter(pl.col("position_group").is_in(_POS))
    )


def apply_blend(pool: pl.DataFrame, model_weight: float) -> pl.DataFrame:
    """Replace the pool's ranking score with the market-anchored blend.

    Mirrors ``decision.blend``: score = w * model-VOR-rank + (1-w) * ADP-rank
    (negated so 'higher = better' matches the sim's convention).
    """
    blended = (
        model_weight * pool["vor"].rank(descending=True)
        + (1 - model_weight) * pool["adp"].rank()
    )
    return pool.with_columns((-blended).alias("vor"))


def draft_sim(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    seasons: list[int],
    model: str = "baseline",
    fmt_key: str = "redraft_ppr",
    n_sims: int = N_SIMS,
    blend_weight: float | None = None,
) -> pl.DataFrame:
    """Per-season draft-simulation report of our board vs ADP.

    ``blend_weight`` evaluates the market-anchored blend (the shipped ranking)
    instead of the pure model board; gate changes on this.
    """
    roster = get_format(fmt_key).roster
    rows: list[dict[str, object]] = []
    for year in seasons:
        pool = build_pool(panel, players, year, model, fmt_key)
        if blend_weight is not None:
            pool = apply_blend(pool, blend_weight)
        metrics = evaluate_pool(pool, roster, n_sims=n_sims)
        rows.append({"season": year, "pool_n": pool.height, **metrics})
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft-simulation wins metric: our board vs ADP."
    )
    parser.add_argument("--start", type=int, default=2021)
    parser.add_argument("--through", type=int, default=2024)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument(
        "--format", dest="fmt_key", choices=sorted(FORMATS), default="redraft_ppr"
    )
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument(
        "--blend",
        type=float,
        default=None,
        metavar="W",
        help="Evaluate the market-anchored blend at model weight W (the shipped "
        "ranking uses decision.blend.MODEL_WEIGHT) instead of the pure board.",
    )
    args = parser.parse_args()

    panel = read_table("feature_panel")
    players = read_table("players")
    seasons = list(range(args.start, args.through + 1))
    label = f"blend w={args.blend}" if args.blend is not None else "pure board"
    print(
        f"[draft-sim] {args.model} {args.fmt_key} ({label}): seasons "
        f"{seasons[0]}-{seasons[-1]}, {args.sims} sims x {N_TEAMS} teams"
    )
    report = draft_sim(
        panel, players, seasons, args.model, args.fmt_key, args.sims, args.blend
    )
    with pl.Config(tbl_rows=report.height, tbl_hide_dataframe_shape=True):
        print(
            report.select(
                "season",
                "pool_n",
                pl.col("our_value").round(1),
                pl.col("adp_value").round(1),
                pl.col("margin").round(1),
                pl.col("win_rate").round(3),
            )
        )
    margin_mean = float(report["margin"].to_numpy().mean())
    win_rate_mean = float(report["win_rate"].to_numpy().mean())
    print(
        f"[draft-sim] means: margin={margin_mean:+.1f} pts/team  "
        f"win_rate={win_rate_mean:.1%} "
        f"(0.5 = no edge; higher = our board drafts better teams)"
    )


if __name__ == "__main__":
    main()
