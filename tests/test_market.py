"""ADP / arbitrage / backtest tests - network-free, synthetic data.

The FFC fetch itself needs the network (exercised by the CLIs); here we test the
pure pieces: name normalization, ADP frame shaping, arbitrage delta logic, and the
rank-correlation helper.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from src.decision.arbitrage import build_arbitrage
from src.ingest.adp import _normalize
from src.names import norm_name_expr
from src.validation.backtest import _spearman


def test_name_normalization() -> None:
    df = pl.DataFrame(
        {
            "name": [
                "Amon-Ra St. Brown",
                "Michael Pittman Jr.",
                "Ja'Marr Chase",
                "D.J. Moore",
            ]
        }
    ).with_columns(norm_name_expr("name"))
    assert df["norm_name"].to_list() == [
        "amon ra st brown",
        "michael pittman",
        "jamarr chase",
        "dj moore",
    ]


def test_adp_normalize_shapes_and_ranks() -> None:
    players: list[dict[str, Any]] = [
        {
            "name": "B Back",
            "position": "RB",
            "team": "SF",
            "adp": 5.0,
            "stdev": 1.0,
            "times_drafted": 100,
        },
        {
            "name": "A Receiver",
            "position": "WR",
            "team": "DAL",
            "adp": 2.0,
            "stdev": 0.5,
            "times_drafted": 120,
        },
    ]
    out = _normalize(players, year=2024, slug="ppr", teams=12)
    assert {"adp_name", "norm_name", "position", "adp", "adp_rank"} <= set(out.columns)
    # Lower ADP -> rank 1; frame sorted by adp ascending.
    assert out["adp_name"].to_list() == ["A Receiver", "B Back"]
    assert out.filter(pl.col("adp_name") == "A Receiver")["adp_rank"][0] == 1
    assert out["adp_year"][0] == 2024


def _board(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """(name, position_group, vor) -> a minimal value board."""
    return pl.DataFrame(
        {
            "player_name": [r[0] for r in rows],
            "position_group": [r[1] for r in rows],
            "team": ["SF"] * len(rows),
            "position_tier": [1] * len(rows),
            "projected_points": [r[2] for r in rows],
            "points_low": [r[2] * 0.6 for r in rows],
            "points_high": [r[2] * 1.4 for r in rows],
            "vor": [r[2] for r in rows],
        }
    )


def _adp(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """(name, position, adp) -> a minimal ADP frame."""
    return pl.DataFrame(
        {
            "adp_name": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "adp": [r[2] for r in rows],
            "adp_stdev": [1.0] * len(rows),
            "times_drafted": [50] * len(rows),
        }
    ).with_columns(norm_name_expr("adp_name"))


def test_arbitrage_delta_flags_targets_and_fades() -> None:
    # We value A far above B; the market (ADP) drafts B far ahead of A.
    board = _board([("A Player", "WR", 100.0), ("B Player", "WR", 50.0)])
    adp = _adp([("A Player", "WR", 30.0), ("B Player", "WR", 5.0)])
    arb = build_arbitrage(board, adp)
    a = arb.filter(pl.col("player_name") == "A Player").row(0, named=True)
    b = arb.filter(pl.col("player_name") == "B Player").row(0, named=True)
    assert a["arbitrage_delta"] > 0  # target: we rank higher than the market
    assert b["arbitrage_delta"] < 0  # fade: market ranks higher than us
    assert arb["player_name"][0] == "A Player"  # sorted targets-first


def test_spearman_helper() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    assert _spearman(df, "a", "b") == 1.0
    assert _spearman(df.head(2), "a", "b") is None  # too few rows


def test_load_adp_ttl_refreshes_current_year_only(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """Completed seasons cache forever; the draft-year board expires after the
    TTL; and a failed refetch serves the stale cache instead of raising."""
    import os
    import time

    import src.ingest.adp as adp_mod
    import src.ingest.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adp_mod.nfl, "get_current_season", lambda roster=False: 2025)
    calls: list[int] = []

    def fake_fetch(year: int, fmt: Any, teams: int) -> pl.DataFrame:
        calls.append(year)
        return _normalize(
            [
                {
                    "name": "A Receiver",
                    "position": "WR",
                    "team": "DAL",
                    "adp": 2.0,
                    "stdev": 0.5,
                    "times_drafted": 10,
                }
            ],
            year,
            "ppr",
            teams,
        )

    monkeypatch.setattr(adp_mod, "fetch_adp", fake_fetch)

    # Completed season: one fetch, then cached forever.
    adp_mod.load_adp(2024, "redraft_ppr")
    adp_mod.load_adp(2024, "redraft_ppr")
    assert calls == [2024]

    # Draft-year board: cached while fresh...
    adp_mod.load_adp(2026, "redraft_ppr")
    adp_mod.load_adp(2026, "redraft_ppr")
    assert calls == [2024, 2026]

    # ...but refetched once older than the TTL.
    path = cache_mod.cache_path("adp2_ppr_12_2026")
    aged = time.time() - (adp_mod.ADP_TTL_DAYS + 1) * 86400
    os.utime(path, (aged, aged))
    adp_mod.load_adp(2026, "redraft_ppr")
    assert calls == [2024, 2026, 2026]

    # Stale beats broken: a failing refetch serves the stale cache.
    def boom(year: int, fmt: Any, teams: int) -> pl.DataFrame:
        raise RuntimeError("FFC down")

    monkeypatch.setattr(adp_mod, "fetch_adp", boom)
    os.utime(path, (aged, aged))
    out = adp_mod.load_adp(2026, "redraft_ppr")
    assert out.height == 1
