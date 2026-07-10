"""Ingestion-layer tests — all network-free.

They exercise the cache I/O, the table registry, and the season-window logic
using synthetic data. The actual nflverse pull is verified by running
``python -m src.ingest.refresh`` (it hits the network, so it is not a unit test).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import src.ingest.cache as cache
from src.config import DATA_START_SEASON
from src.ingest.refresh import _resolve_seasons, _select
from src.ingest.sources import TABLES, TABLES_BY_NAME


def test_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = cache.write_table("demo", df)
    assert path == tmp_path / "demo.parquet"
    assert path.exists()
    assert cache.read_table("demo").equals(df)
    assert cache.cached_tables() == ["demo"]


def test_read_missing_table_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.read_table("nope")


def test_cached_tables_empty_when_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "missing")
    assert cache.cached_tables() == []


def test_registry_includes_verified_names() -> None:
    # The two open-question loaders resolved during scaffolding, plus the §9 core.
    essentials = {
        "pbp",
        "player_stats_week",
        "player_stats_season",
        "ff_opportunity",
        "snap_counts",  # verified: load_snap_counts
        "draft_picks",  # verified: load_draft_picks
        "rosters",
        "schedules",
        "players",
        "teams",
    }
    assert essentials <= set(TABLES_BY_NAME)


def test_registry_names_unique() -> None:
    names = [t.name for t in TABLES]
    assert len(names) == len(set(names))


def test_resolve_seasons_inclusive() -> None:
    assert _resolve_seasons(2016, 2018) == [2016, 2017, 2018]


def test_resolve_seasons_rejects_reversed_window() -> None:
    with pytest.raises(ValueError):
        _resolve_seasons(2020, 2019)


def test_resolve_seasons_rejects_pre_window() -> None:
    with pytest.raises(ValueError):
        _resolve_seasons(DATA_START_SEASON - 1, DATA_START_SEASON)


def test_select_all_when_none() -> None:
    assert _select(None) == list(TABLES)


def test_select_subset_preserves_order() -> None:
    chosen = _select(["schedules", "teams"])
    assert [t.name for t in chosen] == ["schedules", "teams"]


def test_select_unknown_raises() -> None:
    with pytest.raises(ValueError):
        _select(["not_a_real_table"])
