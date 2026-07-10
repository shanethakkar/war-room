"""Parquet cache I/O for the ingestion layer.

The cache under ``data/cache`` is the **offline source of truth**: once refreshed,
every downstream layer reads from here and nothing projection-related touches the
network (constraint #3). All disk I/O for the data layer lives in this module so
the rest of the layer stays pure.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CACHE_DIR


def cache_path(name: str) -> Path:
    """Absolute Parquet path for a cached table."""
    return CACHE_DIR / f"{name}.parquet"


def write_table(name: str, df: pl.DataFrame) -> Path:
    """Write ``df`` to the cache as ``<name>.parquet`` and return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(name)
    df.write_parquet(path)
    return path


def read_table(name: str) -> pl.DataFrame:
    """Read a cached table, or raise if it hasn't been refreshed yet."""
    path = cache_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached table {name!r} at {path}. "
            f"Run `python -m src.ingest.refresh` first."
        )
    return pl.read_parquet(path)


def cached_tables() -> list[str]:
    """Names of all tables currently in the cache (sorted)."""
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob("*.parquet"))
