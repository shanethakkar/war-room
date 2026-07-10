"""Ingestion entrypoint: pull nflverse tables into the local Parquet cache.

    uv run python -m src.ingest.refresh [--start-season Y] [--end-season Y]
                                        [--only NAME ...] [--list]

Iterates the ``sources.TABLES`` registry, fetches each table for the training
window, and writes it to ``data/cache/<name>.parquet``. The window defaults to
``DATA_START_SEASON``..current season (``nflreadpy.get_current_season()``).
Per-table failures are reported but do not abort the run, so one flaky feed does
not cost you the rest of the cache.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import nflreadpy as nfl

from src.config import CACHE_DIR, DATA_START_SEASON
from src.ingest.cache import write_table
from src.ingest.sources import TABLES, TABLES_BY_NAME, NflverseTable


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of refreshing one table."""

    name: str
    ok: bool
    rows: int = 0
    cols: int = 0
    n_bytes: int = 0
    error: str = ""


def _resolve_seasons(start_season: int, end_season: int) -> list[int]:
    """Inclusive season window, validated."""
    if end_season < start_season:
        raise ValueError(
            f"end_season ({end_season}) precedes start_season ({start_season})."
        )
    if start_season < DATA_START_SEASON:
        raise ValueError(
            f"start_season ({start_season}) precedes the data window "
            f"({DATA_START_SEASON})."
        )
    return list(range(start_season, end_season + 1))


def _select(only: Sequence[str] | None) -> list[NflverseTable]:
    """Resolve the ``--only`` subset to registry entries, or all tables."""
    if only is None:
        return list(TABLES)
    unknown = [n for n in only if n not in TABLES_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown table(s): {unknown}. Known: {sorted(TABLES_BY_NAME)}."
        )
    return [TABLES_BY_NAME[n] for n in only]


def refresh_table(table: NflverseTable, seasons: Sequence[int]) -> RefreshResult:
    """Fetch one table and cache it, capturing any failure as a result."""
    try:
        df = table.fetch(seasons)
        path = write_table(table.name, df)
        return RefreshResult(table.name, True, df.height, df.width, path.stat().st_size)
    except Exception as exc:  # report, don't abort the whole run
        return RefreshResult(table.name, False, error=f"{type(exc).__name__}: {exc}")


def refresh(
    *,
    start_season: int = DATA_START_SEASON,
    end_season: int | None = None,
    only: Sequence[str] | None = None,
) -> list[RefreshResult]:
    """Refresh the selected tables for the season window into the cache.

    ``end_season`` defaults to the current NFL season. Returns one result per
    table; callers inspect ``ok`` to detect partial failures.
    """
    if end_season is None:
        end_season = int(nfl.get_current_season())
    seasons = _resolve_seasons(start_season, end_season)
    return [refresh_table(table, seasons) for table in _select(only)]


def _print_report(results: list[RefreshResult]) -> int:
    """Print a per-table report; return the count of failures."""
    failures = 0
    for r in results:
        if r.ok:
            print(
                f"  OK   {r.name:<20} {r.rows:>9,} rows x {r.cols:>3} cols  "
                f"{r.n_bytes / 1e6:6.1f} MB"
            )
        else:
            failures += 1
            print(f"  FAIL {r.name:<20} {r.error}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the nflverse Parquet cache.")
    parser.add_argument(
        "--start-season",
        type=int,
        default=DATA_START_SEASON,
        help=f"First season to pull (default: {DATA_START_SEASON}).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season to pull (default: current NFL season).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Subset of table names to refresh (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the registered tables and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for table in TABLES:
            print(f"  {table.name:<20} {table.description}")
        return

    end_season = (
        args.end_season
        if args.end_season is not None
        else int(nfl.get_current_season())
    )
    tables = _select(args.only)
    print(
        f"[ingest] refreshing {len(tables)} table(s) for "
        f"{args.start_season}-{end_season} into {CACHE_DIR}"
    )
    results = refresh(
        start_season=args.start_season, end_season=end_season, only=args.only
    )
    failures = _print_report(results)
    print(
        f"[ingest] {len(results) - failures}/{len(results)} cached; {failures} failed."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
