"""Ingestion entrypoint: pull nflverse tables → local Parquet cache.

    uv run python -m src.ingest.refresh [--start-season YEAR]

Not implemented yet — this is the next Phase 1 task (see progress.md).
"""

from __future__ import annotations

import argparse

from src.config import CACHE_DIR, DATA_START_SEASON


def refresh(*, start_season: int = DATA_START_SEASON) -> None:
    """Pull nflverse tables from ``start_season`` to present and cache to Parquet.

    Planned tables (confirm exact ``nflreadpy`` signatures at implementation time
    — see design.md §9): ``load_player_stats`` (weekly + seasonal),
    ``load_ff_opportunity``, ``load_pbp``, ``load_players`` / ``load_rosters``,
    ``load_schedules``, plus snap-count and draft-capital datasets.
    """
    raise NotImplementedError(
        "Ingestion is not implemented yet — next Phase 1 task. See progress.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the nflverse Parquet cache.")
    parser.add_argument(
        "--start-season",
        type=int,
        default=DATA_START_SEASON,
        help=f"First season to pull (default: {DATA_START_SEASON}).",
    )
    args = parser.parse_args()
    print(
        f"[ingest] scaffold only — would refresh nflverse cache into {CACHE_DIR} "
        f"for {args.start_season}–present. Not implemented yet (see progress.md)."
    )


if __name__ == "__main__":
    main()
