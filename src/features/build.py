"""Feature entrypoint: cached nflverse tables -> player-season feature panel.

    uv run python -m src.features.build [--summary]

Reads the cached tables (``player_stats_season``, ``ff_opportunity``,
``snap_counts``, ``players``), assembles the panel via the pure transforms in
``panel.py``, and writes it to ``data/cache/feature_panel.parquet``. All I/O is
here; the transforms stay pure and testable.
"""

from __future__ import annotations

import argparse

import polars as pl

from src.features.panel import assemble_panel
from src.ingest.cache import read_table, write_table

PANEL_NAME = "feature_panel"

# Source tables the panel is built from.
_SOURCES = ("player_stats_season", "ff_opportunity", "snap_counts", "players")


def build_panel() -> pl.DataFrame:
    """Assemble the player-season feature panel and cache it."""
    sources = {name: read_table(name) for name in _SOURCES}
    panel = assemble_panel(
        player_stats_season=sources["player_stats_season"],
        ff_opportunity=sources["ff_opportunity"],
        snap_counts=sources["snap_counts"],
        players=sources["players"],
    )
    write_table(PANEL_NAME, panel)
    return panel


def _coverage(panel: pl.DataFrame, column: str) -> float:
    """Fraction of rows with a non-null value in ``column`` (0-1)."""
    if panel.height == 0:
        return 0.0
    return 1.0 - panel[column].null_count() / panel.height


def _print_summary(panel: pl.DataFrame) -> None:
    """Print coverage/sanity stats so the build is verifiable at a glance."""
    seasons = sorted(panel["season"].unique().to_list())
    by_pos = (
        panel.group_by("position_group").len().sort("len", descending=True).to_dicts()
    )
    print(f"[features] panel: {panel.height:,} rows x {panel.width} cols")
    print(f"[features] seasons: {seasons[0]}-{seasons[-1]}")
    print(
        "[features] by position_group: "
        + ", ".join(f"{r['position_group']}={r['len']:,}" for r in by_pos)
    )
    print(
        f"[features] coverage: "
        f"snap_share={_coverage(panel, 'snap_share'):.1%}, "
        f"expected_pts={_coverage(panel, 'expected_fantasy_points'):.1%}, "
        f"age={_coverage(panel, 'age'):.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the player-season feature panel from the cache."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print coverage/sanity stats after building.",
    )
    args = parser.parse_args()

    panel = build_panel()
    print(f"[features] wrote {PANEL_NAME} ({panel.height:,} rows) to the cache.")
    if args.summary:
        _print_summary(panel)


if __name__ == "__main__":
    main()
