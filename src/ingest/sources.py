"""Registry of nflverse tables to cache (design.md §9).

Maps a stable cache name to the ``nflreadpy`` loader that produces it. Three call
conventions:

- **seasoned** — loader receives the resolved training window (pbp, weekly/seasonal
  stats, ff_opportunity, snap counts, rosters, schedules).
- **all-history** — loader pulls every available season; used for ``draft_picks``
  because a player's draft capital predates the window (rookie priors, design.md
  §4.4).
- **reference** — season-agnostic loader taking no season argument (``players``,
  ``teams``).

All loader names verified against nflreadpy 0.1.5. Adding a table = one entry
here; the refresh entrypoint iterates this registry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import nflreadpy as nfl
import polars as pl

# Given the resolved season window, return the raw Polars frame for a table.
Fetch = Callable[[Sequence[int]], pl.DataFrame]


@dataclass(frozen=True)
class NflverseTable:
    """A cacheable nflverse table: stable name, human description, fetch callable."""

    name: str
    description: str
    fetch: Fetch


def _seasoned(loader: Callable[..., pl.DataFrame], **kwargs: object) -> Fetch:
    """Wrap a loader so it is called with the resolved season window."""

    def fetch(seasons: Sequence[int]) -> pl.DataFrame:
        df: pl.DataFrame = loader(seasons=list(seasons), **kwargs)
        return df

    return fetch


def _all_history(loader: Callable[..., pl.DataFrame]) -> Fetch:
    """Wrap a loader so it pulls every available season (``seasons=True``)."""

    def fetch(seasons: Sequence[int]) -> pl.DataFrame:
        df: pl.DataFrame = loader(seasons=True)
        return df

    return fetch


def _reference(loader: Callable[[], pl.DataFrame]) -> Fetch:
    """Wrap a season-agnostic reference loader (no season argument)."""

    def fetch(seasons: Sequence[int]) -> pl.DataFrame:
        df: pl.DataFrame = loader()
        return df

    return fetch


TABLES: tuple[NflverseTable, ...] = (
    NflverseTable(
        "pbp",
        "Play-by-play: team pace, pass rate, red-zone touches, air yards.",
        _seasoned(nfl.load_pbp),
    ),
    NflverseTable(
        "player_stats_week",
        "Weekly player production (the granular backbone).",
        _seasoned(nfl.load_player_stats, summary_level="week"),
    ),
    NflverseTable(
        "player_stats_season",
        "Regular-season player production, pre-aggregated.",
        _seasoned(nfl.load_player_stats, summary_level="reg"),
    ),
    NflverseTable(
        "ff_opportunity",
        "Expected fantasy points given actual opportunity (TD backbone).",
        _seasoned(nfl.load_ff_opportunity, stat_type="weekly"),
    ),
    NflverseTable(
        "snap_counts",
        "Snap counts for snap-share features (PFR, since 2012).",
        _seasoned(nfl.load_snap_counts),
    ),
    NflverseTable(
        "rosters",
        "Season rosters: age, position, depth, team.",
        _seasoned(nfl.load_rosters),
    ),
    NflverseTable(
        "schedules",
        "Game schedules and results (context, games played).",
        _seasoned(nfl.load_schedules),
    ),
    NflverseTable(
        "players",
        "Player reference: identity, position, birthdate.",
        _reference(nfl.load_players),
    ),
    NflverseTable(
        "teams",
        "Team reference: abbreviations, names, divisions.",
        _reference(nfl.load_teams),
    ),
    NflverseTable(
        "draft_picks",
        "Draft capital, all history - draft-position priors for rookies.",
        _all_history(nfl.load_draft_picks),
    ),
)

TABLES_BY_NAME: dict[str, NflverseTable] = {t.name: t for t in TABLES}
