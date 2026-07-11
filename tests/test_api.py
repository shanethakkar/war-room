"""API tests - offline-safe (no cached data / network required).

The board payload itself needs the data cache, so it's exercised by driving the
running app; here we cover routing, metadata, and request validation.
"""

from __future__ import annotations

from src.api.main import SEASONS, app
from starlette.testclient import TestClient

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_seasons_lists_upcoming_first() -> None:
    seasons = client.get("/seasons").json()["seasons"]
    assert seasons == list(SEASONS)
    assert seasons[0] == 2026  # the upcoming draft


def test_formats_endpoint() -> None:
    formats = client.get("/formats").json()
    assert {
        "redraft_ppr",
        "redraft_half",
        "redraft_standard",
        "superflex",
        "two_qb",
    } == set(formats)


def test_board_rejects_bad_override() -> None:
    # teams=99 violates the query validation bounds.
    assert client.get("/board?season=2026&teams=99").status_code == 422


def test_board_rejects_unknown_season() -> None:
    assert client.get("/board?season=1999&format=redraft_ppr").status_code == 404


def test_board_rejects_unknown_format() -> None:
    assert client.get("/board?season=2026&format=nope").status_code == 404
