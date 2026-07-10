"""FastAPI application.

    uv run uvicorn src.api.main:app --reload

Serves the value board (VOR + tiers + 80% interval) and the ADP-arbitrage delta
for a season/format - the data the Next.js board renders. Honest by design: this
is a decision aid (calibrated uncertainty + where we disagree with the market),
not a claim to beat ADP on average.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.service import DEFAULT_MODEL, get_board
from src.formats import FORMATS

# Seasons offered by the board (2026 = the upcoming draft; earlier = backtestable).
SEASONS: tuple[int, ...] = (2026, 2025, 2024, 2023)

app = FastAPI(
    title="War Room API",
    version=__version__,
    description="Fantasy football projections & draft decisions.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@app.get("/formats")
def list_formats() -> dict[str, dict[str, str]]:
    """The registered scoring/roster formats (design.md §6)."""
    return {key: {"key": fmt.key, "name": fmt.name} for key, fmt in FORMATS.items()}


@app.get("/seasons")
def list_seasons() -> dict[str, list[int]]:
    """Seasons the board can be built for."""
    return {"seasons": list(SEASONS)}


@app.get("/board")
def board(
    season: int = Query(default=SEASONS[0]),
    format: str = Query(default="redraft_ppr"),
) -> dict[str, Any]:
    """Value board + arbitrage for a season/format (the frontend's main payload)."""
    if season not in SEASONS:
        raise HTTPException(
            404, f"Unknown season {season}; choose from {list(SEASONS)}."
        )
    if format not in FORMATS:
        raise HTTPException(
            404, f"Unknown format {format!r}; choose from {sorted(FORMATS)}."
        )
    return {
        "season": season,
        "format": format,
        "format_name": FORMATS[format].name,
        "model": DEFAULT_MODEL,
        "players": get_board(season, format),
    }


@app.get("/")
def root() -> dict[str, Any]:
    """Service banner."""
    return {
        "service": "war-room",
        "version": __version__,
        "phase": "Phase 1 - pre-draft research",
        "formats": list(FORMATS),
        "seasons": list(SEASONS),
    }
