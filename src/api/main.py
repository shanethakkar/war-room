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
from src.formats import FORMATS, customize, get_format

# Seasons offered by the board (2026 = the upcoming draft; earlier = backtestable).
SEASONS: tuple[int, ...] = (2026, 2025, 2024, 2023)

app = FastAPI(
    title="War Room API",
    version=__version__,
    description="Fantasy football projections & draft decisions.",
)
app.add_middleware(
    CORSMiddleware,
    # Any localhost port - this is a local-only dev tool, and Next may land on
    # 3000/3001/etc depending on what else is running.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
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
    format: str = Query(default="redraft_ppr", description="Preset format key."),
    # Roster overrides (league setup).
    teams: int | None = Query(default=None, ge=4, le=20),
    qb: int | None = Query(default=None, ge=0, le=3),
    rb: int | None = Query(default=None, ge=0, le=5),
    wr: int | None = Query(default=None, ge=0, le=5),
    te: int | None = Query(default=None, ge=0, le=3),
    flex: int | None = Query(default=None, ge=0, le=4),
    superflex: int | None = Query(default=None, ge=0, le=2),
    dst: int | None = Query(default=None, ge=0, le=2),
    k: int | None = Query(default=None, ge=0, le=2),
    # Scoring overrides (the headline knobs; presets carry the rest).
    rec: float | None = Query(default=None, ge=0.0, le=2.0),
    pass_td: float | None = Query(default=None, ge=0.0, le=8.0),
    pass_int: float | None = Query(default=None, ge=-6.0, le=0.0),
    te_rec_bonus: float | None = Query(default=None, ge=0.0, le=1.5),
) -> dict[str, Any]:
    """The blended board for a season and league setup.

    ``format`` picks a preset; any override param customizes it (e.g.
    ``?format=redraft_ppr&teams=10&pass_td=6&qb=2``). Boards are cached per
    resolved configuration.
    """
    if season not in SEASONS:
        raise HTTPException(
            404, f"Unknown season {season}; choose from {list(SEASONS)}."
        )
    if format not in FORMATS:
        raise HTTPException(
            404, f"Unknown format {format!r}; choose from {sorted(FORMATS)}."
        )
    overrides = {
        name: value
        for name, value in {
            "teams": teams,
            "qb": qb,
            "rb": rb,
            "wr": wr,
            "te": te,
            "flex": flex,
            "superflex": superflex,
            "dst": dst,
            "k": k,
            "rec": rec,
            "pass_td": pass_td,
            "pass_int": pass_int,
            "te_rec_bonus": te_rec_bonus,
        }.items()
        if value is not None
    }
    fmt = get_format(format)
    if overrides:
        fmt = customize(fmt, **overrides)
    return {
        "season": season,
        "format": format,
        "format_name": fmt.name,
        "overrides": overrides,
        "model": DEFAULT_MODEL,
        "players": get_board(season, fmt),
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
