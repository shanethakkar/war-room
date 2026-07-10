"""FastAPI application.

    uv run uvicorn src.api.main:app --reload

Thin by design (design.md §3.4). Projection and decision endpoints land as those
layers come online; for now it exposes health and the registered formats so the
service is real and verifiable from day one.
"""

from __future__ import annotations

from fastapi import FastAPI

from src import __version__
from src.formats import FORMATS

app = FastAPI(
    title="War Room API",
    version=__version__,
    description="Fantasy football projections & draft decisions.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@app.get("/formats")
def list_formats() -> dict[str, dict[str, str]]:
    """List the registered scoring/roster formats (design.md §6)."""
    return {key: {"key": fmt.key, "name": fmt.name} for key, fmt in FORMATS.items()}


@app.get("/")
def root() -> dict[str, object]:
    """Service banner."""
    return {
        "service": "war-room",
        "version": __version__,
        "phase": "Phase 1 — pre-draft research",
        "formats": list(FORMATS),
    }
