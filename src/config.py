"""Central configuration: paths and global constants.

Single source of truth for the data-cache location and the historical training
window, so every layer agrees. Keep this free of heavy imports — it is loaded
everywhere.
"""

from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file (src/config.py → src → repo root).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Local Parquet cache for nflverse pulls. Gitignored and fully reproducible from
# `python -m src.ingest.refresh`. Nothing projection-related should read from the
# network at runtime — it all comes from here.
DATA_DIR: Path = REPO_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"

# Historical training window.
#
# Decision (2026-07-10, see progress.md): train on 2016–present. Recent enough to
# reflect the modern pass-heavy, high-scoring era and to keep opportunity shares
# and aging curves on-regime, without dragging in older rule/scheme eras that
# would need explicit era adjustment.
DATA_START_SEASON: int = 2016


def seasons_through(through: int) -> list[int]:
    """Return the inclusive list of training seasons from the start window.

    Used by the backtest to build a leakage-free train/predict split: train on
    ``seasons_through(N)``, project ``N + 1``.
    """
    if through < DATA_START_SEASON:
        raise ValueError(
            f"`through` ({through}) precedes the data window start "
            f"({DATA_START_SEASON})."
        )
    return list(range(DATA_START_SEASON, through + 1))
