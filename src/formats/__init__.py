"""Format registry.

The decision layer is parameterized by these configs (design.md §6). To add a
format, drop a module here holding a ``FormatConfig`` and register it below —
nothing else in the codebase branches on format.
"""

from __future__ import annotations

from src.formats.base import FormatConfig, RosterConfig, ScoringConfig
from src.formats.redraft_ppr import REDRAFT_PPR
from src.formats.superflex import SUPERFLEX

FORMATS: dict[str, FormatConfig] = {fmt.key: fmt for fmt in (REDRAFT_PPR, SUPERFLEX)}


def get_format(key: str) -> FormatConfig:
    """Look up a registered format by key, or raise with the valid options."""
    try:
        return FORMATS[key]
    except KeyError:
        raise KeyError(
            f"Unknown format {key!r}; registered: {sorted(FORMATS)}."
        ) from None


__all__ = [
    "FORMATS",
    "FormatConfig",
    "RosterConfig",
    "ScoringConfig",
    "get_format",
    "REDRAFT_PPR",
    "SUPERFLEX",
]
