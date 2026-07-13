"""Format registry.

The decision layer is parameterized by these configs (design.md §6). Presets
cover the common league types; ``customize`` (and the API's override params)
handles everything else - team count, any scoring knob, any roster shape.
"""

from __future__ import annotations

from src.formats.base import (
    ROSTER_FIELDS,
    SCORING_FIELDS,
    FormatConfig,
    RosterConfig,
    ScoringConfig,
    customize,
)

REDRAFT_PPR = FormatConfig(
    key="redraft_ppr",
    name="Redraft PPR (12-team)",
    scoring=ScoringConfig(rec=1.0),
    roster=RosterConfig(),
)

REDRAFT_HALF = FormatConfig(
    key="redraft_half",
    name="Redraft Half-PPR (12-team)",
    scoring=ScoringConfig(rec=0.5),
    roster=RosterConfig(),
)

REDRAFT_STANDARD = FormatConfig(
    key="redraft_standard",
    name="Redraft Standard (12-team)",
    scoring=ScoringConfig(rec=0.0),
    roster=RosterConfig(),
)

SUPERFLEX = FormatConfig(
    key="superflex",
    name="Superflex / 2QB (12-team)",
    scoring=ScoringConfig(rec=1.0),
    roster=RosterConfig(superflex=1),
)

TWO_QB = FormatConfig(
    key="two_qb",
    name="2QB (12-team)",
    scoring=ScoringConfig(rec=1.0),
    roster=RosterConfig(qb=2),
)

# A real league as a config file - the pattern CLAUDE.md promises ("adding a
# format = a config file, not code branches"). Settings read from the ESPN API
# on 2026-07-12 and stable across all 11 league years; every other scoring rule
# matches the full-PPR defaults (verified against the league's scoringItems).
PIGSKIN17 = FormatConfig(
    key="pigskin17",
    name="pigskin17 (10-team 2QB PPR, 6pt pass TD)",
    scoring=ScoringConfig(rec=1.0, pass_td=6.0),
    roster=RosterConfig(teams=10, qb=2, bench=7),
)

FORMATS: dict[str, FormatConfig] = {
    fmt.key: fmt
    for fmt in (
        REDRAFT_PPR,
        REDRAFT_HALF,
        REDRAFT_STANDARD,
        SUPERFLEX,
        TWO_QB,
        PIGSKIN17,
    )
}


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
    "ROSTER_FIELDS",
    "RosterConfig",
    "SCORING_FIELDS",
    "ScoringConfig",
    "customize",
    "get_format",
    "PIGSKIN17",
    "REDRAFT_PPR",
    "REDRAFT_HALF",
    "REDRAFT_STANDARD",
    "SUPERFLEX",
    "TWO_QB",
]
