"""Format config schema.

Formats are **configuration, not code branches** (CLAUDE.md, design.md §6). A
format is a scoring rule set plus a roster/replacement rule set. One projection
layer feeds all formats; only the decision layer reads these to set scoring
weights and positional replacement levels.

Adding a format = a new module holding a ``FormatConfig`` instance, registered in
``__init__``. Never an ``if superflex:`` scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Points per unit of production. Full-PPR defaults; every value is a knob.

    Note: ``pass_int`` is the classic −2. Sleeper's platform default is −1; since
    ADP is our benchmark, revisit this against the specific league when it starts
    to matter for QB valuation.
    """

    pass_yd: float = 0.04  # 1 pt / 25 passing yards
    pass_td: float = 4.0
    pass_int: float = -2.0
    rush_yd: float = 0.1  # 1 pt / 10 rushing yards
    rush_td: float = 6.0
    rec: float = 1.0  # PPR knob: 1.0 full, 0.5 half, 0.0 standard
    rec_yd: float = 0.1  # 1 pt / 10 receiving yards
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_point: float = 2.0


@dataclass(frozen=True)
class RosterConfig:
    """Starting-lineup construction — sets positional replacement levels.

    ``flex`` accepts RB/WR/TE; ``superflex`` additionally accepts QB. The superflex
    slot is the whole edge (design.md §6.2): it lifts replacement-level QB far
    above replacement-level flex, so elite QBs rise to where they belong — with no
    special modeling, just correct VOR against the right baseline.
    """

    teams: int = 12
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1  # RB/WR/TE
    superflex: int = 0  # QB/RB/WR/TE
    dst: int = 1
    k: int = 1
    bench: int = 6


@dataclass(frozen=True)
class FormatConfig:
    """A named format: a key, a display name, and its scoring + roster rules."""

    key: str
    name: str
    scoring: ScoringConfig
    roster: RosterConfig
