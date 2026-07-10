"""Superflex / 2QB — full-PPR with an added QB-eligible flex (design.md §6.2).

Identical to redraft PPR except for the extra ``superflex`` slot. That single
config difference is what makes elite QBs first-round value here: it raises the
QB replacement baseline well above the flex baseline, and correct VOR math does
the rest. This is the project's most concrete source of edge.
"""

from __future__ import annotations

from src.formats.base import FormatConfig, RosterConfig, ScoringConfig

SUPERFLEX = FormatConfig(
    key="superflex",
    name="Superflex / 2QB (12-team)",
    scoring=ScoringConfig(rec=1.0),
    roster=RosterConfig(
        teams=12,
        qb=1,
        rb=2,
        wr=2,
        te=1,
        flex=1,
        superflex=1,  # QB/RB/WR/TE — the edge
        dst=1,
        k=1,
        bench=6,
    ),
)
