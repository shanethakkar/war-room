"""Redraft PPR — standard full-PPR, single-QB roster (design.md §6.1)."""

from __future__ import annotations

from src.formats.base import FormatConfig, RosterConfig, ScoringConfig

REDRAFT_PPR = FormatConfig(
    key="redraft_ppr",
    name="Redraft PPR (12-team)",
    scoring=ScoringConfig(rec=1.0),
    roster=RosterConfig(
        teams=12,
        qb=1,
        rb=2,
        wr=2,
        te=1,
        flex=1,
        superflex=0,
        dst=1,
        k=1,
        bench=6,
    ),
)
