"""Format config schema.

Formats are **configuration, not code branches** (CLAUDE.md, design.md §6). A
format is a scoring rule set plus a roster/replacement rule set. One projection
layer feeds all formats; only the decision layer reads these to set scoring
weights and positional replacement levels.

Every scoring number is a knob (pass TD 4 vs 6, PPR 0/0.5/1, TE premium, kicker
distance values, DST points-allowed brackets), and rosters are fully
parameterized (team count, slots, 2QB vs superflex), so a league is a set of
overrides - never an ``if superflex:`` scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ScoringConfig:
    """Points per unit of production. Full-PPR defaults; every value is a knob.

    Offense notes: ``pass_int`` is the classic -2 (Sleeper's platform default is
    -1). ``te_rec_bonus`` implements TE-premium (extra points per TE reception).
    Kicker values default to the common 3/4/5 distance scale; DST to Yahoo-style
    points-allowed brackets.
    """

    # --- offense ---
    pass_yd: float = 0.04  # 1 pt / 25 passing yards
    pass_td: float = 4.0
    pass_int: float = -2.0
    rush_yd: float = 0.1  # 1 pt / 10 rushing yards
    rush_td: float = 6.0
    rec: float = 1.0  # PPR knob: 1.0 full, 0.5 half, 0.0 standard
    rec_yd: float = 0.1  # 1 pt / 10 receiving yards
    rec_td: float = 6.0
    te_rec_bonus: float = 0.0  # TE premium: extra per TE reception
    fumble_lost: float = -2.0
    two_point: float = 2.0
    # --- kicker ---
    fg_0_39: float = 3.0
    fg_40_49: float = 4.0
    fg_50_plus: float = 5.0
    fg_miss: float = -1.0
    pat_made: float = 1.0
    pat_miss: float = -1.0
    # --- team defense / special teams ---
    dst_sack: float = 1.0
    dst_int: float = 2.0
    dst_fumble_rec: float = 2.0
    dst_td: float = 6.0
    dst_safety: float = 2.0
    # Points-allowed brackets (per game): 0, 1-6, 7-13, 14-20, 21-27, 28-34, 35+.
    dst_pa_0: float = 10.0
    dst_pa_1_6: float = 7.0
    dst_pa_7_13: float = 4.0
    dst_pa_14_20: float = 1.0
    dst_pa_21_27: float = 0.0
    dst_pa_28_34: float = -1.0
    dst_pa_35_plus: float = -4.0


@dataclass(frozen=True)
class RosterConfig:
    """Starting-lineup construction - sets positional replacement levels.

    ``flex`` accepts RB/WR/TE; ``superflex`` additionally accepts QB. A 2QB
    league is simply ``qb=2`` (dedicated slots) - both raise QB replacement value
    the same way, via correct VOR math against the right baseline.
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


# Override names accepted by ``customize`` (and therefore the API): every
# scoring field plus every roster field.
SCORING_FIELDS: tuple[str, ...] = tuple(ScoringConfig.__dataclass_fields__)
ROSTER_FIELDS: tuple[str, ...] = tuple(RosterConfig.__dataclass_fields__)


def customize(base: FormatConfig, **overrides: Any) -> FormatConfig:
    """A new format from ``base`` with any scoring/roster fields overridden.

    Unknown keys raise so a typo in a league setting fails loudly.
    """
    scoring_kv = {k: v for k, v in overrides.items() if k in SCORING_FIELDS}
    roster_kv = {k: v for k, v in overrides.items() if k in ROSTER_FIELDS}
    unknown = set(overrides) - set(scoring_kv) - set(roster_kv)
    if unknown:
        raise KeyError(
            f"Unknown format override(s): {sorted(unknown)}. "
            f"Valid: {sorted(SCORING_FIELDS + ROSTER_FIELDS)}."
        )
    return FormatConfig(
        key=f"{base.key}_custom",
        name=f"{base.name} (custom)",
        scoring=replace(base.scoring, **scoring_kv),
        roster=replace(base.roster, **roster_kv),
    )
