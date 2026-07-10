"""Player-name normalization for cross-source joins.

External ADP (Fantasy Football Calculator) has no shared player id with the
nflverse gsis ids, so we join on a normalized name + position. Normalization
lower-cases, strips punctuation and common generational suffixes, and collapses
whitespace so "Amon-Ra St. Brown", "Ja'Marr Chase", and "Michael Pittman Jr."
match across sources.
"""

from __future__ import annotations

import polars as pl

# Generational suffixes to drop (as whole words); longest-first so "iii" is
# consumed before "ii".
_SUFFIX_PATTERN = r"\b(iii|ii|iv|jr|sr)\b"


def norm_name_expr(column: str) -> pl.Expr:
    """A Polars expression normalizing ``column`` for name matching."""
    return (
        pl.col(column)
        .str.to_lowercase()
        .str.replace_all(r"[.'`]", "")
        .str.replace_all(r"[-]", " ")
        .str.replace_all(_SUFFIX_PATTERN, "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("norm_name")
    )
