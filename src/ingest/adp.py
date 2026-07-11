"""Average Draft Position (ADP) ingestion - the market benchmark.

ADP is the real field consensus and the honest benchmark the projections must
beat (design.md §8). Source: Fantasy Football Calculator's free, no-key API, which
provides both current and historical ADP for PPR and superflex (2QB). ADP is a
**benchmark only - it never feeds projections** (the open/offline projection
guardrail is intact).

    https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams=N&year=Y

Cached to Parquet like every other data source; offline-reproducible thereafter.
"""

from __future__ import annotations

import argparse

import httpx
import polars as pl

from src.formats.base import FormatConfig
from src.ingest.cache import read_table, write_table
from src.names import norm_name_expr

_FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{ffc_format}"
_HEADERS = {"User-Agent": "war-room (github.com/shanethakkar/war-room)"}
DEFAULT_TEAMS = 12

# Preset format keys -> Fantasy Football Calculator market slugs.
FORMAT_TO_FFC: dict[str, str] = {
    "redraft_ppr": "ppr",
    "redraft_half": "half-ppr",
    "redraft_standard": "standard",
    "superflex": "2qb",
    "two_qb": "2qb",
}

# FFC position labels -> our position groups.
_FFC_POSITIONS: dict[str, str] = {"PK": "K", "DEF": "DST"}


def _dst_name_map() -> dict[str, str]:
    """FFC defense-name variants -> our DST norm_name (the team's full name).

    FFC names defenses by city ("Denver Defense", "LA Rams Defense"), while our
    DST rows use full team names. Variants per team: the full name, the city
    alone (dropped when two teams share it, e.g. New York), and LA/NY short
    forms. Empty when the teams reference isn't cached.
    """
    try:
        teams = read_table("teams")
    except FileNotFoundError:
        return {}
    mapping: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in teams.select("team_name", "team_nick").unique().iter_rows(named=True):
        full = str(row["team_name"]).lower()
        nick = str(row["team_nick"]).lower()
        city = full.removesuffix(nick).strip()
        variants = {full, city}
        for prefix, short in (("los angeles", "la"), ("new york", "ny")):
            if city == prefix:
                variants.add(f"{short} {nick}")
        for variant in variants:
            if not variant:
                continue
            if variant in mapping and mapping[variant] != full:
                ambiguous.add(variant)  # e.g. bare "new york"
            mapping[variant] = full
    for variant in ambiguous:
        mapping.pop(variant, None)
    return mapping


def _canonical_dst(df: pl.DataFrame) -> pl.DataFrame:
    """Rewrite DEF/DST norm_names to our team-full-name convention."""
    name_map = _dst_name_map()
    if not name_map:
        return df
    stripped = pl.col("norm_name").str.replace(r"\s*defense$", "").str.strip_chars()
    return df.with_columns(
        pl.when(pl.col("position") == "DST")
        .then(stripped.replace(name_map))
        .otherwise(pl.col("norm_name"))
        .alias("norm_name")
    )


def ffc_slug(fmt: FormatConfig | str) -> str:
    """The nearest FFC ADP market for a format (preset key or custom config).

    Custom configs resolve by their rules: 2+ startable QBs -> the 2QB market;
    otherwise by the reception value (full / half / standard PPR).
    """
    if isinstance(fmt, str):
        try:
            return FORMAT_TO_FFC[fmt]
        except KeyError:
            known = sorted(FORMAT_TO_FFC)
            raise KeyError(
                f"No FFC ADP mapping for format {fmt!r}; known: {known}."
            ) from None
    if fmt.roster.qb + fmt.roster.superflex >= 2:
        return "2qb"
    if fmt.scoring.rec >= 0.75:
        return "ppr"
    if fmt.scoring.rec >= 0.25:
        return "half-ppr"
    return "standard"


def _normalize(
    players: list[dict[str, object]], year: int, slug: str, teams: int
) -> pl.DataFrame:
    """Shape the FFC player list into a clean, join-ready ADP frame.

    FFC's ``PK``/``DEF`` positions map to our ``K``/``DST`` groups; DEF names are
    full team names, which is exactly how our DST rows are named.
    """
    df = pl.DataFrame(players)
    return (
        df.select(
            pl.col("name").alias("adp_name"),
            pl.col("position").replace(_FFC_POSITIONS),
            pl.col("team").alias("adp_team"),
            pl.col("adp").cast(pl.Float64),
            pl.col("stdev").cast(pl.Float64).alias("adp_stdev"),
            pl.col("times_drafted").cast(pl.Int64),
        )
        .with_columns(
            norm_name_expr("adp_name"),
            pl.lit(year).cast(pl.Int32).alias("adp_year"),
            pl.lit(slug).alias("format"),
            pl.lit(teams).cast(pl.Int32).alias("teams"),
        )
        # Lower ADP = drafted earlier = market rank 1.
        .with_columns(pl.col("adp").rank("ordinal").alias("adp_rank"))
        .sort("adp")
    )


def fetch_adp(
    year: int, fmt: FormatConfig | str, teams: int = DEFAULT_TEAMS
) -> pl.DataFrame:
    """Fetch and normalize ADP from FFC for a season/format (network)."""
    slug = ffc_slug(fmt)
    resp = httpx.get(
        _FFC_URL.format(ffc_format=slug),
        params={"teams": teams, "year": year},
        headers=_HEADERS,
        timeout=30.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "Success":
        status = payload.get("status")
        raise RuntimeError(
            f"FFC ADP request failed for {slug} {year}: status={status!r}"
        )
    return _normalize(payload["players"], year, slug, teams)


def load_adp(
    year: int,
    fmt: FormatConfig | str = "redraft_ppr",
    teams: int = DEFAULT_TEAMS,
    *,
    refresh: bool = False,
) -> pl.DataFrame:
    """Return cached ADP, fetching from FFC (and caching) on first use.

    ``fmt`` is a preset key or a full FormatConfig (custom leagues resolve to
    the nearest FFC market via ``ffc_slug``). v2 cache name: includes the
    PK/DEF position mapping.
    """
    name = f"adp2_{ffc_slug(fmt)}_{teams}_{year}"
    if not refresh:
        try:
            return _canonical_dst(read_table(name))
        except FileNotFoundError:
            pass
    df = fetch_adp(year, fmt, teams)
    write_table(name, df)
    return _canonical_dst(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch/cache ADP from FFC.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--format", dest="fmt_key", choices=sorted(FORMAT_TO_FFC), default="redraft_ppr"
    )
    parser.add_argument("--teams", type=int, default=DEFAULT_TEAMS)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    adp = load_adp(args.season, args.fmt_key, args.teams, refresh=args.refresh)
    print(
        f"[adp] {args.fmt_key} {args.season} ({args.teams}-team): {adp.height} players"
    )
    with pl.Config(tbl_rows=args.top, tbl_hide_dataframe_shape=True):
        print(
            adp.head(args.top).select(
                "adp_rank", "adp_name", "position", "adp_team", "adp", "times_drafted"
            )
        )


if __name__ == "__main__":
    main()
