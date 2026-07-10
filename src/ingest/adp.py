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

from src.ingest.cache import read_table, write_table
from src.names import norm_name_expr

_FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{ffc_format}"
_HEADERS = {"User-Agent": "war-room (github.com/shanethakkar/war-room)"}
DEFAULT_TEAMS = 12

# Our format keys -> Fantasy Football Calculator format slugs.
FORMAT_TO_FFC: dict[str, str] = {"redraft_ppr": "ppr", "superflex": "2qb"}


def _ffc_format(fmt_key: str) -> str:
    try:
        return FORMAT_TO_FFC[fmt_key]
    except KeyError:
        known = sorted(FORMAT_TO_FFC)
        raise KeyError(
            f"No FFC ADP mapping for format {fmt_key!r}; known: {known}."
        ) from None


def _normalize(
    players: list[dict[str, object]], year: int, fmt_key: str, teams: int
) -> pl.DataFrame:
    """Shape the FFC player list into a clean, join-ready ADP frame."""
    df = pl.DataFrame(players)
    return (
        df.select(
            pl.col("name").alias("adp_name"),
            pl.col("position"),
            pl.col("team").alias("adp_team"),
            pl.col("adp").cast(pl.Float64),
            pl.col("stdev").cast(pl.Float64).alias("adp_stdev"),
            pl.col("times_drafted").cast(pl.Int64),
        )
        .with_columns(
            norm_name_expr("adp_name"),
            pl.lit(year).cast(pl.Int32).alias("adp_year"),
            pl.lit(fmt_key).alias("format"),
            pl.lit(teams).cast(pl.Int32).alias("teams"),
        )
        # Lower ADP = drafted earlier = market rank 1.
        .with_columns(pl.col("adp").rank("ordinal").alias("adp_rank"))
        .sort("adp")
    )


def fetch_adp(year: int, fmt_key: str, teams: int = DEFAULT_TEAMS) -> pl.DataFrame:
    """Fetch and normalize ADP from FFC for a season/format (network)."""
    resp = httpx.get(
        _FFC_URL.format(ffc_format=_ffc_format(fmt_key)),
        params={"teams": teams, "year": year},
        headers=_HEADERS,
        timeout=30.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "Success":
        status = payload.get("status")
        raise RuntimeError(
            f"FFC ADP request failed for {fmt_key} {year}: status={status!r}"
        )
    return _normalize(payload["players"], year, fmt_key, teams)


def load_adp(
    year: int, fmt_key: str, teams: int = DEFAULT_TEAMS, *, refresh: bool = False
) -> pl.DataFrame:
    """Return cached ADP, fetching from FFC (and caching) on first use."""
    name = f"adp_{_ffc_format(fmt_key)}_{teams}_{year}"
    if not refresh:
        try:
            return read_table(name)
        except FileNotFoundError:
            pass
    df = fetch_adp(year, fmt_key, teams)
    write_table(name, df)
    return df


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
