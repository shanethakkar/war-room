"""User-league draft history from ESPN's fantasy v3 API - room-bias data.

Scoped exception to constraint #3 (progress.md decision, 2026-07-12): the
user's OWN league's historical draft picks are ingested as draft-BEHAVIOR
data - they calibrate how the room deviates from market ADP (the warm-start
prior for room-aware draft guidance) and they NEVER feed projections, exactly
like ADP itself. No other ESPN data is used.

Private leagues authenticate with the ``espn_s2`` + ``SWID`` browser cookies,
read from the environment or the gitignored repo-root ``.env``:

    ESPN_LEAGUE_ID=291362
    ESPN_S2=...        # long URL-encoded blob; expires after weeks-months
    SWID={...}         # keep the curly braces

Draft picks cache to Parquet per (league, year) like every other source, so
analysis is offline after the first pull. Sign convention throughout:
``early_by = adp - overall`` - POSITIVE means the room drafts that position
EARLIER than the market; the room simulator's shift convention is the
negation (see ``to_room_shift``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import httpx
import polars as pl

from src.config import REPO_ROOT
from src.formats import get_format
from src.ingest.adp import load_adp
from src.ingest.cache import cache_path, read_table, write_table
from src.names import norm_name_expr

_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
_HEADERS = {"User-Agent": "war-room (github.com/shanethakkar/war-room)"}
# ESPN serves 2018+ from the season endpoint; older years via leagueHistory.
_HISTORY_BEFORE = 2018
# Positions whose picks are benchmarked against FFC ADP (DST is excluded: its
# ESPN ids are synthetic negatives and its timing signal is measured noise).
BIAS_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")

_ENV_KEYS = ("ESPN_LEAGUE_ID", "ESPN_S2", "SWID")


@dataclass(frozen=True)
class LeagueAuth:
    """Credentials for one private ESPN league (cookies are login-equivalent)."""

    league_id: int
    espn_s2: str
    swid: str


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines -> dict; comments/blanks skipped, quotes stripped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def league_auth_from_env() -> LeagueAuth:
    """Resolve credentials from the repo-root ``.env`` (never committed)."""
    env_file = REPO_ROOT / ".env"
    values = parse_env(env_file.read_text()) if env_file.exists() else {}
    missing = [k for k in _ENV_KEYS if not values.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing {missing} in {env_file}. Private ESPN leagues need the "
            "espn_s2 + SWID cookies from a logged-in browser session "
            "(DevTools -> Storage -> Cookies -> espn.com)."
        )
    return LeagueAuth(
        league_id=int(values["ESPN_LEAGUE_ID"]),
        espn_s2=values["ESPN_S2"],
        swid=values["SWID"],
    )


def fetch_league_year(year: int, auth: LeagueAuth) -> dict[str, Any]:
    """Raw league payload (draft picks + teams + settings) for one season."""
    if year >= _HISTORY_BEFORE:
        url = f"{_BASE}/seasons/{year}/segments/0/leagues/{auth.league_id}"
        params: list[tuple[str, str]] = []
    else:
        url = f"{_BASE}/leagueHistory/{auth.league_id}"
        params = [("seasonId", str(year))]
    params += [("view", "mDraftDetail"), ("view", "mTeam"), ("view", "mSettings")]
    resp = httpx.get(
        url,
        params=tuple(params),
        headers=_HEADERS,
        cookies={"espn_s2": auth.espn_s2, "SWID": auth.swid},
        timeout=30.0,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            f"ESPN returned 401 for league {auth.league_id} year {year} - the "
            "espn_s2 cookie has likely expired; re-copy it from the browser."
        )
    resp.raise_for_status()
    body = resp.json()
    payload = body[0] if isinstance(body, list) else body
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected ESPN payload shape for year {year}.")
    return payload


def draft_frame(payload: dict[str, Any], year: int) -> pl.DataFrame:
    """Shape one season's raw payload into a tidy pick frame (pure).

    One row per pick: draft position, drafting team (id + name), the ESPN
    player id (negative ids are team defenses), and the keeper flag - enough
    to reconstruct the room's behavior without any ESPN player metadata.
    """
    detail = payload.get("draftDetail")
    picks = detail.get("picks", []) if isinstance(detail, dict) else []
    teams = payload.get("teams")
    names: dict[int, str] = {}
    for team in teams if isinstance(teams, list) else []:
        label = team.get("name") or (
            f"{team.get('location', '')} {team.get('nickname', '')}".strip()
        )
        names[int(team["id"])] = str(label)
    return pl.DataFrame(
        [
            {
                "year": year,
                "overall": int(p["overallPickNumber"]),
                "round": int(p["roundId"]),
                "round_pick": int(p["roundPickNumber"]),
                "team_id": int(p["teamId"]),
                "team_name": names.get(int(p["teamId"]), ""),
                "espn_id": int(p["playerId"]),
                "keeper": bool(p.get("keeper", False)),
            }
            for p in picks
        ],
        schema={
            "year": pl.Int32,
            "overall": pl.Int32,
            "round": pl.Int32,
            "round_pick": pl.Int32,
            "team_id": pl.Int32,
            "team_name": pl.String,
            "espn_id": pl.Int64,
            "keeper": pl.Boolean,
        },
    )


def load_league_draft(
    year: int, auth: LeagueAuth | None = None, *, refresh: bool = False
) -> pl.DataFrame:
    """One season's draft picks, cached as ``espn_draft_<league>_<year>``."""
    resolved = auth if auth is not None else league_auth_from_env()
    name = f"espn_draft_{resolved.league_id}_{year}"
    if not refresh and cache_path(name).exists():
        return read_table(name)
    df = draft_frame(fetch_league_year(year, resolved), year)
    write_table(name, df)
    return df


def league_history(
    years: list[int], auth: LeagueAuth | None = None, *, refresh: bool = False
) -> pl.DataFrame:
    """All seasons' picks concatenated (fetching/caching per year)."""
    resolved = auth if auth is not None else league_auth_from_env()
    return pl.concat([load_league_draft(y, resolved, refresh=refresh) for y in years])


def attach_identity(picks: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Join nflverse identity (norm_name, position) onto picks via espn_id (pure).

    Negative ESPN ids are team defenses -> position "DST" with no name match
    (DST is excluded from market benchmarking anyway). Unmapped ids keep null
    identity so match rates stay measurable.
    """
    identity = (
        players.filter(pl.col("espn_id").is_not_null())
        .select(
            pl.col("espn_id").cast(pl.Int64),
            pl.col("display_name"),
            pl.col("position"),
        )
        .unique(subset="espn_id", keep="first")
        .with_columns(norm_name_expr("display_name"))
    )
    return picks.join(identity, on="espn_id", how="left").with_columns(
        pl.when(pl.col("espn_id") < 0)
        .then(pl.lit("DST"))
        .otherwise(pl.col("position"))
        .alias("position")
    )


def pick_deltas(picks: pl.DataFrame, adp: pl.DataFrame) -> pl.DataFrame:
    """Picks joined to a market board, with ``early_by = adp - overall`` (pure).

    Positive ``early_by`` = the room took the player EARLIER than the market.
    Rows without a market match carry null ``early_by`` (deep picks the market
    never drafts); bias aggregation skips them.
    """
    market = adp.select("norm_name", "position", "adp")
    return picks.join(market, on=["norm_name", "position"], how="left").with_columns(
        (pl.col("adp") - pl.col("overall")).alias("early_by")
    )


def positional_bias(deltas: pl.DataFrame) -> pl.DataFrame:
    """Per-position room bias pooled across seasons (pure).

    ``mean_early``/``median_early`` are in picks (positive = the room reaches
    for that position); ``years_early``/``years_total`` show sign stability -
    a decade of one sign is a real room signature, one year is noise.
    """
    matched = deltas.filter(
        pl.col("early_by").is_not_null() & pl.col("position").is_in(BIAS_POSITIONS)
    )
    yearly = matched.group_by("year", "position").agg(
        pl.col("early_by").mean().alias("year_mean")
    )
    stability = yearly.group_by("position").agg(
        (pl.col("year_mean") > 0).sum().cast(pl.Int64).alias("years_early"),
        pl.len().cast(pl.Int64).alias("years_total"),
    )
    return (
        matched.group_by("position")
        .agg(
            pl.col("early_by").mean().round(1).alias("mean_early"),
            pl.col("early_by").median().round(1).alias("median_early"),
            pl.len().cast(pl.Int64).alias("n"),
        )
        .join(stability, on="position")
        .sort("position")
    )


def to_room_shift(bias: pl.DataFrame) -> dict[str, float]:
    """Bias table -> a room-simulator shift dict (sign flip, pure).

    The simulator's convention (``room_sim.BIAS_SCENARIOS``) shifts effective
    ADP, so NEGATIVE = drafted earlier - the negation of ``mean_early``.
    """
    return {
        str(row["position"]): -float(row["mean_early"])
        for row in bias.iter_rows(named=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull/cache ESPN league draft history; report room bias."
    )
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--through", type=int, default=2025)
    parser.add_argument(
        "--format", dest="fmt_key", default="pigskin17", help="Market benchmark."
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    auth = league_auth_from_env()
    fmt = get_format(args.fmt_key)
    years = list(range(args.start, args.through + 1))
    picks = attach_identity(
        league_history(years, auth, refresh=args.refresh), read_table("players")
    )
    print(f"[espn] league {auth.league_id}: {picks.height} picks, {len(years)} drafts")

    frames: list[pl.DataFrame] = []
    for year in years:
        try:
            adp = load_adp(year, fmt, teams=fmt.roster.teams)
        except (RuntimeError, OSError, httpx.HTTPError):
            print(f"[espn] {year}: no FFC board for this market - skipped")
            continue
        frames.append(pick_deltas(picks.filter(pl.col("year") == year), adp))
    if not frames:
        print("[espn] no benchmarkable years - nothing to report")
        return
    deltas = pl.concat(frames)
    bias = positional_bias(deltas)
    print("[espn] room bias vs market (early_by: + = drafted earlier than ADP):")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(bias)
    print(f"[espn] room-sim shift dict: {to_room_shift(bias)}")


if __name__ == "__main__":
    main()
