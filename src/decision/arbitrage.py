"""ADP arbitrage board (design.md §7) - the headline pre-draft view.

Ranks players by the disagreement between our value (VOR) and the market (ADP):
where do we and the field most disagree? A player we rank far above their ADP is a
**target** (the market is sleeping); one the market drafts well ahead of our value
is a **fade**. These are the mispriced contracts.

Both ranks are computed over the same matched pool (players who both project and
have an ADP), so the delta is apples-to-apples.
"""

from __future__ import annotations

import argparse

import polars as pl

from src.decision.board import build_value_board
from src.formats import FORMATS, get_format
from src.ingest.adp import DEFAULT_TEAMS, load_adp
from src.names import norm_name_expr
from src.projections.run import run as run_projection

_ARB_COLS: tuple[str, ...] = (
    "player_name",
    "position_group",
    "team",
    "position_tier",
    "projected_points",
    "points_low",
    "points_high",
    "vor",
    "our_value_rank",
    "adp",
    "adp_market_rank",
    "arbitrage_delta",
    "times_drafted",
)


def build_arbitrage(board: pl.DataFrame, adp: pl.DataFrame) -> pl.DataFrame:
    """Join the value board to ADP and rank by value-vs-market disagreement.

    ``arbitrage_delta`` = market rank - our value rank (both over the matched pool).
    Positive = we value the player above where the market drafts them (a target);
    negative = the market drafts them ahead of our value (a fade).
    """
    with_norm = board.with_columns(norm_name_expr("player_name"))
    market = adp.select("norm_name", "position", "adp", "adp_stdev", "times_drafted")
    joined = with_norm.join(
        market,
        left_on=["norm_name", "position_group"],
        right_on=["norm_name", "position"],
        how="inner",
    )
    ranked = joined.with_columns(
        pl.col("vor")
        .rank("ordinal", descending=True)
        .cast(pl.Int64)
        .alias("our_value_rank"),
        pl.col("adp").rank("ordinal").cast(pl.Int64).alias("adp_market_rank"),
    ).with_columns(
        # Signed: positive when the market drafts a player later than we value them.
        (pl.col("adp_market_rank") - pl.col("our_value_rank")).alias("arbitrage_delta")
    )
    return ranked.select(_ARB_COLS).sort("arbitrage_delta", descending=True)


def _print_side(df: pl.DataFrame, title: str) -> None:
    view = df.select(
        "player_name",
        "position_group",
        "position_tier",
        pl.col("projected_points").round(0),
        pl.col("vor").round(1),
        "our_value_rank",
        pl.col("adp").round(1),
        "adp_market_rank",
        "arbitrage_delta",
    )
    print(title)
    with pl.Config(tbl_rows=view.height, tbl_hide_dataframe_shape=True):
        print(view)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ADP arbitrage board.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--format", dest="fmt_key", choices=sorted(FORMATS), default="redraft_ppr"
    )
    parser.add_argument("--teams", type=int, default=DEFAULT_TEAMS)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    scored = run_projection(season=args.season, model="baseline", fmt_key=args.fmt_key)
    board = build_value_board(scored, get_format(args.fmt_key))
    adp = load_adp(args.season, args.fmt_key, args.teams)
    arb = build_arbitrage(board, adp)

    print(
        f"[arbitrage] {args.fmt_key} {args.season}: matched {arb.height} of "
        f"{board.height} projected players to ADP."
    )
    _print_side(arb.head(args.top), "\n=== TARGETS (we value >> market) ===")
    _print_side(
        arb.sort("arbitrage_delta").head(args.top), "\n=== FADES (market >> us) ==="
    )


if __name__ == "__main__":
    main()
