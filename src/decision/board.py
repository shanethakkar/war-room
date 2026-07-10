"""Value board: rank players by VOR for a format (design.md §7).

Ties the projection to draftable value. Given a scored projection and a format,
compute replacement-level VOR, positional tiers, and overall/position ranks - the
board a drafter actually reads. This is where the superflex QB edge shows up, for
free, via the correct replacement baseline.

    uv run python -m src.decision.board --season 2025 --format superflex [--top N]
"""

from __future__ import annotations

import argparse

import polars as pl

from src.decision.replacement import add_starter_flags, add_vor
from src.decision.tiers import add_overlap_tiers
from src.formats import FORMATS, FormatConfig, get_format
from src.ingest.cache import write_table
from src.projections import MODELS
from src.projections.run import run as run_projection

_BOARD_COLS: tuple[str, ...] = (
    "overall_rank",
    "position_rank",
    "position_tier",
    "player_name",
    "position",
    "position_group",
    "team",
    "is_rookie",
    "projected_games",
    "projected_points",
    "points_low",
    "points_high",
    "replacement_level",
    "vor",
)


def build_value_board(scored: pl.DataFrame, fmt: FormatConfig) -> pl.DataFrame:
    """Compute VOR, overlap tiers, and ranks; sorted by VOR (best first).

    ``scored`` must carry prediction intervals (``points_low`` / ``points_high``)
    from the uncertainty layer - tiers come from interval overlap (design.md §5).
    """
    df = add_overlap_tiers(add_vor(add_starter_flags(scored, fmt.roster)))
    df = df.with_columns(
        pl.col("vor")
        .rank("ordinal", descending=True)
        .over("position_group")
        .alias("position_rank")
    )
    return (
        df.sort("vor", descending=True)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("overall_rank")
        )
        .select(_BOARD_COLS)
    )


def _print_top(board: pl.DataFrame, fmt_key: str, n: int) -> None:
    top = board.head(n).select(
        "overall_rank",
        "position",
        "position_tier",
        "player_name",
        "team",
        pl.col("projected_points").round(1),
        pl.col("vor").round(1),
    )
    print(f"[board] top {n} by VOR for format '{fmt_key}':")
    with pl.Config(tbl_rows=n, tbl_hide_dataframe_shape=True):
        print(top)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the format-aware value board.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument(
        "--format", dest="fmt_key", choices=sorted(FORMATS), default="redraft_ppr"
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    scored = run_projection(season=args.season, model=args.model, fmt_key=args.fmt_key)
    board = build_value_board(scored, get_format(args.fmt_key))
    write_table(f"value_board_{args.fmt_key}_{args.season}", board)
    print(
        f"[board] wrote value_board_{args.fmt_key}_{args.season} "
        f"({board.height:,} players)."
    )
    _print_top(board, args.fmt_key, args.top)


if __name__ == "__main__":
    main()
