"""Projection entrypoint: feature panel -> projected distributions for a season.

    uv run python -m src.projections.run --season 2025 [--model baseline|bayesian]
                                         [--format redraft_ppr|superflex] [--top N]

Baseline is the default and is implemented; the Bayesian backend is a swap-in that
must beat the baseline on the backtest scoreboard before it becomes default
(CLAUDE.md constraint #4). Writes the scored projection to the cache and prints the
top N by projected points.
"""

from __future__ import annotations

import argparse

import polars as pl

from src.formats import FORMATS, get_format
from src.formats.score import score_components
from src.ingest.cache import read_table, write_table
from src.projections import MODELS
from src.projections.baseline.project import project_season
from src.projections.uncertainty import add_intervals, load_or_fit_interval_model


def run(
    *, season: int, model: str = "baseline", fmt_key: str = "redraft_ppr"
) -> pl.DataFrame:
    """Produce a scored projection with prediction intervals for ``season``.

    Projects format-agnostic component stats, scores them under ``fmt_key``, and
    attaches an empirical prediction interval (design.md §5) - the distribution,
    not just the point estimate.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}.")
    if model == "bayesian":
        raise NotImplementedError(
            "Bayesian backend not implemented yet - ship/beat the baseline first."
        )

    panel = read_table("feature_panel")
    players = read_table("players")
    projection = project_season(panel, players, season)
    scored = score_components(projection, get_format(fmt_key))
    scored = add_intervals(scored, load_or_fit_interval_model(panel, players)).sort(
        "projected_points", descending=True
    )
    write_table(f"projections_{model}_{fmt_key}_{season}", scored)
    return scored


def _print_top(scored: pl.DataFrame, fmt_key: str, n: int) -> None:
    top = scored.head(n).select(
        "player_name",
        "position",
        "team",
        pl.col("projected_points").round(1),
        pl.col("points_low").round(1),
        pl.col("points_high").round(1),
    )
    print(f"[projections] top {n} for format '{fmt_key}' (with 80% interval):")
    with pl.Config(tbl_rows=n, tbl_hide_dataframe_shape=True):
        print(top)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run player projections for a season.")
    parser.add_argument("--season", type=int, required=True, help="Season to project.")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="baseline",
        help="Projection backend (default: baseline).",
    )
    parser.add_argument(
        "--format",
        dest="fmt_key",
        choices=sorted(FORMATS),
        default="redraft_ppr",
        help="Scoring format (default: redraft_ppr).",
    )
    parser.add_argument(
        "--top", type=int, default=25, help="How many to print (default: 25)."
    )
    args = parser.parse_args()

    scored = run(season=args.season, model=args.model, fmt_key=args.fmt_key)
    print(
        f"[projections] wrote projections_{args.model}_{args.fmt_key}_{args.season} "
        f"({scored.height:,} players)."
    )
    _print_top(scored, args.fmt_key, args.top)


if __name__ == "__main__":
    main()
