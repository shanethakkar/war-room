"""Backtest & benchmark - the scoreboard (design.md §8).

    uv run python -m src.validation.backtest [--start 2021] [--through 2024]
                                             [--format redraft_ppr|superflex]

For each season Y (leakage-free: train strictly before Y):
  - **Accuracy** - Spearman rank correlation and MAE of projection vs actual finish.
  - **Calibration** - do the 80% intervals contain the outcome ~80% of the time?
    (The interval model is refit on seasons < Y, so calibration is forward, not
    in-sample.)
  - **Beat ADP** - over the drafted pool, does our ranking predict actual finish
    better than the market's ADP does? This is the honest benchmark.

Every projection change should be re-run through this.
"""

from __future__ import annotations

import argparse

import polars as pl

from src.formats import FORMATS
from src.ingest.adp import load_adp
from src.ingest.cache import read_table
from src.names import norm_name_expr
from src.projections import MODELS
from src.projections.pipeline import scored_projection
from src.projections.uncertainty import default_seasons

_REPORT_COLS: tuple[str, ...] = (
    "season",
    "n",
    "spearman",
    "mae",
    "coverage",
    "adp_pool_n",
    "our_spearman",
    "adp_spearman",
    "beat_adp",
)


def _spearman(df: pl.DataFrame, a: str | pl.Expr, b: str | pl.Expr) -> float | None:
    """Spearman rank correlation, or None if too few rows."""
    if df.height < 3:
        return None
    value = df.select(pl.corr(a, b, method="spearman")).item()
    return None if value is None else float(value)


def season_backtest(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    year: int,
    model: str,
    fmt_key: str,
    residual_seasons: list[int],
    bayes_kwargs: dict[str, int] | None = None,
) -> dict[str, object]:
    """Compute accuracy, calibration, and the ADP benchmark for one season."""
    projection = scored_projection(
        panel,
        players,
        year,
        model=model,
        fmt_key=fmt_key,
        interval_residual_seasons=residual_seasons or None,
        bayes_kwargs=bayes_kwargs,
    )
    coverage: float | None = None

    actual = panel.filter((pl.col("season") == year) & (pl.col("games") >= 1)).select(
        "player_id", pl.col("fantasy_points_ppr").alias("actual")
    )
    matched = projection.join(actual, on="player_id", how="inner")

    spearman = _spearman(matched, "projected_points", "actual")
    mae = float(
        matched.select(
            (pl.col("projected_points") - pl.col("actual")).abs().mean()
        ).item()
    )
    if matched.height:
        coverage = float(
            matched.select(
                (
                    (pl.col("actual") >= pl.col("points_low"))
                    & (pl.col("actual") <= pl.col("points_high"))
                ).mean()
            ).item()
        )

    # ADP benchmark over the drafted pool.
    our_sp = adp_sp = None
    adp_pool_n = 0
    try:
        adp = load_adp(year, fmt_key)
        pool = matched.with_columns(norm_name_expr("player_name")).join(
            adp.select("norm_name", "position", "adp"),
            left_on=["norm_name", "position_group"],
            right_on=["norm_name", "position"],
            how="inner",
        )
        adp_pool_n = pool.height
        our_sp = _spearman(pool, "projected_points", "actual")
        # Lower ADP = earlier pick; negate so "higher = predicts a better finish".
        adp_sp = _spearman(pool, -pl.col("adp"), "actual")
    except (RuntimeError, OSError) as exc:  # network / FFC hiccup: skip benchmark
        print(f"[backtest] {year}: ADP benchmark unavailable ({exc})")

    beat = our_sp - adp_sp if our_sp is not None and adp_sp is not None else None
    return {
        "season": year,
        "n": matched.height,
        "spearman": spearman,
        "mae": mae,
        "coverage": coverage,
        "adp_pool_n": adp_pool_n,
        "our_spearman": our_sp,
        "adp_spearman": adp_sp,
        "beat_adp": beat,
    }


def backtest(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    seasons: list[int],
    model: str = "baseline",
    fmt_key: str = "redraft_ppr",
    bayes_kwargs: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Run the season-by-season backtest and return the per-season report."""
    projectable = set(default_seasons(panel))
    rows = [
        season_backtest(
            panel,
            players,
            year,
            model,
            fmt_key,
            sorted(s for s in projectable if s < year),
            bayes_kwargs,
        )
        for year in seasons
    ]
    return pl.DataFrame(rows).select(_REPORT_COLS)


def _mean(report: pl.DataFrame, column: str) -> float | None:
    value = report.select(pl.col(column).mean()).item()
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest projections vs actual + ADP."
    )
    parser.add_argument("--start", type=int, default=2021)
    parser.add_argument("--through", type=int, default=2024)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument(
        "--format", dest="fmt_key", choices=sorted(FORMATS), default="redraft_ppr"
    )
    parser.add_argument(
        "--draws", type=int, default=500, help="Bayesian sampler draws (per chain)."
    )
    args = parser.parse_args()

    panel = read_table("feature_panel")
    players = read_table("players")
    seasons = list(range(args.start, args.through + 1))
    bayes_kwargs = {"draws": args.draws, "tune": args.draws}
    print(
        f"[backtest] {args.model} {args.fmt_key}: seasons {seasons[0]}-{seasons[-1]} "
        f"(train strictly before each)"
    )
    report = backtest(panel, players, seasons, args.model, args.fmt_key, bayes_kwargs)
    with pl.Config(
        tbl_rows=report.height, tbl_hide_dataframe_shape=True, tbl_width_chars=180
    ):
        print(
            report.select(
                "season",
                "n",
                pl.col("spearman").round(3),
                pl.col("mae").round(1),
                pl.col("coverage").round(3),
                "adp_pool_n",
                pl.col("our_spearman").round(3),
                pl.col("adp_spearman").round(3),
                pl.col("beat_adp").round(3),
            )
        )

    beat = _mean(report, "beat_adp")
    verdict = (
        "BEATS ADP"
        if beat is not None and beat > 0
        else "trails ADP"
        if beat is not None
        else "no ADP comparison"
    )
    print(
        f"[backtest] means: spearman={_fmt(_mean(report, 'spearman'))} "
        f"mae={_fmt(_mean(report, 'mae'))} coverage={_fmt(_mean(report, 'coverage'))}"
    )
    print(
        f"[backtest] vs ADP: ours={_fmt(_mean(report, 'our_spearman'))} "
        f"adp={_fmt(_mean(report, 'adp_spearman'))} "
        f"beat_margin={_fmt(beat)} -> {verdict}"
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
