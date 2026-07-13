"""Stage 0 of the room-bias research: ground the parameters in observed data.

    uv run python -m src.validation.room_calibration

Before simulating "biased rooms" (QB-early leagues, herding runs), we bound how
much real rooms CAN deviate from market timing, using FFC's observed per-player
pick dispersion (``adp_stdev``, measured across thousands of real mock drafts).

Logic: if rooms systematically shifted a position by delta picks, every player
at that position would carry at least sd(delta) in his observed pick stdev.
Observed stdevs therefore UPPER-BOUND room-level positional bias - loosely, since
they also contain within-room randomness and season-long ADP drift. Stage 1's
archetype sweep must stay inside these bounds, and its herding parameter must
not push simulated per-player pick dispersion beyond the observed targets.
"""

from __future__ import annotations

import argparse

import polars as pl

from src.ingest.adp import load_adp

# Draft phases (12-team rounds).
PHASES: tuple[tuple[str, int, int], ...] = (
    ("R1-3", 1, 36),
    ("R4-8", 37, 96),
    ("R9-15", 97, 180),
)


def dispersion_profile(years: list[int], fmt: str = "redraft_ppr") -> pl.DataFrame:
    """Median observed pick stdev per (phase, position), pooled over ``years``.

    The upper bound on room-level positional timing bias, by draft phase.
    """
    frames = []
    for year in years:
        adp = load_adp(year, fmt)
        frames.append(
            adp.select("position", "adp", "adp_stdev").with_columns(
                pl.lit(year).alias("year")
            )
        )
    pooled = pl.concat(frames).drop_nulls(["adp", "adp_stdev"])
    labels = (
        pl.when(pl.col("adp") <= PHASES[0][2])
        .then(pl.lit(PHASES[0][0]))
        .when(pl.col("adp") <= PHASES[1][2])
        .then(pl.lit(PHASES[1][0]))
        .otherwise(pl.lit(PHASES[2][0]))
    )
    return (
        pooled.with_columns(labels.alias("phase"))
        .group_by("phase", "position")
        .agg(
            pl.col("adp_stdev").median().alias("median_stdev"),
            pl.col("adp_stdev").quantile(0.9).alias("p90_stdev"),
            pl.len().alias("n"),
        )
        .sort("phase", "position")
    )


def phase_targets(profile: pl.DataFrame) -> pl.DataFrame:
    """Overall per-phase dispersion targets (herding calibration ceilings)."""
    return (
        profile.group_by("phase")
        .agg(
            (pl.col("median_stdev") * pl.col("n")).sum().alias("_w"),
            pl.col("n").sum().alias("n"),
        )
        .with_columns((pl.col("_w") / pl.col("n")).round(1).alias("median_stdev"))
        .drop("_w")
        .sort("phase")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: room-bias bounds from FFC.")
    parser.add_argument("--start", type=int, default=2019)
    parser.add_argument("--through", type=int, default=2024)
    parser.add_argument("--format", dest="fmt", default="redraft_ppr")
    args = parser.parse_args()

    years = list(range(args.start, args.through + 1))
    profile = dispersion_profile(years, args.fmt)
    print(f"[stage0] observed pick dispersion, {args.fmt} {years[0]}-{years[-1]}")
    print("[stage0] median stdev by phase x position (upper bound on room bias):")
    with pl.Config(tbl_rows=30, tbl_hide_dataframe_shape=True):
        print(
            profile.with_columns(
                (pl.col("median_stdev") / 12).round(2).alias("~rounds"),
                pl.col("median_stdev").round(1),
                pl.col("p90_stdev").round(1),
            )
        )
    print("[stage0] per-phase herding calibration ceilings (player-level stdev):")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(phase_targets(profile))
    print(
        "[stage0] Stage-1 sweep must keep positional bias within ~the median "
        "stdev per phase and simulated player dispersion under the ceilings."
    )


if __name__ == "__main__":
    main()
