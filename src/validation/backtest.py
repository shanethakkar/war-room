"""Backtest entrypoint: train-through-N / project-N+1, scored against ADP.

    uv run python -m src.validation.backtest --through 2024

Reports rank correlation, MAE, and interval calibration vs. actual finish, and
benchmarks all three against Sleeper ADP. Not implemented yet (see progress.md).
"""

from __future__ import annotations

import argparse

from src.config import seasons_through


def backtest(*, through: int, model: str = "baseline") -> None:
    """Train on the window up to ``through`` and score projections for ``through + 1``.

    Enforces a leakage-free split (design.md §8): only ``seasons_through(through)``
    is visible to the model; the projected season is held out. Accuracy (rank
    corr, MAE) and calibration (interval coverage) are both reported, each against
    the ADP benchmark.
    """
    train_seasons = seasons_through(through)
    del train_seasons  # used by the implementation; validated here for the split
    raise NotImplementedError("Backtest is not implemented yet. See progress.md.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest projections vs. actual finish and ADP."
    )
    parser.add_argument(
        "--through",
        type=int,
        required=True,
        help="Last season in the training window; project the following season.",
    )
    parser.add_argument(
        "--model", default="baseline", help="Projection backend (default: baseline)."
    )
    args = parser.parse_args()
    span = seasons_through(args.through)
    print(
        f"[validation] scaffold only — would train the {args.model} model on "
        f"{span[0]}–{span[-1]} and score {args.through + 1} against ADP. "
        f"Not implemented yet (see progress.md)."
    )


if __name__ == "__main__":
    main()
