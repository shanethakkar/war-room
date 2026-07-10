"""Projection entrypoint: feature panel → projected distributions for a season.

    uv run python -m src.projections.run --season 2025 [--model baseline|bayesian]

Baseline is the default. Bayesian is a swap-in that must beat the baseline on the
backtest scoreboard before it becomes default (CLAUDE.md constraint #4). Not
implemented yet (see progress.md).
"""

from __future__ import annotations

import argparse

from src.projections import MODELS


def run(*, season: int, model: str = "baseline") -> None:
    """Produce projections for ``season`` using the chosen backend.

    Both backends emit a point projection *and* an uncertainty interval per
    player (design.md §5) — the distribution is the product. ``model`` selects the
    interchangeable implementation; the interface is identical.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}.")
    raise NotImplementedError(
        f"Projections ({model}) are not implemented yet. See progress.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run player projections for a season.")
    parser.add_argument("--season", type=int, required=True, help="Season to project.")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="baseline",
        help="Projection backend (default: baseline).",
    )
    args = parser.parse_args()
    print(
        f"[projections] scaffold only — would run the {args.model} model for "
        f"{args.season}. Not implemented yet (see progress.md)."
    )


if __name__ == "__main__":
    main()
