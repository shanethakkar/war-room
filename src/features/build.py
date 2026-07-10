"""Feature entrypoint: cached nflverse tables → player-season feature panel.

    uv run python -m src.features.build

Not implemented yet (see progress.md).
"""

from __future__ import annotations

import argparse


def build_panel() -> None:
    """Build the player-season panel from the cached weekly, pbp, and ff-opportunity
    tables.

    Volume/role features are the load-bearing output (design.md §4.1–4.2):
    target share, carry share, route participation, snap share, red-zone volume,
    and air yards — the inputs the projection layer allocates against.
    """
    raise NotImplementedError(
        "Feature panel build is not implemented yet. See progress.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the player-season feature panel."
    )
    parser.parse_args()
    print(
        "[features] scaffold only — would build the player-season panel from the "
        "Parquet cache. Not implemented yet (see progress.md)."
    )


if __name__ == "__main__":
    main()
