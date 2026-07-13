"""Stage-0 room-calibration tests - network-free (monkeypatched ADP)."""

from __future__ import annotations

import polars as pl
import pytest
import src.validation.room_calibration as rc


def _fake_adp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "position": ["RB", "QB", "TE", "K"],
            "adp": [10.0, 40.0, 36.5, 120.0],
            "adp_stdev": [3.0, 8.0, 5.0, 15.0],
        }
    )


def test_phase_labels_cover_all_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc, "load_adp", lambda year, fmt: _fake_adp())
    profile = rc.dispersion_profile([2024], "redraft_ppr")
    by_pos = {r["position"]: r["phase"] for r in profile.iter_rows(named=True)}
    assert by_pos["RB"] == "R1-3"
    assert by_pos["QB"] == "R4-8"
    # Regression: adp 36.5 (between phase bounds) must land in R4-8, and late
    # picks must actually reach R9-15 (a when-chain bug once swallowed both).
    assert by_pos["TE"] == "R4-8"
    assert by_pos["K"] == "R9-15"


def test_phase_targets_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc, "load_adp", lambda year, fmt: _fake_adp())
    targets = rc.phase_targets(rc.dispersion_profile([2024], "redraft_ppr"))
    r48 = targets.filter(pl.col("phase") == "R4-8").row(0, named=True)
    assert r48["n"] == 2
    assert r48["median_stdev"] == pytest.approx(6.5)  # (8 + 5) / 2
