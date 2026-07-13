"""Scaffold smoke tests.

These assert the skeleton is wired correctly — the package imports, config is
sane, the format registry loads, and the API is constructed. They do NOT test
projection logic (there is none yet); real behavior gets real tests as each layer
lands.
"""

from __future__ import annotations

import pytest
from src import __version__
from src.config import DATA_START_SEASON, seasons_through
from src.formats import FORMATS, get_format
from src.projections import MODELS


def test_version() -> None:
    assert __version__


def test_data_window_starts_2016() -> None:
    # Locked decision (progress.md): train on 2016–present.
    assert DATA_START_SEASON == 2016


def test_seasons_through_is_inclusive() -> None:
    span = seasons_through(2024)
    assert span[0] == 2016
    assert span[-1] == 2024
    assert len(span) == 2024 - 2016 + 1


def test_seasons_through_rejects_pre_window() -> None:
    with pytest.raises(ValueError):
        seasons_through(DATA_START_SEASON - 1)


def test_preset_formats_registered() -> None:
    assert set(FORMATS) == {
        "redraft_ppr",
        "redraft_half",
        "redraft_standard",
        "superflex",
        "two_qb",
        "pigskin17",
    }


def test_full_ppr_reception_point() -> None:
    assert get_format("redraft_ppr").scoring.rec == 1.0


def test_superflex_has_the_edge_slot() -> None:
    # The single config difference that drives correct superflex QB valuation.
    assert get_format("redraft_ppr").roster.superflex == 0
    assert get_format("superflex").roster.superflex == 1


def test_get_format_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_format("does_not_exist")


def test_projection_models_registered() -> None:
    assert MODELS == ("baseline", "bayesian")


def test_api_app_constructs() -> None:
    # Import here so a broken FastAPI app fails this test, not collection.
    from src.api.main import app

    routes = {getattr(r, "path", None) for r in app.routes}
    assert {"/health", "/formats", "/"} <= routes
