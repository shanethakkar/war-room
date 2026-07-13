"""K/DST projection + flexible-format tests - network-free, synthetic frames."""

from __future__ import annotations

import polars as pl
import pytest
from src.formats import REDRAFT_PPR, customize, get_format
from src.formats.base import ScoringConfig
from src.formats.score import score_components, score_special
from src.ingest.adp import ffc_slug
from src.projections.special import ALL_COMPONENTS, project_special


def _special_row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "player_id": "DST_SF",
        "player_name": "San Francisco 49ers",
        "position": "DST",
        "position_group": "DST",
        "team": "SF",
        "season": 2023,
        "games": 17,
        **{c: 0.0 for c in ALL_COMPONENTS},
    }
    row.update(over)
    return row


def _panel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))


# --------------------------------------------------------------------- scoring
def test_kicker_scoring_hand_calc() -> None:
    df = pl.DataFrame(
        {
            "position_group": ["K"],
            **{c: [0.0] for c in ALL_COMPONENTS},
        }
    ).with_columns(
        pl.lit(20.0).alias("fg_0_39"),
        pl.lit(5.0).alias("fg_40_49"),
        pl.lit(2.0).alias("fg_50_plus"),
        pl.lit(3.0).alias("fg_missed"),
        pl.lit(30.0).alias("pat_made"),
        pl.lit(1.0).alias("pat_missed"),
    )
    scored = score_special(df, ScoringConfig())
    # 20*3 + 5*4 + 2*5 - 3 + 30 - 1 = 116
    assert scored["projected_points"][0] == pytest.approx(116.0)


def test_dst_scoring_hand_calc() -> None:
    df = pl.DataFrame(
        {
            "position_group": ["DST"],
            **{c: [0.0] for c in ALL_COMPONENTS},
        }
    ).with_columns(
        pl.lit(40.0).alias("sacks"),
        pl.lit(12.0).alias("ints"),
        pl.lit(8.0).alias("fumble_recs"),
        pl.lit(3.0).alias("dst_tds"),
        pl.lit(1.0).alias("safeties"),
        pl.lit(4.0).alias("games_pa_7_13"),
        pl.lit(9.0).alias("games_pa_14_20"),
        pl.lit(4.0).alias("games_pa_21_27"),
    )
    scored = score_special(df, ScoringConfig())
    # 40 + 24 + 16 + 18 + 2 + 4*4 + 9*1 + 4*0 = 125
    assert scored["projected_points"][0] == pytest.approx(125.0)


def test_te_premium_applies_only_to_te() -> None:
    df = pl.DataFrame(
        {
            "position_group": ["TE", "WR"],
            "pass_yards": [0.0, 0.0],
            "pass_tds": [0.0, 0.0],
            "interceptions": [0.0, 0.0],
            "rush_yards": [0.0, 0.0],
            "rush_tds": [0.0, 0.0],
            "receptions": [10.0, 10.0],
            "rec_yards": [0.0, 0.0],
            "rec_tds": [0.0, 0.0],
            "fumbles_lost": [0.0, 0.0],
        }
    )
    fmt = customize(REDRAFT_PPR, te_rec_bonus=0.5)
    scored = score_components(df, fmt.scoring)
    te, wr = scored["projected_points"].to_list()
    assert te == pytest.approx(15.0)  # 10 * (1.0 + 0.5)
    assert wr == pytest.approx(10.0)


# ------------------------------------------------------------------ customize
def test_customize_overrides_scoring_and_roster() -> None:
    fmt = customize(get_format("redraft_ppr"), pass_td=6.0, teams=10, qb=2)
    assert fmt.scoring.pass_td == 6.0
    assert fmt.roster.teams == 10
    assert fmt.roster.qb == 2
    assert fmt.scoring.rec == 1.0  # untouched fields keep the preset value


def test_customize_rejects_unknown_key() -> None:
    with pytest.raises(KeyError):
        customize(get_format("redraft_ppr"), not_a_setting=1)


def test_ffc_slug_resolution() -> None:
    assert ffc_slug("redraft_half") == "half-ppr"
    assert ffc_slug(customize(REDRAFT_PPR, qb=2)) == "2qb"
    assert ffc_slug(customize(REDRAFT_PPR, rec=0.5)) == "half-ppr"
    assert ffc_slug(customize(REDRAFT_PPR, rec=0.0)) == "standard"
    # Registered presets without an explicit FFC mapping resolve by their rules.
    assert ffc_slug("pigskin17") == "2qb"
    with pytest.raises(KeyError):
        ffc_slug("not_a_format")


def test_pigskin17_league_preset() -> None:
    fmt = get_format("pigskin17")
    assert fmt.roster.teams == 10
    assert fmt.roster.qb == 2
    assert fmt.roster.superflex == 0  # true 2QB, not superflex
    assert fmt.roster.bench == 7
    assert fmt.scoring.pass_td == 6.0
    assert fmt.scoring.rec == 1.0


# ----------------------------------------------------------------- projection
def test_project_special_is_leakage_free_and_regressed() -> None:
    strong = [
        _special_row(season=2022, sacks=60.0, games_pa_7_13=10.0, games_pa_14_20=7.0),
        _special_row(season=2023, sacks=55.0, games_pa_7_13=9.0, games_pa_14_20=8.0),
    ]
    weak = [
        _special_row(
            player_id="DST_CAR",
            player_name="Carolina Panthers",
            team="CAR",
            season=2022,
            sacks=20.0,
            games_pa_21_27=9.0,
            games_pa_28_34=8.0,
        ),
        _special_row(
            player_id="DST_CAR",
            player_name="Carolina Panthers",
            team="CAR",
            season=2023,
            sacks=25.0,
            games_pa_21_27=10.0,
            games_pa_28_34=7.0,
        ),
    ]
    # A absurd future row that must NOT affect the 2024 projection.
    future = [_special_row(season=2024, sacks=999.0)]
    proj = project_special(_panel(strong + weak + future), 2024)

    sf = proj.filter(pl.col("player_id") == "DST_SF").row(0, named=True)
    car = proj.filter(pl.col("player_id") == "DST_CAR").row(0, named=True)
    league_sacks = (60 + 55 + 20 + 25) / (4 * 17) * 17  # league mean per season
    # Ordering preserved but both shrunk toward the league mean; no leakage.
    assert car["sacks"] < league_sacks < sf["sacks"] < 57.5
    assert sf["projected_games"] == 17.0
    # PA bucket rates renormalize: expected bucket-games sum to ~17.
    buckets = [
        "games_pa_0",
        "games_pa_1_6",
        "games_pa_7_13",
        "games_pa_14_20",
        "games_pa_21_27",
        "games_pa_28_34",
        "games_pa_35_plus",
    ]
    assert sum(sf[b] for b in buckets) == pytest.approx(17.0)


def test_project_special_kicker_games_shrunk() -> None:
    rows = [
        _special_row(
            player_id="K1",
            player_name="Iron Leg",
            position="K",
            position_group="K",
            season=2023,
            games=10,
            fg_0_39=15.0,
            pat_made=20.0,
        )
    ]
    proj = project_special(_panel(rows), 2024)
    k = proj.row(0, named=True)
    # 0.85*10 + 0.15*16 = 10.9 projected games.
    assert k["projected_games"] == pytest.approx(10.9)
    # Lone kicker: the league mean IS his rate (1.5/gm), so shrink is a no-op
    # and the projection is rate x projected games.
    assert k["fg_0_39"] == pytest.approx(1.5 * 10.9)
