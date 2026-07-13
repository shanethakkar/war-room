"""ESPN league-history ingest tests - network-free, synthetic payloads.

The fetch itself needs the private league's cookies (exercised by the CLI);
here we test the pure pieces: .env parsing, payload -> pick-frame shaping,
identity attachment (incl. negative-id defenses), the bias math, and the sign
flip into the room simulator's shift convention.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from src.ingest.espn_league import (
    attach_identity,
    draft_frame,
    parse_env,
    pick_deltas,
    positional_bias,
    to_room_shift,
)


def _payload() -> dict[str, Any]:
    return {
        "draftDetail": {
            "picks": [
                {
                    "overallPickNumber": 1,
                    "roundId": 1,
                    "roundPickNumber": 1,
                    "teamId": 3,
                    "playerId": 1001,
                    "keeper": False,
                },
                {
                    "overallPickNumber": 2,
                    "roundId": 1,
                    "roundPickNumber": 2,
                    "teamId": 7,
                    "playerId": 2002,
                },
                {
                    "overallPickNumber": 3,
                    "roundId": 1,
                    "roundPickNumber": 3,
                    "teamId": 3,
                    "playerId": -16005,  # a team defense
                },
            ]
        },
        "teams": [
            {"id": 3, "name": "Todd the God"},
            {"id": 7, "location": "Team", "nickname": "Thakkar"},
        ],
    }


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "espn_id": ["1001", "2002", "9999"],
            "display_name": ["Josh Allen", "Bijan Robinson", "Someone Else"],
            "position": ["QB", "RB", "WR"],
        }
    )


def test_parse_env_strips_quotes_and_comments() -> None:
    text = '# creds\nESPN_S2="abc%2Bdef"\nSWID={AAA-BBB}\n\nbroken line\n'
    env = parse_env(text)
    assert env == {"ESPN_S2": "abc%2Bdef", "SWID": "{AAA-BBB}"}


def test_draft_frame_shapes_picks_and_team_names() -> None:
    df = draft_frame(_payload(), year=2024)
    assert df.height == 3
    assert df["overall"].to_list() == [1, 2, 3]
    # Both name styles resolve; missing keeper flag defaults False.
    assert df["team_name"].to_list() == ["Todd the God", "Team Thakkar", "Todd the God"]
    assert df["keeper"].to_list() == [False, False, False]


def test_attach_identity_maps_players_and_dst() -> None:
    picks = attach_identity(draft_frame(_payload(), year=2024), _players())
    rows = {r["espn_id"]: r for r in picks.iter_rows(named=True)}
    assert rows[1001]["norm_name"] == "josh allen"
    assert rows[1001]["position"] == "QB"
    # Negative id = defense: position forced to DST, no name mapping.
    assert rows[-16005]["position"] == "DST"
    assert rows[-16005]["norm_name"] is None


def test_bias_math_recovers_known_shifts() -> None:
    # Two seasons; QBs consistently drafted 10 picks EARLIER than market, the
    # RB 5 picks later in the one season it appears.
    picks = pl.DataFrame(
        {
            "year": [2023, 2023, 2024],
            "overall": [5, 20, 7],
            "norm_name": ["josh allen", "bijan robinson", "josh allen"],
            "position": ["QB", "RB", "QB"],
        }
    )
    adp = pl.DataFrame(
        {
            "norm_name": ["josh allen", "bijan robinson", "nobody"],
            "position": ["QB", "RB", "WR"],
            "adp": [15.0, 15.0, 50.0],
        }
    )
    deltas = pick_deltas(picks, adp)
    bias = positional_bias(deltas)
    by_pos = {r["position"]: r for r in bias.iter_rows(named=True)}
    assert by_pos["QB"]["mean_early"] == 9.0  # (+10 in 2023, +8 in 2024)
    assert by_pos["QB"]["years_early"] == 2
    assert by_pos["QB"]["years_total"] == 2
    assert by_pos["RB"]["mean_early"] == -5.0
    assert by_pos["RB"]["years_early"] == 0
    # Room-sim convention flips the sign: early (+9) -> shift -9.
    shift = to_room_shift(bias)
    assert shift["QB"] == -9.0
    assert shift["RB"] == 5.0


def test_pick_deltas_keeps_unmatched_as_null() -> None:
    picks = pl.DataFrame(
        {
            "year": [2024],
            "overall": [150],
            "norm_name": ["deep sleeper"],
            "position": ["WR"],
        }
    )
    adp = pl.DataFrame({"norm_name": ["someone"], "position": ["WR"], "adp": [10.0]})
    deltas = pick_deltas(picks, adp)
    assert deltas["early_by"].to_list() == [None]
