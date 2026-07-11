"""Blend tests - network-free.

The blend anchors the board to market ADP with a validated model tilt; these
verify the limiting cases (w=0 -> pure ADP order, w=1 -> pure model order), the
middle case, tilt signs, and that ADP-less players append after the matched pool.
"""

from __future__ import annotations

import polars as pl
from src.decision.blend import blend_with_market
from src.names import norm_name_expr


def _board(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """(name, position_group, vor) -> minimal value board."""
    return pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(len(rows))],
            "player_name": [r[0] for r in rows],
            "position_group": [r[1] for r in rows],
            "vor": [r[2] for r in rows],
        }
    )


def _adp(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "adp_name": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "adp": [r[2] for r in rows],
        }
    ).with_columns(norm_name_expr("adp_name"))


BOARD = _board(
    [
        ("Elite Tight End", "TE", 140.0),  # model loves him; market drafts him late
        ("Consensus Back", "RB", 130.0),
        ("Consensus Wideout", "WR", 120.0),
        ("Market Darling", "RB", 60.0),  # market loves him; model does not
    ]
)
ADP = _adp(
    [
        ("Elite Tight End", "TE", 25.0),
        ("Consensus Back", "RB", 2.0),
        ("Consensus Wideout", "WR", 3.0),
        ("Market Darling", "RB", 1.0),
    ]
)


def _order(df: pl.DataFrame) -> list[str]:
    return df.sort("board_rank")["player_name"].to_list()


def test_weight_zero_is_pure_market_order() -> None:
    out = blend_with_market(BOARD, ADP, model_weight=0.0)
    assert _order(out) == [
        "Market Darling",
        "Consensus Back",
        "Consensus Wideout",
        "Elite Tight End",
    ]


def test_weight_one_is_pure_model_order() -> None:
    out = blend_with_market(BOARD, ADP, model_weight=1.0)
    assert _order(out) == [
        "Elite Tight End",
        "Consensus Back",
        "Consensus Wideout",
        "Market Darling",
    ]


def test_default_blend_anchors_to_market() -> None:
    out = blend_with_market(BOARD, ADP)  # default w
    # A minority tilt shifts scores but crosses no rank boundary in this small
    # pool - the market order holds, which is exactly the anchoring we want.
    assert _order(out) == [
        "Market Darling",
        "Consensus Back",
        "Consensus Wideout",
        "Elite Tight End",
    ]
    te = out.filter(pl.col("player_name") == "Elite Tight End").row(0, named=True)
    assert te["model_tilt"] == te["adp_market_rank"] - te["board_rank"]


def test_blend_reorders_adjacent_picks_on_strong_disagreement() -> None:
    board = _board(
        [
            ("Model Fave", "WR", 200.0),  # model's #1; market's #2
            ("Mid One", "RB", 120.0),
            ("Mid Two", "RB", 110.0),
            ("Mid Three", "TE", 100.0),
            ("Mid Four", "WR", 90.0),
            ("Mid Five", "RB", 80.0),
            ("Mid Six", "TE", 70.0),
            ("Model Fade", "WR", 50.0),  # market's #1; model's last
        ]
    )
    adp = _adp(
        [
            ("Model Fade", "WR", 1.0),
            ("Model Fave", "WR", 2.0),
            ("Mid One", "RB", 3.0),
            ("Mid Two", "RB", 4.0),
            ("Mid Three", "TE", 5.0),
            ("Mid Four", "WR", 6.0),
            ("Mid Five", "RB", 7.0),
            ("Mid Six", "TE", 8.0),
        ]
    )
    out = blend_with_market(board, adp)  # w=0.15
    # Fave blend 0.15*1+0.85*2=1.85 overtakes Fade 0.15*8+0.85*1=2.05.
    assert _order(out)[0] == "Model Fave"
    fave = out.filter(pl.col("player_name") == "Model Fave").row(0, named=True)
    fade = out.filter(pl.col("player_name") == "Model Fade").row(0, named=True)
    assert fave["model_tilt"] > 0  # model moved him up from market
    assert fade["model_tilt"] < 0  # model dragged him down


def test_dst_and_k_rank_purely_by_market() -> None:
    board = _board(
        [
            ("Alpha Wideout", "WR", 200.0),
            ("Great Defense", "DST", 500.0),  # model loves it - must NOT matter
            ("Meh Defense", "DST", 10.0),  # model hates it - must NOT matter
            ("Some Kicker", "K", 400.0),
        ]
    )
    adp = _adp(
        [
            ("Alpha Wideout", "WR", 1.0),
            ("Meh Defense", "DST", 2.0),  # market prefers the 'meh' one
            ("Great Defense", "DST", 3.0),
            ("Some Kicker", "K", 4.0),
        ]
    )
    out = blend_with_market(board, adp)
    # DST order follows the market exactly, ignoring the model's huge VOR gap.
    assert _order(out) == [
        "Alpha Wideout",
        "Meh Defense",
        "Great Defense",
        "Some Kicker",
    ]


def test_aux_model_participates_in_blend() -> None:
    board = _board(
        [
            ("Model Fave", "WR", 200.0),
            ("Model Fade", "WR", 50.0),
            ("Mid One", "RB", 120.0),
            ("Mid Two", "RB", 110.0),
            ("Mid Three", "TE", 100.0),
        ]
    )
    adp = _adp(
        [
            ("Model Fade", "WR", 1.0),
            ("Model Fave", "WR", 2.0),
            ("Mid One", "RB", 3.0),
            ("Mid Two", "RB", 4.0),
            ("Mid Three", "TE", 5.0),
        ]
    )
    # Aux model AGREES with the market (Fade best) -> with base 0.2/aux 0.1,
    # Fave: 0.2*1+0.1*2+0.7*2 = 1.8 ; Fade: 0.2*5+0.1*1+0.7*1 = 1.8 -> tie in
    # score; use a decisive aux instead: aux says Fade is best by far.
    aux = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(5)],
            "aux_vor": [10.0, 500.0, 120.0, 110.0, 100.0],  # Fade aux rank 1
        }
    )
    out = blend_with_market(board, adp, model_weight=0.2, aux=aux, aux_weight=0.3)
    # Fave: 0.2*1 + 0.3*5 + 0.5*2 = 2.7 ; Fade: 0.2*5 + 0.3*1 + 0.5*1 = 1.8
    assert _order(out)[0] == "Model Fade"  # the aux vote keeps the market's pick


def test_unmatched_players_append_after_matched_by_vor() -> None:
    board = pl.concat(
        [BOARD, _board([("Deep Rookie", "WR", 90.0), ("Deeper Rookie", "WR", 80.0)])]
    ).with_columns(pl.Series("player_id", [f"q{i}" for i in range(6)]))
    out = blend_with_market(board, ADP)
    tail = out.sort("board_rank").tail(2)
    assert tail["player_name"].to_list() == ["Deep Rookie", "Deeper Rookie"]
    assert tail["adp"].to_list() == [None, None]
    assert tail["model_tilt"].to_list() == [None, None]
    # Ranks continue after the matched pool with no gaps.
    assert out["board_rank"].sort().to_list() == list(range(1, 7))
