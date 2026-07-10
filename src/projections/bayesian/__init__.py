"""Bayesian projection backend — PyMC hierarchical model (swap-in).

Requires the optional ``bayesian`` extra (``uv sync --extra bayesian``). Does
properly what the baseline approximates (design.md §4.3):

- Group level: offensive environment (team, scheme, pace).
- Individual level: player role, **partially pooled** — thin-data players shrink
  toward role priors; strong-signal players pull away.
- Aging curves: position-specific, estimated hierarchically from history.

Critically, it emits **posterior predictive intervals** per player, which feed the
uncertainty layer directly. It becomes the default only if it beats the baseline
on the backtest scoreboard.
"""
