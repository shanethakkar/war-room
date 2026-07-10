"""Decision layer — format-aware.

Converts projection distributions into draftable value (design.md §7):

- **VOR** — projected points minus positional replacement level, where the
  replacement level is set by the format config.
- **Tiers** — derived from *distribution overlap*: a tier is a set of players
  whose credible intervals overlap enough that pick order barely matters
  (design.md §5), not a manual break.
- **ADP arbitrage board** — rank players by the disagreement between our
  projection distribution and market ADP (Sleeper). The headline pre-draft view.

Monte Carlo draft simulation and positional-run detection arrive in Phase 2.
"""
