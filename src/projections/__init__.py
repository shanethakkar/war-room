"""Projection layer.

Turns the player-season panel into next-season projections, each expressed as a
**distribution** rather than a point estimate. Two interchangeable implementations
behind one interface (design.md §3.2, §4):

- ``baseline`` — transparent regression + share allocation. Build FIRST; it is the
  benchmark the Bayesian model must beat.
- ``bayesian`` — PyMC hierarchical model (swap-in), emitting posterior predictive
  intervals.

Selection is by ``--model {baseline,bayesian}`` on the ``run`` entrypoint.
"""

from __future__ import annotations

MODELS: tuple[str, ...] = ("baseline", "bayesian")
"""Registered projection backends, selectable via ``run --model``."""
