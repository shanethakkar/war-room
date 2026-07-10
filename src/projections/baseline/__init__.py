"""Baseline projection backend — transparent and non-Bayesian.

Build FIRST (CLAUDE.md constraint #4). Approximates the hierarchical structure
with explicit, readable steps (design.md §4.1):

1. Team environment — project plays/game, pass rate, scoring context from pbp.
2. Share allocation — distribute team pass/rush volume via projected target,
   carry, and route shares. The load-bearing step.
3. Efficiency — YPT/YPC/catch rate regressed toward role/positional means,
   shrinkage weighted by sample size.
4. Touchdowns — from *expected* TDs (red-zone volume + air yards), never raw
   prior-year TDs; then regressed.

Uncertainty comes from empirical residual spread by role/position (design.md §5),
so the pipeline has intervals from day one.
"""
