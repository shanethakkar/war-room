"""Validation & benchmarking.

The scoreboard (design.md §8). Every projection change is re-backtested here:

- **Backtest protocol** — train through season N, project N+1, compare to actual
  finish. No leakage across the split.
- **Accuracy** — rank correlation and MAE vs. actual end-of-season finish.
- **Calibration** — do the 80% intervals contain the outcome ~80% of the time?
- **Benchmark** — beat ADP (public via Sleeper) at predicting finish.
"""
