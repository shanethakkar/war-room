"""Hierarchical Bayesian ppg model in PyMC (design.md §4.3).

Predicts next-season points-per-game with the structure the baseline only
approximates:

- **Partially-pooled player effects** - a per-player random intercept
  ``u ~ Normal(0, tau)``; thin-data players shrink to the position role, strong
  signals pull away. tau is learned.
- **Position-varying slopes** - each opportunity/efficiency predictor's effect is
  drawn per position from a shared hyper-prior (partial pooling across positions).
- **Position-specific aging curves** - age and age^2 are predictors with
  position-varying slopes, so each position gets its own curve.
- **Heteroscedastic, fat-tailed noise** - Student-T with a spread that grows with
  prior volume, so boom/bust and volume risk fall out of the posterior predictive.

The fit emits full posteriors; ``FitResult.predict`` draws the posterior
predictive per player (new players draw their effect from the population), giving
calibrated intervals directly - the Bayesian replacement for the empirical ones.

PyMC is the sanctioned pandas/numpy boundary (CLAUDE.md #2): Polars in, numpy at
the model edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pymc as pm

from src.projections.bayesian.features import PREDICTORS

_PREV_PPG = PREDICTORS.index("prev_ppg")

# Only players with at least this many training seasons get their own random
# effect; thinner players pool to their position (they shrink to ~0 anyway). This
# keeps the sampled parameter space small enough to fit in minutes, not hours.
MIN_PLAYER_SEASONS = 3


@dataclass
class FitResult:
    """Flattened posterior draws + the maps needed for posterior prediction."""

    alpha: np.ndarray  # [S, n_pos]
    beta: np.ndarray  # [S, n_pred, n_pos]
    u: np.ndarray  # [S, n_player]
    tau: np.ndarray  # [S]
    gamma0: np.ndarray  # [S, n_pos]
    gamma1: np.ndarray  # [S]
    nu: np.ndarray  # [S]
    pos_index: dict[str, int]
    player_index: dict[str, int]
    means: dict[str, float]
    sds: dict[str, float]

    def _z(self, features: pl.DataFrame) -> np.ndarray:
        cols = [
            (features[p].to_numpy() - self.means[p]) / self.sds[p] for p in PREDICTORS
        ]
        return np.column_stack(cols)

    def predict(self, features: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
        """Posterior-predictive season points per player: mean + 80% interval.

        ``features`` must carry PREDICTORS, ``player_id``, ``position_group`` and a
        ``projected_games`` column. Players absent from training draw their effect
        from ``Normal(0, tau)`` (wider intervals - we have not seen them).
        """
        rng = np.random.default_rng(seed)
        z = self._z(features)
        pos_idx = np.array(
            [self.pos_index[g] for g in features["position_group"].to_list()]
        )
        player_pos = [
            self.player_index.get(pid, -1) for pid in features["player_id"].to_list()
        ]
        games = features["projected_games"].to_numpy()
        n_draws = self.alpha.shape[0]

        means, lows, meds, highs = [], [], [], []
        for i in range(features.height):
            p = pos_idx[i]
            mu = self.alpha[:, p] + (self.beta[:, :, p] * z[i][None, :]).sum(axis=1)
            if player_pos[i] >= 0:
                mu = mu + self.u[:, player_pos[i]]
            else:
                mu = mu + rng.standard_normal(n_draws) * self.tau
            sigma = np.exp(self.gamma0[:, p] + self.gamma1 * z[i][_PREV_PPG])
            ppg = mu + sigma * rng.standard_t(self.nu)
            points = np.clip(ppg, 0.0, None) * games[i]
            means.append(float(points.mean()))
            lows.append(float(np.percentile(points, 10)))
            meds.append(float(np.percentile(points, 50)))
            highs.append(float(np.percentile(points, 90)))

        return features.select("player_id").with_columns(
            pl.Series("projected_points", means),
            pl.Series("points_low", lows),
            pl.Series("points_median", meds),
            pl.Series("points_high", highs),
        )


def _standardize(
    pairs: pl.DataFrame,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    cols: list[np.ndarray] = []
    for predictor in PREDICTORS:
        arr = pairs[predictor].to_numpy().astype(float)
        mean = float(arr.mean())
        sd = float(arr.std()) or 1.0  # guard a constant predictor
        means[predictor], sds[predictor] = mean, sd
        cols.append((arr - mean) / sd)
    return np.column_stack(cols), means, sds


def _stack(posterior: Any, name: str, dims: tuple[str, ...]) -> np.ndarray:
    """Flatten a posterior variable to [sample, *dims] as a numpy array."""
    da = posterior[name].stack(sample=("chain", "draw")).transpose("sample", *dims)
    return np.asarray(da.values, dtype=float)


def fit_model(
    pairs: pl.DataFrame,
    *,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    cores: int = 2,
    target_accept: float = 0.9,
    nuts_sampler: str = "nutpie",
    seed: int = 0,
) -> FitResult:
    """Fit the hierarchical ppg model and return flattened posterior draws."""
    x, means, sds = _standardize(pairs)
    positions = sorted(pairs["position_group"].unique().to_list())
    pos_index = {g: i for i, g in enumerate(positions)}
    pos_idx = np.array([pos_index[g] for g in pairs["position_group"].to_list()])

    # Random effects only for players with enough seasons; others pool (u = 0).
    counts = pairs["player_id"].value_counts()
    established = sorted(
        counts.filter(pl.col("count") >= MIN_PLAYER_SEASONS)["player_id"].to_list()
    )
    player_index = {pid: i for i, pid in enumerate(established)}
    raw_idx = np.array(
        [player_index.get(pid, -1) for pid in pairs["player_id"].to_list()]
    )
    u_mask = (raw_idx >= 0).astype(float)
    u_safe = np.where(raw_idx >= 0, raw_idx, 0)

    y = pairs["target_ppg"].to_numpy()
    ppg_mean = float(y.mean())

    coords = {"pos": positions, "pred": list(PREDICTORS), "player": established}
    with pm.Model(coords=coords):
        mu_alpha = pm.Normal("mu_alpha", ppg_mean, 5.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 3.0)
        alpha = pm.Deterministic(
            "alpha",
            mu_alpha + sigma_alpha * pm.Normal("alpha_z", 0.0, 1.0, dims="pos"),
            dims="pos",
        )
        mu_beta = pm.Normal("mu_beta", 0.0, 1.0, dims="pred")
        sigma_beta = pm.HalfNormal("sigma_beta", 1.0, dims="pred")
        beta = pm.Deterministic(
            "beta",
            mu_beta[:, None]
            + sigma_beta[:, None] * pm.Normal("beta_z", 0.0, 1.0, dims=("pred", "pos")),
            dims=("pred", "pos"),
        )
        tau = pm.HalfNormal("tau", 2.0)
        u = pm.Deterministic(
            "u", tau * pm.Normal("u_z", 0.0, 1.0, dims="player"), dims="player"
        )
        gamma0 = pm.Normal("gamma0", np.log(3.0), 0.5, dims="pos")
        gamma1 = pm.Normal("gamma1", 0.0, 0.5)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)

        contrib = (x * beta.T[pos_idx]).sum(axis=1)
        u_contrib = u[u_safe] * u_mask  # 0 for pooled (non-established) players
        mu = alpha[pos_idx] + contrib + u_contrib
        sigma = pm.math.exp(gamma0[pos_idx] + gamma1 * x[:, _PREV_PPG])
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            random_seed=seed,
            progressbar=False,
        )

    post = idata.posterior
    return FitResult(
        alpha=_stack(post, "alpha", ("pos",)),
        beta=_stack(post, "beta", ("pred", "pos")),
        u=_stack(post, "u", ("player",)),
        tau=_stack(post, "tau", ()),
        gamma0=_stack(post, "gamma0", ("pos",)),
        gamma1=_stack(post, "gamma1", ()),
        nu=_stack(post, "nu", ()),
        pos_index=pos_index,
        player_index=player_index,
        means=means,
        sds=sds,
    )
