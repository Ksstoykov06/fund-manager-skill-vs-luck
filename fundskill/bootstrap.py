"""Skill vs luck: the Fama & French (2010) bootstrap.

Idea
----
1. Estimate each fund's alpha.
2. Subtract it from the fund's returns, so every fund now has a TRUE alpha
   of exactly zero, while keeping its real factor exposures, volatility,
   fat tails and lifetime.
3. Resample calendar months with replacement, the SAME months for every
   fund at once. This keeps the correlation between funds (they all hold
   overlapping stocks), which is the key improvement over resampling each
   fund separately.
4. Re-estimate t(alpha) for every fund in the resampled world. That gives a
   cross-section of t-stats produced by pure luck.
5. Repeat ~1,000 times and compare the REAL cross-section with the luck
   distribution, percentile by percentile.

Reading the result: if the real 95th-percentile t(alpha) beats the luck
world's 95th percentile in, say, 98% of simulations, the best funds are
better than luck alone would produce, which is evidence of skill in the right tail.
If the real t-stats sit BELOW the simulated ones, managers as a group are
worse than zero-alpha (usually because of fees).

We use t(alpha) rather than alpha because t-stats correct for funds with
short histories or high volatility that produce extreme alphas by chance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regression import align, batched_ols

DEFAULT_PCTS = (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99)


@dataclass
class BootstrapResult:
    actual: pd.DataFrame          # per-fund alpha / t_alpha from the real data
    sim_t: np.ndarray             # (n_sims, N) simulated t(alpha), NaN if fund excluded
    sim_alpha: np.ndarray         # (n_sims, N) simulated monthly alpha
    percentiles: tuple

    def table(self, stat: str = "t_alpha") -> pd.DataFrame:
        """Percentile comparison table, the main output of the study.

        Columns
        -------
        actual      : percentile of the real cross-section
        sim_mean    : average of that percentile across luck simulations
        pct_sim_lt  : % of simulations where the luck percentile is BELOW
                      the actual one. High (>95%) in the right tail = skill.
                      Low (<5%) in the left tail = genuinely bad managers.
        """
        if stat == "t_alpha":
            act, sim = self.actual["t_alpha"].to_numpy(), self.sim_t
        elif stat == "alpha_ann":
            act, sim = self.actual["alpha_ann"].to_numpy(), self.sim_alpha * 12
        else:
            raise ValueError("stat must be 't_alpha' or 'alpha_ann'")
        p = np.array(self.percentiles)
        act_p = np.nanpercentile(act, p)
        sim_p = np.nanpercentile(sim, p, axis=1).T  # (n_sims, len(p))
        return pd.DataFrame(
            {
                "actual": act_p,
                "sim_mean": sim_p.mean(axis=0),
                "pct_sim_lt": (sim_p < act_p).mean(axis=0) * 100,
            },
            index=pd.Index(p, name="percentile"),
        )

    def fund_pvalues(self) -> pd.Series:
        """Two-sided bootstrap p-value for each fund's t(alpha).

        Compares each fund's real |t| with its own luck distribution, which
        respects fat tails better than the textbook t-distribution.
        """
        act = self.actual["t_alpha"].to_numpy()
        sim = self.sim_t
        n_valid = np.sum(~np.isnan(sim), axis=0)
        with np.errstate(invalid="ignore"):
            exceed = np.nansum(np.abs(sim) >= np.abs(act)[None, :], axis=0)
            p = (exceed + 1) / (n_valid + 1)
        p = np.where(np.isnan(act) | (n_valid < 50), np.nan, p)
        return pd.Series(p, index=self.actual.index, name="p_boot")


def fama_french_bootstrap(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    n_sims: int = 1000,
    min_obs: int = 36,
    min_obs_sim: int | None = None,
    percentiles=DEFAULT_PCTS,
    seed: int = 0,
    chunk: int = 50,
) -> BootstrapResult:
    """Run the Fama-French (2010) zero-alpha bootstrap.

    min_obs     : months a fund needs in the real data to be included
    min_obs_sim : months a fund needs within one resample (defaults to
                  min_obs). Resampling with replacement makes this number
                  vary from run to run, as in the original paper.
    """
    from .regression import fit_alphas  # local import avoids a cycle

    min_obs_sim = min_obs if min_obs_sim is None else min_obs_sim
    actual = fit_alphas(returns, factors, min_obs=min_obs)
    Y, M, X, funds, _, _ = align(returns, factors)

    keep = ~np.isnan(actual["t_alpha"].to_numpy())
    alpha = np.nan_to_num(actual["alpha"].to_numpy())
    Y0 = np.where(M > 0, Y - alpha[None, :], 0.0)  # zero true alpha
    M0 = M * keep[None, :]                          # drop funds failing min_obs

    rng = np.random.default_rng(seed)
    T, N = Y.shape
    sim_t = np.full((n_sims, N), np.nan)
    sim_a = np.full((n_sims, N), np.nan)
    for s in range(n_sims):
        idx = rng.integers(0, T, size=T)
        coefs, t, _ = batched_ols(Y0[idx], M0[idx], X[idx], min_obs_sim)
        sim_t[s] = t
        sim_a[s] = coefs[:, 0]
    return BootstrapResult(actual=actual, sim_t=sim_t, sim_alpha=sim_a, percentiles=tuple(percentiles))
