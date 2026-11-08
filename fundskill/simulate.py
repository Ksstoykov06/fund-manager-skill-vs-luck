"""Fake fund industry with KNOWN skill, for testing the method.

If the bootstrap and FDR can recover skill we planted ourselves, we can
trust what they say about real funds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Rough monthly means / vols in line with US factor history.
FACTOR_SPECS = {"MKT_RF": (0.006, 0.045), "SMB": (0.002, 0.030),
                "HML": (0.003, 0.030), "MOM": (0.006, 0.045)}


def simulate_factors(n_months: int = 300, seed: int = 0, start="2000-01-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    data = {k: rng.normal(m, s, n_months) for k, (m, s) in FACTOR_SPECS.items()}
    data["RF"] = np.full(n_months, 0.002)
    return pd.DataFrame(data, index=idx)


def simulate_funds(factors: pd.DataFrame, n_funds: int = 1000, share_skilled: float = 0.05,
                   share_unskilled: float = 0.20, alpha_skilled_ann: float = 0.036,
                   alpha_unskilled_ann: float = -0.024, idio_vol: float = 0.02,
                   common_shock_vol: float = 0.005, min_life: int = 24, t_dist_df: int = 5,
                   seed: int = 1):
    """Return (excess returns wide DataFrame, true annual alpha Series).

    Features that make it realistic enough to stress the method:
      * unbalanced panel: funds launch and close at random
      * fat-tailed residuals (Student-t)
      * a common shock, so residuals are correlated across funds
    """
    rng = np.random.default_rng(seed)
    T = len(factors)
    F = factors[["MKT_RF", "SMB", "HML", "MOM"]].to_numpy()

    labels = rng.choice(3, size=n_funds, p=[1 - share_skilled - share_unskilled,
                                            share_skilled, share_unskilled])
    alpha_ann = np.select([labels == 1, labels == 2],
                          [alpha_skilled_ann, alpha_unskilled_ann], 0.0)
    betas = np.column_stack([
        rng.normal(1.0, 0.15, n_funds),   # market
        rng.normal(0.2, 0.30, n_funds),   # size
        rng.normal(0.0, 0.30, n_funds),   # value
        rng.normal(0.0, 0.10, n_funds),   # momentum
    ])
    scale = np.sqrt((t_dist_df - 2) / t_dist_df)
    eps = rng.standard_t(t_dist_df, size=(T, n_funds)) * scale * idio_vol
    shock = rng.normal(0, common_shock_vol, size=(T, 1))
    shock -= shock.mean()  # correlated noise, not a hidden source of alpha
    common = shock * rng.normal(1, 0.3, n_funds)[None]
    R = alpha_ann[None] / 12 + F @ betas.T + eps + common

    # Random lifetimes.
    start = rng.integers(0, T - min_life, n_funds)
    life = rng.integers(min_life, T + 1, n_funds)
    end = np.minimum(start + life, T)
    t_idx = np.arange(T)[:, None]
    alive = (t_idx >= start[None]) & (t_idx < end[None])
    R = np.where(alive, R, np.nan)

    cols = [f"F{i:04d}" for i in range(n_funds)]
    return (pd.DataFrame(R, index=factors.index, columns=cols),
            pd.Series(alpha_ann, index=cols, name="true_alpha_ann"))
