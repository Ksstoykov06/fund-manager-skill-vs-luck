"""Per-fund factor regressions, vectorised across funds.

Every fund is regressed on the same factor matrix, but funds live for
different stretches of time (unbalanced panel). Rather than looping over
funds, we stack them and use a mask: month t counts for fund n only if
M[t, n] == 1. That lets us run thousands of regressions in one shot,
which matters because the bootstrap repeats this ~1,000 times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def batched_ols(Y: np.ndarray, M: np.ndarray, X: np.ndarray, min_obs: int):
    """OLS of each column of Y on X, using only rows where M == 1.

    Parameters
    ----------
    Y : (T, N) excess returns, any value where M == 0 (ignored)
    M : (T, N) 0/1 mask of observed months
    X : (T, P) regressors, first column must be the constant
    min_obs : funds with fewer observed months get NaN

    Returns
    -------
    coefs (N, P), t_alpha (N,), n_obs (N,)
    """
    Y = np.where(M > 0, Y, 0.0)
    M = M.astype(float)
    N = Y.shape[1]
    P = X.shape[1]

    n_obs = M.sum(axis=0)
    XtX = np.einsum("tn,tk,tl->nkl", M, X, X)
    Xty = np.einsum("tn,tk->nk", M * Y, X)
    yy = np.einsum("tn,tn->n", M * Y, Y)

    valid = n_obs >= max(min_obs, P + 1)
    # Guard against singular X'X (e.g. a fund alive for too few months).
    if valid.any():
        cond = np.full(N, np.inf)
        cond[valid] = np.linalg.cond(XtX[valid])
        valid &= cond < 1e12
    XtX_safe = np.where(valid[:, None, None], XtX, np.eye(P)[None])

    XtX_inv = np.linalg.inv(XtX_safe)
    coefs = np.einsum("nkl,nl->nk", XtX_inv, Xty)
    ssr = np.clip(yy - np.einsum("nk,nk->n", coefs, Xty), 0.0, None)
    dof = np.where(valid, n_obs - P, 1.0)
    s2 = ssr / dof
    se_alpha = np.sqrt(s2 * XtX_inv[:, 0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_alpha = coefs[:, 0] / se_alpha

    coefs[~valid] = np.nan
    t_alpha[~valid] = np.nan
    return coefs, t_alpha, n_obs


def align(returns: pd.DataFrame, factors: pd.DataFrame):
    """Line up fund returns and factors on common months.

    returns : wide DataFrame, index = month, columns = fund ids,
              values = monthly EXCESS returns (fund return minus risk-free),
              NaN when the fund did not exist.
    factors : DataFrame, index = month, columns = factor names
              (e.g. MKT_RF, SMB, HML, MOM), decimals not percent.
    """
    common = returns.index.intersection(factors.index)
    if len(common) == 0:
        raise ValueError("No overlapping months between fund returns and factors.")
    r = returns.loc[common]
    f = factors.loc[common]
    if f.isna().any().any():
        raise ValueError("Factor data has missing values in the sample period.")
    Y = r.to_numpy(dtype=float)
    M = (~np.isnan(Y)).astype(float)
    X = np.column_stack([np.ones(len(f)), f.to_numpy(dtype=float)])
    return Y, M, X, r.columns, list(f.columns), common


def fit_alphas(returns: pd.DataFrame, factors: pd.DataFrame, min_obs: int = 36) -> pd.DataFrame:
    """Estimate alpha, t(alpha) and factor loadings for every fund.

    Alpha is monthly; `alpha_ann` is simply 12 x monthly for readability.
    """
    Y, M, X, funds, fnames, _ = align(returns, factors)
    coefs, t_alpha, n_obs = batched_ols(Y, M, X, min_obs)
    out = pd.DataFrame(coefs, index=funds, columns=["alpha"] + [f"b_{c}" for c in fnames])
    out.insert(1, "t_alpha", t_alpha)
    out.insert(2, "n_obs", n_obs.astype(int))
    out.insert(1, "alpha_ann", out["alpha"] * 12)
    return out
