"""What share of funds are skilled, unskilled, or zero-alpha?

Barras, Scaillet & Wermers (2010), "False Discoveries in Mutual Fund
Performance". The bootstrap asks "are the best funds better than luck?".
This asks a different question: "what FRACTION of the population is truly
skilled?", by subtracting the number of lucky funds we expect to see
from the number of funds that look significant.

Logic
-----
* Under zero alpha, p-values are uniform on [0, 1]. Skilled/unskilled
  funds pile up near p = 0. So the share of p-values above some lambda
  (e.g. 0.6) is mostly zero-alpha funds -> estimate pi0.
* At significance level gamma, we expect pi0 * gamma / 2 lucky funds in
  each tail. Significant positive funds minus that = truly skilled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_pi0(pvals: np.ndarray, lam: float | None = None,
                 grid=np.arange(0.30, 0.71, 0.05), n_boot: int = 1000, seed: int = 0) -> float:
    """Share of zero-alpha funds.

    lam=None picks lambda by bootstrap MSE (Storey 2002), as Barras,
    Scaillet & Wermers do: small lambda is biased, large lambda is noisy.
    """
    p = pvals[~np.isnan(pvals)]

    def pi0_at(x, l):
        return min(1.0, np.mean(x > l) / (1 - l))

    if lam is not None:
        return float(pi0_at(p, lam))
    est = np.array([pi0_at(p, l) for l in grid])
    rng = np.random.default_rng(seed)
    boots = np.array([[pi0_at(b, l) for l in grid]
                      for b in (rng.choice(p, size=len(p)) for _ in range(n_boot))])
    mse = ((boots - est.min()) ** 2).mean(axis=0)
    return float(est[np.argmin(mse)])


def fdr_table(t_alpha: pd.Series, pvals: pd.Series, lam: float | None = None,
              gammas=(0.05, 0.10, 0.20, 0.30, 0.40)) -> tuple[pd.DataFrame, dict]:
    """Return a per-gamma table and headline proportions.

    Table columns (all as % of funds):
      S_plus / S_minus : funds significant with positive / negative t
      F_plus / F_minus : expected lucky ones among them (false discoveries)
      T_plus / T_minus : estimated truly skilled / unskilled
      FDR_plus         : share of the 'significant winners' that are just lucky
    """
    df = pd.DataFrame({"t": t_alpha, "p": pvals}).dropna()
    p, t = df["p"].to_numpy(), df["t"].to_numpy()
    pi0 = estimate_pi0(p, lam)

    rows = []
    for g in gammas:
        s_plus = np.mean((p <= g) & (t > 0))
        s_minus = np.mean((p <= g) & (t < 0))
        f = pi0 * g / 2
        rows.append({
            "gamma": g,
            "S_plus": s_plus * 100, "F_plus": f * 100, "T_plus": max(s_plus - f, 0) * 100,
            "S_minus": s_minus * 100, "F_minus": f * 100, "T_minus": max(s_minus - f, 0) * 100,
            "FDR_plus": (min(f / s_plus, 1.0) * 100) if s_plus > 0 else np.nan,
        })
    table = pd.DataFrame(rows).set_index("gamma")

    # BSW take the proportions at a large gamma, where nearly all truly
    # non-zero funds are captured. We use the largest gamma supplied.
    g_star = table.index.max()
    pi_plus = table.loc[g_star, "T_plus"] / 100
    pi_minus = table.loc[g_star, "T_minus"] / 100
    # Note: the three shares are estimated separately, so they add up to
    # roughly (not exactly) 100%.
    headline = {
        "n_funds": len(df),
        "pi0_zero_alpha": pi0,
        "pi_plus_skilled": pi_plus,
        "pi_minus_unskilled": pi_minus,
    }
    return table, headline
