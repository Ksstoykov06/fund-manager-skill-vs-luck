import numpy as np
import pandas as pd
import pytest

from fundskill import simulate, fit_alphas, fama_french_bootstrap, fdr_table
from fundskill.data import parse_ken_french, long_to_wide


@pytest.fixture(scope="module")
def factors():
    return simulate.simulate_factors(240, seed=3)


def fac4(f):
    return f[["MKT_RF", "SMB", "HML", "MOM"]]


def test_regression_matches_lstsq(factors):
    R, _ = simulate.simulate_funds(factors, n_funds=50, seed=4)
    est = fit_alphas(R, fac4(factors), min_obs=24)
    for c in R.columns[:10]:
        y = R[c].dropna()
        if len(y) < 24:
            continue
        X = np.column_stack([np.ones(len(y)), fac4(factors).loc[y.index]])
        b, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - X @ b
        s2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        assert est.loc[c, "alpha"] == pytest.approx(b[0], abs=1e-12)
        assert est.loc[c, "t_alpha"] == pytest.approx(b[0] / se, rel=1e-8)


def test_short_funds_excluded(factors):
    R, _ = simulate.simulate_funds(factors, n_funds=200, seed=5)
    est = fit_alphas(R, fac4(factors), min_obs=60)
    assert est.loc[est.n_obs < 60, "t_alpha"].isna().all()


def test_null_world_looks_like_luck(factors):
    """No one is skilled -> actual percentiles sit inside the luck distribution."""
    R, _ = simulate.simulate_funds(factors, n_funds=400, share_skilled=0, share_unskilled=0,
                                   common_shock_vol=0.0, seed=6)
    res = fama_french_bootstrap(R, fac4(factors), n_sims=200, seed=1)
    tab = res.table()
    assert tab.loc[[5, 95], "pct_sim_lt"].between(2, 98).all()
    _, head = fdr_table(res.actual.t_alpha, res.fund_pvalues())
    assert head["pi0_zero_alpha"] > 0.85


def test_skill_detected_in_right_tail(factors):
    R, _ = simulate.simulate_funds(factors, n_funds=600, share_skilled=0.15, share_unskilled=0,
                                   alpha_skilled_ann=0.06, idio_vol=0.015, seed=7)
    res = fama_french_bootstrap(R, fac4(factors), n_sims=200, seed=2)
    assert res.table().loc[[95, 99], "pct_sim_lt"].min() > 95
    _, head = fdr_table(res.actual.t_alpha, res.fund_pvalues())
    assert head["pi_plus_skilled"] > 0.05


def test_parse_ken_french():
    text = ("This file was created by CMPT_ME_BEME_RETS\n\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "192607,    2.96,   -2.56,   -2.43,    0.22\n"
            "192608,    2.64,   -1.17,    3.82,    0.25\n"
            "\n Annual Factors: January-December \n"
            ",Mkt-RF,SMB,HML,RF\n"
            "1927,   29.47,   -2.04,   -4.54,    3.12\n")
    df = parse_ken_french(text)
    assert list(df.columns) == ["MKT_RF", "SMB", "HML", "RF"]
    assert len(df) == 2
    assert df.iloc[0]["MKT_RF"] == pytest.approx(0.0296)
    assert df.index[0] == pd.Timestamp("1926-07-31")


def test_long_to_wide():
    long = pd.DataFrame({"crsp_fundno": [1, 1, 2], "caldt": ["2020-01-31", "2020-02-29", "2020-02-15"],
                         "mret": [0.01, 0.02, -0.01]})
    w = long_to_wide(long)
    assert w.shape == (2, 2)
    assert np.isnan(w.loc["2020-01-31", 2])
    assert w.loc["2020-02-29", 2] == -0.01
