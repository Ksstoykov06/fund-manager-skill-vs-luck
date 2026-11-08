# Fund manager skill vs luck

Do active fund managers who beat the market have skill, or would a world of
zero-skill managers produce the same winners by chance?

## Method
1. **Alphas.** Regress each fund's monthly excess return on the Carhart
   four factors (market, size, value, momentum). Funds need 36+ months.
2. **Bootstrap (Fama & French, 2010).** Strip out every fund's estimated
   alpha so true alpha is zero, resample calendar months jointly across all
   funds (keeping cross-fund correlation), re-estimate t(alpha). 1,000 runs
   give the cross-section of t-stats that luck alone produces. Compare with
   the real one, percentile by percentile.
3. **False discoveries (Barras, Scaillet & Wermers, 2010).** Use bootstrap
   p-values to estimate what share of funds is skilled, zero-alpha or
   unskilled, net of the funds that look significant by luck.

## Layout
| file | does |
|---|---|
| `fundskill/regression.py` | vectorised per-fund OLS on an unbalanced panel |
| `fundskill/bootstrap.py` | Fama-French bootstrap, percentile table, bootstrap p-values |
| `fundskill/fdr.py` | Barras-Scaillet-Wermers proportions, λ chosen by bootstrap MSE |
| `fundskill/data.py` | Ken French CSV parser, generic factor loader (Exeter UK), CRSP long→wide, price files→returns |
| `fundskill/simulate.py` | fake fund industry with known skill, for validation |
| `fundskill/plots.py` | actual vs luck t(alpha) distribution chart |
| `run_demo.py` | full study on simulated data |
| `tests/` | `pytest` checks (regression matches lstsq, null world, skill detection, parsers) |

## Validation on simulated funds (known truth)
1,500 funds; 5% skilled (+4%/yr), 20% unskilled (−3%/yr), the rest zero.
* Top 3% of real t-stats beat the luck distribution in 95–100% of runs: the planted skill is detected.
* FDR estimate: 4.2% skilled (true 5.4%), 10.7% unskilled (true 20.7%).
  Small negative alphas are hard to tell from noise, so the method is
  **conservative**: it undercounts modestly bad managers.

## Running on real data
```python
from fundskill import data, fama_french_bootstrap, fdr_table
ff  = data.load_ken_french("F-F_Research_Data_Factors_CSV.zip", "F-F_Momentum_Factor_CSV.zip")
raw = data.long_to_wide(crsp_monthly)          # crsp_fundno, caldt, mret
ex  = data.to_excess(raw, ff)
res = fama_french_bootstrap(ex, ff[["MKT_RF", "SMB", "HML", "MOM"]], n_sims=1000)
res.table()                                    # the headline table
fdr_table(res.actual.t_alpha, res.fund_pvalues())
```

## To do once data arrives
* Data source: CRSP (WRDS, US, survivor-bias-free) and/or Bloomberg (UK).
* Filters: actively managed equity funds only (drop index funds and ETFs), merge share classes.
* Net vs gross returns (add back expense ratio) to separate skill from fees.
* Sub-periods; survivor-only vs full-sample comparison to measure survivorship bias.
