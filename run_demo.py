"""Run the full study on a simulated fund industry with KNOWN skill.

    python run_demo.py            # 1,000 bootstrap runs, ~30s

Once real data arrives, the same three calls apply:
    res = fama_french_bootstrap(excess_returns, factors[["MKT_RF","SMB","HML","MOM"]])
    res.table()
    fdr_table(res.actual.t_alpha, res.fund_pvalues())
"""
from pathlib import Path

import pandas as pd

from fundskill import simulate, fama_french_bootstrap, fdr_table
from fundskill.plots import plot_skill_vs_luck

OUT = Path("results")
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 120)

factors = simulate.simulate_factors(n_months=300, seed=0)
returns, true_alpha = simulate.simulate_funds(
    factors, n_funds=1500, share_skilled=0.05, share_unskilled=0.20,
    alpha_skilled_ann=0.04, alpha_unskilled_ann=-0.03, idio_vol=0.015, seed=1)
fac = factors[["MKT_RF", "SMB", "HML", "MOM"]]

res = fama_french_bootstrap(returns, fac, n_sims=1000, min_obs=36, seed=42)

print("\n=== Percentiles of t(alpha): actual vs luck ===")
tab = res.table("t_alpha").round(2)
print(tab)
tab.to_csv(OUT / "percentile_table_t.csv")
res.table("alpha_ann").round(4).to_csv(OUT / "percentile_table_alpha.csv")

fdr, head = fdr_table(res.actual["t_alpha"], res.fund_pvalues())
print("\n=== False discovery rate (BSW 2010), % of funds ===")
print(fdr.round(1))
fdr.round(2).to_csv(OUT / "fdr_table.csv")

incl = res.actual["t_alpha"].notna()
truth = true_alpha[incl]
print("\n=== Estimated vs TRUE shares ===")
print(f"{'':12}{'estimated':>10}{'true':>8}")
print(f"{'skilled':12}{head['pi_plus_skilled']:>10.1%}{(truth > 0).mean():>8.1%}")
print(f"{'zero alpha':12}{head['pi0_zero_alpha']:>10.1%}{(truth == 0).mean():>8.1%}")
print(f"{'unskilled':12}{head['pi_minus_unskilled']:>10.1%}{(truth < 0).mean():>8.1%}")

plot_skill_vs_luck(res, OUT / "skill_vs_luck.png",
                   title="Simulated funds: actual t(alpha) vs pure luck")
print(f"\nSaved tables and chart to {OUT.resolve()}")
