"""Quick sanity tests.  Run: python tests_analysis.py"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import analysis as an

df = an.clean_data(an.load_data())
assert df.isna().sum().sum() == 0, "cleaning should remove all missing values"
assert set(an.REQUIRED) <= set(df.columns)

# OLS must recover known coefficients on noiseless data
rng = np.random.default_rng(0)
t = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
t["y"] = 1.5 + 2.0 * t["a"] - 3.0 * t["b"]
tab, info, _ = an.ols(t, y="y", xs=("a", "b"))
assert np.allclose(tab["coef"].values, [1.5, 2.0, -3.0], atol=1e-3), tab
assert info["R2"] > 0.999

# bootstrap CI must contain the sample mean
ci, _ = an.bootstrap_ci(df["yield_t_ha"], B=500)
assert ci[0] < df["yield_t_ha"].mean() < ci[1]

# drier scenario should not increase expected production
assert an.monte_carlo_production(df, -100).mean() < an.monte_carlo_production(df, 0).mean()
print("All tests passed")
