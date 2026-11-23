"""Statistical analysis of maize production (ICD341 capstone)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "maize_production.csv"
OUT = ROOT / "outputs"
SEED = 123
REQUIRED = ["district", "year", "area_ha", "rainfall_mm",
            "fertilizer_kg_ha", "yield_t_ha", "production_t"]

# ---------- 1. Load and clean ----------
def load_data(path=DATA):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def clean_data(df):
    """Fill missing rainfall/fertiliser with the district median; drop duplicates."""
    df = df.drop_duplicates().copy()
    for col in ["rainfall_mm", "fertilizer_kg_ha"]:
        df[col] = df.groupby("district")[col].transform(lambda s: s.fillna(s.median()))
    return df.dropna().reset_index(drop=True)

# ---------- 2. Descriptive statistics ----------
def descriptive_stats(df):
    return df[["area_ha", "rainfall_mm", "fertilizer_kg_ha",
               "yield_t_ha", "production_t"]].describe().round(2)

def district_summary(df):
    g = df.groupby("district").agg(
        mean_production=("production_t", "mean"),
        sd_production=("production_t", "std"),
        mean_yield=("yield_t_ha", "mean"),
        cv_yield_pct=("yield_t_ha", lambda s: 100 * s.std() / s.mean()),
    ).round(2)
    return g.sort_values("mean_production", ascending=False)

# ---------- 3. Trend ----------
def national_trend(df):
    """Linear trend of average yield over time."""
    yearly = df.groupby("year")["yield_t_ha"].mean()
    res = stats.linregress(yearly.index, yearly.values)
    return yearly, res

# ---------- 4. Regression (OLS from first principles) ----------
def ols(df, y="yield_t_ha", xs=("rainfall_mm", "fertilizer_kg_ha", "year")):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in xs])
    yv = df[y].values
    n, k = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ yv)
    resid = yv - X @ beta
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    table = pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p},
                         index=["intercept", *xs]).round(5)
    return table, {"R2": round(float(r2), 4), "adj_R2": round(float(adj), 4), "n": int(n),
                   "resid_sd": round(float(np.sqrt(sigma2)), 4)}, resid

def ols_fixed_effects(df):
    """Same model plus district dummy variables (district fixed effects)."""
    d = pd.get_dummies(df["district"], prefix="d", drop_first=True, dtype=float)
    dfx = pd.concat([df, d], axis=1)
    return ols(dfx, xs=("rainfall_mm", "fertilizer_kg_ha", "year", *d.columns))

# ---------- 5. Bootstrap ----------
def bootstrap_ci(values, stat=np.mean, B=5000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = np.array([stat(rng.choice(values, len(values))) for _ in range(B)])
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]), boots

# ---------- 6. Monte Carlo: production risk under rainfall scenarios ----------
def monte_carlo_production(df, rain_shift=0.0, N=10000, seed=SEED):
    """Simulate a district-year's production using the fitted model.
    rain_shift: change in mean rainfall in mm (e.g. -100 = drier season)."""
    table, info, _ = ols(df)
    b = table["coef"].values
    rng = np.random.default_rng(seed)
    rain = rng.normal(df["rainfall_mm"].mean() + rain_shift, df["rainfall_mm"].std(), N)
    fert = rng.normal(df["fertilizer_kg_ha"].mean(), df["fertilizer_kg_ha"].std(), N)
    yld = (b[0] + b[1] * rain + b[2] * fert + b[3] * df["year"].max()
           + rng.normal(0, info["resid_sd"], N))
    return np.maximum(yld, 0) * df["area_ha"].mean()

# ---------- 7. Gaussian mixture on yields ----------
def gmm_yield_groups(df, k_range=(1, 2, 3, 4)):
    X = df[["yield_t_ha"]].values
    bics = {}
    for k in k_range:
        bics[k] = GaussianMixture(k, random_state=SEED, n_init=5).fit(X).bic(X)
    best = min(bics, key=bics.get)
    gmm = GaussianMixture(best, random_state=SEED, n_init=5).fit(X)
    return gmm, bics, gmm.predict(X)

# ---------- Figures ----------
def make_figures(df, out=OUT):
    out.mkdir(exist_ok=True)
    order = district_summary(df).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(order, df.groupby("district")["production_t"].mean().loc[order], color="#2E7D32")
    ax.set(title="Mean annual maize production by district", ylabel="Tonnes")
    plt.xticks(rotation=35, ha="right"); fig.tight_layout()
    fig.savefig(out / "production_by_district.png", dpi=150); plt.close(fig)

    yearly, res = national_trend(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(yearly.index, yearly.values, "o-", label="Mean yield")
    ax.plot(yearly.index, res.intercept + res.slope * yearly.index, "--",
            label=f"Trend ({res.slope:+.3f} t/ha per year)")
    ax.set(title="Average maize yield over time", xlabel="Year", ylabel="t/ha"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "yield_trend.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, c, lab in zip(axes, ["rainfall_mm", "fertilizer_kg_ha"],
                          ["Rainfall (mm)", "Fertiliser (kg/ha)"]):
        ax.scatter(df[c], df["yield_t_ha"], s=14, alpha=.6)
        m = stats.linregress(df[c], df["yield_t_ha"])
        xs = np.linspace(df[c].min(), df[c].max(), 50)
        ax.plot(xs, m.intercept + m.slope * xs, "r-")
        ax.set(xlabel=lab, ylabel="Yield (t/ha)", title=f"r = {m.rvalue:.2f}")
    fig.tight_layout(); fig.savefig(out / "yield_drivers.png", dpi=150); plt.close(fig)

    _, _, resid = ols(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(resid, plot=ax); ax.set_title("Residual Q-Q plot")
    fig.tight_layout(); fig.savefig(out / "residual_qq.png", dpi=150); plt.close(fig)

    gmm, bics, labels = gmm_yield_groups(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["yield_t_ha"], bins=25, edgecolor="black", alpha=.7)
    for m in gmm.means_.ravel():
        ax.axvline(m, color="red", ls="--")
    ax.set(title=f"Yield distribution, GMM components = {gmm.n_components} (BIC-selected)",
           xlabel="Yield (t/ha)", ylabel="Frequency")
    fig.tight_layout(); fig.savefig(out / "gmm_yield.png", dpi=150); plt.close(fig)

    base, dry = monte_carlo_production(df), monte_carlo_production(df, rain_shift=-100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(base, bins=60, alpha=.6, label="Baseline")
    ax.hist(dry, bins=60, alpha=.6, label="Rainfall -100 mm")
    ax.set(title="Monte Carlo: simulated production for an average district",
           xlabel="Tonnes", ylabel="Frequency"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "monte_carlo.png", dpi=150); plt.close(fig)

# ---------- Run everything ----------
def main():
    df = clean_data(load_data())
    OUT.mkdir(exist_ok=True)
    lines = []
    def p(*a):
        print(*a); lines.append(" ".join(str(x) for x in a))
    p("=== Descriptive statistics ==="); p(descriptive_stats(df).to_string())
    p("\n=== District summary ==="); p(district_summary(df).to_string())
    yearly, res = national_trend(df)
    p(f"\n=== Trend === slope={res.slope:.4f} t/ha/yr, r={res.rvalue:.3f}, p={res.pvalue:.4g}")
    table, info, _ = ols(df)
    p("\n=== OLS: yield ~ rainfall + fertiliser + year ==="); p(table.to_string()); p(info)
    tfe, ife, _ = ols_fixed_effects(df)
    p("\n=== OLS with district fixed effects ==="); p(tfe.to_string()); p(ife)
    ci, _ = bootstrap_ci(df["yield_t_ha"])
    p(f"\n=== Bootstrap 95% CI for mean yield === [{ci[0]:.3f}, {ci[1]:.3f}] t/ha")
    base, dry = monte_carlo_production(df), monte_carlo_production(df, rain_shift=-100)
    p(f"\n=== Monte Carlo === mean baseline={base.mean():,.0f} t; mean dry={dry.mean():,.0f} t "
      f"({100*(dry.mean()/base.mean()-1):+.1f}%); P(production<20000 t): "
      f"baseline={np.mean(base<20000):.3f}, dry={np.mean(dry<20000):.3f}")
    gmm, bics, _ = gmm_yield_groups(df)
    p("\n=== GMM BIC ===", {k: round(v, 1) for k, v in bics.items()}, "-> best k =", gmm.n_components)
    make_figures(df)
    (OUT / "results_summary.txt").write_text("\n".join(lines))
    p("\nFigures and summary saved to", OUT)

if __name__ == "__main__":
    matplotlib.use("Agg")   # script mode: save figures without a display
    main()
