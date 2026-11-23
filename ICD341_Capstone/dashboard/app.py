"""Interactive maize production dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import analysis as an  # noqa: E402

st.set_page_config(page_title="Zambia Maize Dashboard", layout="wide")
st.title("Zambia Maize Production Dashboard")
st.caption("ICD341 capstone. Data in data/maize_production.csv "
           "(simulated unless you replaced it with official data).")

@st.cache_data
def get_data():
    return an.clean_data(an.load_data())

df = get_data()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
districts = st.sidebar.multiselect("Districts", sorted(df["district"].unique()),
                                   default=sorted(df["district"].unique()))
y0, y1 = int(df["year"].min()), int(df["year"].max())
years = st.sidebar.slider("Years", y0, y1, (y0, y1))
view = df[df["district"].isin(districts) & df["year"].between(*years)]

if view.empty:
    st.warning("No data for this selection."); st.stop()

# ---- Headline metrics ----
by_d = view.groupby("district")["production_t"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total production", f"{view['production_t'].sum():,.0f} t")
c2.metric("Mean annual production / district", f"{by_d.mean():,.0f} t")
c3.metric("Highest-producing district", by_d.idxmax())
c4.metric("Lowest-producing district", by_d.idxmin())

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Drivers & model", "Risk simulation", "Data"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Mean production by district")
        fig, ax = plt.subplots()
        by_d.sort_values().plot.barh(ax=ax, color="#2E7D32"); ax.set_xlabel("Tonnes")
        st.pyplot(fig)
    with right:
        st.subheader("Mean yield over time")
        fig, ax = plt.subplots()
        view.groupby("year")["yield_t_ha"].mean().plot(marker="o", ax=ax)
        ax.set_ylabel("t/ha"); st.pyplot(fig)
    st.subheader("District detail")
    pick = st.selectbox("Choose a district", sorted(districts))
    d = view[view["district"] == pick].sort_values("year")
    st.line_chart(d.set_index("year")[["production_t"]])

with tab2:
    st.subheader("Regression: yield ~ rainfall + fertiliser + year")
    if len(view) > 10:
        table, info, _ = an.ols(view)
        st.dataframe(table)
        st.write(f"R² = {info['R2']}, adjusted R² = {info['adj_R2']}, n = {info['n']}")
        st.caption("Coefficient of rainfall_mm = change in yield (t/ha) per extra mm, "
                   "holding the other variables fixed. Association, not proof of cause.")
    else:
        st.info("Select more data (at least 11 rows) to fit the model.")
    xvar = st.radio("Scatter plot x-variable", ["rainfall_mm", "fertilizer_kg_ha"], horizontal=True)
    st.scatter_chart(view, x=xvar, y="yield_t_ha", color="district")

with tab3:
    st.subheader("Monte Carlo production simulation")
    shift = st.slider("Change in mean rainfall (mm)", -300, 300, -100, step=25)
    if len(view) > 10:
        sim = an.monte_carlo_production(view, rain_shift=shift, N=5000)
        base = an.monte_carlo_production(view, rain_shift=0, N=5000)
        a, b = st.columns(2)
        a.metric("Expected production (scenario)", f"{sim.mean():,.0f} t",
                 f"{100*(sim.mean()/base.mean()-1):+.1f}% vs baseline")
        lo, hi = np.percentile(sim, [5, 95])
        b.metric("90% interval", f"{lo:,.0f} to {hi:,.0f} t")
        fig, ax = plt.subplots()
        ax.hist(base, bins=50, alpha=.5, label="Baseline")
        ax.hist(sim, bins=50, alpha=.5, label=f"Rainfall {shift:+d} mm")
        ax.set_xlabel("Tonnes"); ax.legend(); st.pyplot(fig)
    else:
        st.info("Select more data to run the simulation.")

with tab4:
    st.dataframe(view, use_container_width=True)
    st.download_button("Download filtered data (CSV)", view.to_csv(index=False),
                       file_name="maize_filtered.csv", mime="text/csv")
