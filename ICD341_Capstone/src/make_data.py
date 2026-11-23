"""Generate the SIMULATED maize dataset used for development.

IMPORTANT: these numbers are synthetic (seeded, reproducible). They are NOT
official statistics. To use real data, replace data/maize_production.csv with a
file that has the same columns (see README, "Using real data").
"""
import numpy as np
import pandas as pd
from pathlib import Path

DISTRICTS = {  # name: (baseline yield t/ha, mean area ha, rainfall mean mm)
    "Lusaka": (2.6, 9000, 850), "Chongwe": (2.3, 7000, 830),
    "Kabwe": (2.5, 8500, 880), "Chipata": (3.0, 14000, 980),
    "Mumbwa": (2.2, 9500, 800), "Mkushi": (3.1, 12000, 1000),
    "Choma": (2.7, 11000, 760), "Mazabuka": (2.8, 10000, 790),
}

def make_data(seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    for district, (base, area0, rain0) in DISTRICTS.items():
        for year in range(2010, 2025):
            rain = rng.normal(rain0, 120)
            fert = max(60, rng.normal(180 + 4 * (year - 2010), 35))
            area = max(1000, rng.normal(area0 * (1 + 0.015 * (year - 2010)), 900))
            yld = (base + 0.0015 * (rain - rain0) + 0.0040 * (fert - 180)
                   + 0.03 * (year - 2010) + rng.normal(0, 0.25))
            yld = max(0.8, yld)
            rows.append([district, year, round(area), round(rain, 1),
                         round(fert, 1), round(yld, 3), round(area * yld)])
    df = pd.DataFrame(rows, columns=["district", "year", "area_ha", "rainfall_mm",
                                     "fertilizer_kg_ha", "yield_t_ha", "production_t"])
    # a few missing values so the cleaning step is meaningful
    for col, n in [("rainfall_mm", 3), ("fertilizer_kg_ha", 3)]:
        df.loc[rng.choice(df.index, n, replace=False), col] = np.nan
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "maize_production.csv"
    make_data().to_csv(out, index=False)
    print("Wrote", out)
