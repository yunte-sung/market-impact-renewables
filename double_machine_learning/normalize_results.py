# rescales the DML results (EUR/MWh per GW of forecast) into size-normalized effects for comparing countries
# cate_per_pct_load  = effect of extra wind/solar equal to 1% of the zone's average load (currency/MWh)
# Run from repo root (after DML_overall_wind.py and DML_overall_solar.py)
# ZONE=DE_LU uv run python -m double_machine_learning.normalize_results

import pandas as pd

from analysis_settings import get_analysis_settings, filter_period
from data_pipeline import schema

S = get_analysis_settings()
c = S["cols"]
CURRENCY = S["currency"]

# zone averages over the analysis period (all hours, same for wind and solar)
df = pd.read_csv(S["data_path"])
df[S["index_col"]] = pd.to_datetime(df[S["index_col"]])
df = filter_period(df.set_index(S["index_col"]), S)
dates = df.index
mean_load_mw = df[c[schema.LOAD_FORECAST]].mean()
mean_price = df[c[schema.PRICE]].mean()
gw_per_pct_load = mean_load_mw / 1000 / 100 # 1% of average load, in GW

factors = pd.DataFrame([{
    "zone": S["zone"],
    "period_start": str(dates.min()),
    "period_end": str(dates.max()),
    "mean_load_mw": round(mean_load_mw, 1),
    "gw_per_pct_load": round(gw_per_pct_load, 4),
    f"mean_price_{CURRENCY.lower()}_per_MWh": round(mean_price, 2)
}])
factors.to_csv(S["output_dir"] / "normalization_factors.csv", index=False)
print(factors.T.to_string(header=False))

# rescale each bootstrap estimate (a positive constant, so window means and quantiles scale the same way)
for tech in ["wind", "solar"]:
    path = S["output_dir"] / f"results_{tech}{S["file_suffix"]}.csv"
    if not path.exists():
        print(f"\n{tech}: no results file at {path}, skipped")
        continue
    r = pd.read_csv(path)
    r["cate_per_pct_load"] = r["cate"] * gw_per_pct_load
    r["cate_pct_of_price"] = r["cate_per_pct_load"] / mean_price * 100
    r.to_csv(S["output_dir"] / f"results_{tech}{S["file_suffix"]}_normalized.csv", index=False)

    pen = f"mean_{tech}_penetration"
    summary = r.groupby(pen)[["cate", "cate_per_pct_load", "cate_pct_of_price"]].mean()
    print(f"\n{tech}: window means (first, middle, last)")
    print(summary.iloc[[0, len(summary) // 2, -1]].round(2).to_string())