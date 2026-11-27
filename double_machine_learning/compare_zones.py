# combines the normalized DML results of all zones into one comparison figure and one summary table
# run from repo root after all zones have run: uv run python -m double_machine_learning.compare_zones
# PERIOD=full compares the full-period results instead

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_settings import get_analysis_settings
from zone_settings import ROOT, list_zones

TECHS = ["wind", "solar"]
MEASURE = "cate_per_pct_load" # main comparison measure: EUR/MWh per 1% of average load
CHECK_LEVELS = {"wind": [10, 20, 30], "solar": [5, 10, 20]} # penetration levels (%) for the summary table

# window means and 80% bands (10th/90th bootstrap percentiles) for one zone and technology
def window_summary(path, tech):
    r = pd.read_csv(path)
    pen = f"mean_{tech}_penetration"
    g = r.groupby(pen)[MEASURE]
    out = pd.DataFrame({"mean": g.mean(), "lower": g.quantile(0.1), "upper": g.quantile(0.9)})
    return out.rename_axis("penetration").reset_index()

zones = list_zones()
results = {}
for zone in zones:
    S = get_analysis_settings(zone)
    for tech in TECHS:
        path = S["output_dir"] / f"results_{tech}_normalized.csv"
        if path.exists():
            results[(zone, tech)] = window_summary(path, tech)
        else:
            print(f"missing: {zone} {tech} ({path})")

out_dir = ROOT / "outputs" / "comparison" / S["output_dir"].name # e.g. outputs/comparison/2020-2025
out_dir.mkdir(parents=True, exist_ok=True)

# figure: one panel per technology, one line per zone; penetration on the y-axis as in the original UK figures
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
for ax, tech in zip(axes, TECHS):
    for zone in zones:
        if (zone, tech) not in results:
            continue
        w = results[(zone, tech)]
        line, = ax.plot(w["mean"], w["penetration"], lw=2, label=zone)
        ax.fill_betweenx(w["penetration"], w["lower"], w["upper"], color=line.get_color(), alpha=0.15)
    ax.axvline(0, color="gray", lw=0.8)
    ax.grid(axis="x", linestyle="-", alpha=0.3)
    ax.set_title(tech.capitalize(), fontsize=16)
    ax.set_xlabel("Effect on price (EUR/MWh per 1% of avg. load)", fontsize=14)
    ax.set_ylabel(f"Forecasted {tech} penetration (%)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "compare_zones.png")

# summary table: penetration range per zone, and the effect at fixed penetration levels (only inside a zone's range)
rows = []
for (zone, tech), w in results.items():
    row = {"zone": zone, "tech": tech,
           "pen_min": round(w["penetration"].min(), 1), "pen_max": round(w["penetration"].max(), 1)}
    for lvl in CHECK_LEVELS[tech]:
        inside = w["penetration"].min() <= lvl <= w["penetration"].max()
        row[f"at_{lvl}pct"] = round(float(np.interp(lvl, w["penetration"], w["mean"])), 2) if inside else np.nan
    rows.append(row)
table = pd.DataFrame(rows).sort_values(["tech", "zone"])
table.to_csv(out_dir / "compare_zones_summary.csv", index=False)
print(table.to_string(index=False))