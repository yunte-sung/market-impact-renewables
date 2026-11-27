# settings for analysis scripts (DML and LWPR): which dataset, which columns, where outputs go
# choose the zone with an environment variable; without it, everything runs exactly as for the UK paper
# uv run python -m double_machine_learning.DML_overall_wind              -> UK (original)
# ZONE=DE_LU uv run python -m double_machine_learning.DML_overall_wind   -> Germany
# Optional: N_ITER=5 for a quick test run (default 100 bootstraps per window).

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd

from data_pipeline import schema
from zone_settings import ROOT, get_zone

# column names in the original UK dataset mapped to standard names in schema.py
UK_COLUMNS = {
    schema.PRICE: "Day Ahead APX Price",
    schema.LOAD_FORECAST: "estimated_load",
    schema.WIND_FORECAST: "wind_forecast",
    schema.SOLAR_FORECAST: "solar_forecast",
    schema.WIND_CAPACITY: "wind_capacity",
    schema.SOLAR_CAPACITY: "solar_capacity",
    schema.GAS_PRICE: "MID_NBP",
    schema.CO2_PRICE: "MID_CO2",
    schema.WIND_PENETRATION: "forecasted_wind_penetration",
    schema.SOLAR_PENETRATION: "forecasted_solar_penetration",
    schema.HOUR: "Hour",
    schema.DAY_OF_WEEK: "Day",
    schema.MONTH: "Month",
    schema.YEAR: "Year",
    schema.DAYLIGHT_HOURS: "Daylight Hours",
}

# common analysis period for cross-country comparison (inclusive dates, zone local time)
# PERIOD=full uses each zone's whole period from zones.yaml instead (robustness)
COMMON_PERIOD = ("2020-01-01", "2025-12-31")

# keep only rows inside the analysis period; df must have a timezone-aware DatetimeIndex
def filter_period(df, S):
    if S["period_start"] is None:
        return df
    local_dates = df.index.tz_convert(S["timezone"]).tz_localize(None)
    start = pd.Timestamp(S["period_start"])
    end = pd.Timestamp(S["period_end"]) + pd.Timedelta(days=1) # end date inclusive
    return df[(local_dates >= start) & (local_dates < end)]

# plot smoothing over neighbouring windows; sigma = 0 means no smoothing (returns values unchanged)
def smooth(values, sigma):
    if sigma == 0:
        return np.asarray(values)
    return gaussian_filter1d(values, sigma=sigma)

# analysis settings for one zone - zone defaults to the ZONE environment variable, otherwise uses UK
def get_analysis_settings(zone=None):
    zone = zone or os.environ.get("ZONE", "UK")
    n_iter = int(os.environ.get("N_ITER", 100))
    if zone == "UK":
        return {
            "zone": "UK",
            "data_path": ROOT / "data" / "market_impact_renewables.csv",
            "index_col": "Date",
            "cols": UK_COLUMNS,
            "output_dir": ROOT / "outputs",
            "file_suffix": "_apx",
            "currency": "GBP",
            "solar_daylight_only": False, # original UK setup includes all hours for solar
            "n_iterations": n_iter,
            "plot_smoothing_sigma": None,     # None = original script rules (reproduces the paper); or a fixed number of windows
            "period_start": None,
            "period_end": None,
            "timezone": None
        }
    period = os.environ.get("PERIOD", "common")
    if period == "common":
        period_start, period_end = COMMON_PERIOD
        label = f"{period_start[:4]}-{period_end[:4]}"
    elif period == "full":
        period_start, period_end, label = None, None, "full"
    else:
        raise ValueError(f"PERIOD must be 'common' or 'full', got {period!r}")
    return {
        "zone": zone,
        "data_path": ROOT / "data" / "processed" / f"{zone}.csv",
        "index_col": schema.TIMESTAMP,
        "cols": {c: c for c in schema.COLUMNS},       # processed files already use the standard names
        "output_dir": ROOT / "outputs" / zone / label,
        "file_suffix": "",
        "currency": "EUR",
        "solar_daylight_only": True,                  # solar model: hours with the sun above horizon
        "n_iterations": n_iter,
        "plot_smoothing_sigma": 0,        # 0 = no smoothing (default); a number = fixed sigma in windows
        "period_start": period_start,
        "period_end": period_end,
        "timezone": get_zone(zone)["timezone"]
    }