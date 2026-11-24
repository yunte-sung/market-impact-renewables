# settings for analysis scripts (DML and LWPR): which dataset, which columns, where outputs go
# choose the zone with an environment variable; without it, everything runs exactly as for the UK paper
# uv run python -m double_machine_learning.DML_overall_wind              -> UK (original)
# ZONE=DE_LU uv run python -m double_machine_learning.DML_overall_wind   -> Germany
# Optional: N_ITER=5 for a quick test run (default 100 bootstraps per window).

import os

from data_pipeline import schema
from zone_settings import ROOT

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
            "n_iterations": n_iter
        }
    return {
        "zone": zone,
        "data_path": ROOT / "data" / "processed" / f"{zone}.csv",
        "index_col": schema.TIMESTAMP,
        "cols": {c: c for c in schema.COLUMNS},       # processed files already use the standard names
        "output_dir": ROOT / "outputs" / zone,
        "file_suffix": "",
        "currency": "EUR",
        "solar_daylight_only": True,                  # solar model: hours with solar forecast > 0
        "n_iterations": n_iter
    }