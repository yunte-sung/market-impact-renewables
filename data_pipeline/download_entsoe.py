# downloads raw ENTSO-E data for one zone, one file per series per year
# to run from repo root: uv run python -m data_pipeline.download_entsoe --zone DE_LU

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from zone_settings import ROOT, get_zone

RAW_DIR = ROOT / "data" / "raw" / "entsoe"

# one entry per series - format for API call
SERIES = {
    "price": lambda c, area, s, e: c.query_day_ahead_prices(area, start=s, end=e, include_currency=True),
    "load_forecast": lambda c, area, s, e: c.query_load_forecast(area, start=s, end=e),
    "wind_solar_forecast": lambda c, area, s, e: c.query_wind_and_solar_forecast(area, start=s, end=e),
    "capacity": lambda c, area, s, e: c.query_installed_generation_capacity(area, start=s, end=e)
}

# time period for data download with start date (inclusive) and end date (exclusive)
def year_window(zone, year, series):
    tz = zone["timezone"]
    year_start = pd.Timestamp(f"{year}-01-01", tz=tz)
    year_end = pd.Timestamp(f"{year + 1}-01-01", tz=tz)
    if series == "capacity": # because capacity only has one annual value with date some time early in January
        return year_start, year_end
    return max(year_start, zone["start"]), min(year_end, zone["end"])

# download data for a zone
def download_zone(code, series_list=None, years=None, force=False): # can choose a subset of the SERIES lambda functions if not using the entire set
    zone=get_zone(code)
    load_dotenv(ROOT / ".env") # loads API key, only works if .env exists
    client = EntsoePandasClient(api_key = os.getenv("ENTSOE_API_KEY"))

    out_dir = RAW_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True) # creates directory if it doesn't already exist
    default_years = range(zone["start"].year, zone["end"].year) # the zone's period, capacity gets one extra year for extrapolation
    log = [] # empty log to track download progress

    # loop to download data across all (or selected) variables
    for series in series_list or SERIES:
        if years: # years specified use only those
            series_years = years
        elif series == "capacity":
            series_years = range(zone["start"].year, zone["end"].year + 1) # additional year added for capacity extrapolation
        else:
            series_years = default_years
        for year in series_years:
            path = out_dir / f"{series}_{year}.csv"
            if path.exists() and not force:
                log.append((series, year, "skipped (exists)", None))
                continue
            start, end = year_window(zone, year, series)
            try:
                data = SERIES[series](client, zone["entsoe_code"], start, end)
                data = data[data.index < end] # drop the extra end row
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=series)
                data.to_csv(path)
                log.append((series, year, "ok", len(data)))
            except NoMatchingDataError:
                log.append((series, year, "no data", 0))
            except Exception as err: # for network errors
                log.append((series, year, f"error: {err}", None))
            print(series, year, log[-1][2])

    log_df = pd.DataFrame(log, columns=["series", "year", "status", "rows"])
    log_df.to_csv(out_dir / "download_log.csv", index=False)
    return log_df

# prevents immediate downloading of data when function is pulled into a notebook (until it is specifically run as a script)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    parser.add_argument("--series", nargs="*", choices=list(SERIES))
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_zone(args.zone, args.series, args.years, args.force))