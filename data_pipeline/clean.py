# turns multiple raw ENTSO-E files from one zone into one combined hourly table with a UTC time index
# to run from repo root: uv run python -m data_pipeline.clean --zone DE_LU

import argparse
import pandas as pd
from data_pipeline import schema
from zone_settings import ROOT, get_zone

RAW_DIR = ROOT / "data" / "raw" / "entsoe"

# combine all yearly raw files of one series into one with a sorted UTC index
def read_series(code, series):
    files = sorted((RAW_DIR / code).glob(f"{series}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files for {series} in {RAW_DIR/code}")
    df = pd.concat(pd.read_csv(f, index_col=0) for f in files)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

# create hourly values for all periods (to allow later merging of 15-minute periods)
def hourly_index(zone):
    return pd.date_range(zone["start"], zone["end"], freq="h", inclusive="left").tz_convert("UTC")

# average values to hours (hourly values stay the same, 15-minute values get averaged across the hour)
def to_hourly(df, index):
    return df.resample("h").mean().reindex(index)

# clean price data
def clean_price(code, index):
    raw = read_series(code, "price")
    currencies = set(raw["Currency"].dropna())
    if currencies != {"EUR"}: # check to ensure consistent currencies
        raise NotImplementedError(f"{code}: non-EUR prices found {currencies}; add FX conversion first")
    return to_hourly(raw[["Price"]], index).rename(columns={"Price": schema.PRICE}) # rename to standard column name

# clean load forecast data
def clean_load_forecast(code, index):
    raw = read_series(code, "load_forecast")
    return to_hourly(raw[["Forecasted Load"]], index).rename(columns={"Forecasted Load": schema.LOAD_FORECAST}) # rename to standard column name

# clean wind and solar forecast data
def clean_wind_solar(code, index):
    raw = read_series(code, "wind_solar_forecast")
    if "Wind Offshore" not in raw.columns: # for zones without offshore wind
        raw["Wind Offshore"] = 0.0
    hourly = to_hourly(raw[["Solar", "Wind Onshore", "Wind Offshore"]], index)
    return pd.DataFrame({
        schema.WIND_FORECAST: hourly["Wind Onshore"] + hourly["Wind Offshore"], # combine wind onshore and wind offshore
        schema.SOLAR_FORECAST: hourly["Solar"]
    }, index=index)

# combine annual capacity
def annual_capacity(code, tz):
    raw = read_series(code, "capacity")
    if "Wind Offshore" not in raw.columns:
        raw["Wind Offshore"] = 0.0
    years = raw.index.tz_convert(tz).year
    annual = pd.DataFrame({
        schema.WIND_CAPACITY: (raw["Wind Onshore"] + raw["Wind Offshore"].fillna(0)).values,
        schema.SOLAR_CAPACITY: raw["Solar"].values
    }, index=years)
    return annual[~annual.index.duplicated(keep="last")].sort_index()

# check annual capacity data for missing years and exact repeats to feed into interpolation
def check_capacity(annual, code):
    missing = sorted(set(range(annual.index.min(), annual.index.max() + 1)) - set(annual.index))
    if missing:
        raise ValueError(f"{code}: capacity missing for years {missing}")
    repeats = set()
    for col in annual.columns:
        change = annual[col].pct_change()
        for year, c in change.dropna().items():
            if c == 0:
                repeats.add(year)
                print(f"WARNING {code} {col}: {year} identical to {year - 1}, treated as missing (extrapolated)")
    return repeats

# set up final clean (interpolated capacity values) - monthly steps between year to year January values with each hour getting the monthly value
def clean_capacity(code, zone, index):
    tz = zone["timezone"]
    annual = annual_capacity(code, tz)
    repeats = check_capacity(annual, code)
    local = index.tz_convert(tz)
    months = pd.period_range(local.min().strftime("%Y-%m"), local.max().strftime("%Y-%m"), freq="M")
    monthly = {}
    for m in months:
        y = m.year
        if y not in annual.index:
            raise ValueError(f"{code}: no capacity for {y}")
        if y + 1 in annual.index and y + 1 not in repeats:
            step = annual.loc[y + 1] - annual.loc[y] # interpolate to next January
        elif y - 1 in annual.index:
            step = annual.loc[y] - annual.loc[y - 1] # carry previous year's growth
        else:
            raise ValueError(f"{code}: cannot interpolate capacity for {y}")
        monthly[m] = (annual.loc[y] + (m.month -1)/12 * step).round()
    monthly = pd.DataFrame(monthly).T
    hour_month = local.tz_localize(None).to_period("M") # local month of each hour
    return pd.DataFrame(monthly.loc[hour_month].values, index=index, columns=monthly.columns)

# create combined dataset for one zone combining all ENTSO-E variables
def clean_zone(code):
    zone = get_zone(code)
    index = hourly_index(zone)
    index.name = schema.TIMESTAMP
    parts = [
        clean_price(code, index),
        clean_load_forecast(code, index),
        clean_wind_solar(code, index),
        clean_capacity(code, zone, index)
    ]
    return pd.concat(parts, axis=1)

# identify missing data
def missing_report(df, tz):
    return df.isna().groupby(df.index.tz_convert(tz).year).sum()

# prevents script from running directly on import into a notebook and only runs when called
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    args = parser.parse_args()
    out = clean_zone(args.zone)
    print(out.head(3), "\n", out.shape)
    print(missing_report(out, get_zone(args.zone)["timezone"]).to_string())