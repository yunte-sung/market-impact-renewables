# turns multiple raw ENTSO-E files (plus Energinet for DK1 and Fingrid for FI) from one zone into one combined hourly table with a UTC time index
# to run from repo root: uv run python -m data_pipeline.clean --zone DE_LU

import argparse
import pandas as pd
from data_pipeline import schema
from zone_settings import ROOT, get_zone

RAW_DIR = ROOT / "data" / "raw" / "entsoe"
ENERGINET_FILE = ROOT / "data/external/source/energinet/capacity_per_municipality.csv"
# Energinet municipality numbers by price area; anything outside these ranges (e.g. code 1) is excluded
ENERGINET_AREAS = {"DK_1": (410, 860), "DK_2": (101, 400)}
FINGRID_DIR = ROOT / "data" / "external" / "source" / "fingrid"
# Fingrid dataset IDs: 268 = wind capacity used in its wind forecast, 267 = solar capacity used in its solar forecast
FINGRID_DATASETS = {schema.WIND_CAPACITY: "268", schema.SOLAR_CAPACITY: "267"}

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
    raw = raw[(raw.index >= index.min()) & (raw.index < index.max() + pd.Timedelta(hours=1))] # only the zone's period
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
    # offshore not yet reported before its first value -> 0; later gaps stay missing (dropped)
    first = raw["Wind Offshore"].first_valid_index()
    if first is not None:
        raw.loc[raw.index < first, "Wind Offshore"] = 0.0
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

# energinet capacity (specifically for DK1 due to missing data in ENTSO-E)
def energinet_capacity(code):
    lo, hi = ENERGINET_AREAS[code]
    raw = pd.read_csv(ENERGINET_FILE, parse_dates=["Month"])
    raw = raw[raw["MunicipalityNo"].between(lo, hi)]
    jan = raw[raw["Month"].dt.month == 1] # January snapshot, same timing as ENTSO-E
    annual = jan.groupby(jan["Month"].dt.year)[
        ["OnshoreWindCapacity", "OffshoreWindCapacity", "SolarPowerCapacity"]].sum()
    return pd.DataFrame({
        schema.WIND_CAPACITY: annual["OnshoreWindCapacity"] + annual["OffshoreWindCapacity"],
        schema.SOLAR_CAPACITY: annual["SolarPowerCapacity"]
    })

# read one Fingrid capacity file (hourly, UTC; semicolon-separated; value in the third column)
def _read_fingrid(dataset):
    path = next(FINGRID_DIR.glob(f"{dataset}_*.csv"))
    df = pd.read_csv(path, sep=";")
    return pd.Series(df.iloc[:, 2].values, index=pd.to_datetime(df["startTime"], utc=True)).sort_index()

# fingrid capacity (Finland; ENTSO-E solar capacity far too low): value at the first hour of each January
def fingrid_capacity(tz):
    cols = {}
    for col, dataset in FINGRID_DATASETS.items():
        local = _read_fingrid(dataset).tz_convert(tz)
        jan = local[local.index.month == 1]
        cols[col] = jan.groupby(jan.index.year).first()
    return pd.DataFrame(cols)

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
    if zone["capacity_source"] == "energinet":
        annual = energinet_capacity(code)
    elif zone["capacity_source"] == "fingrid":
        annual = fingrid_capacity(tz)
    else:
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

