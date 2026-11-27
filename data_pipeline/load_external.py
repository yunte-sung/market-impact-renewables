# turns the external source files (gas, carbon price) into clean daily series, then maps them onto a zone's hours
# run from repo root: uv run python -m data_pipeline.load_external --zone DE_LU

'''Choices documented here:
1. Gas price = front-month TTF futures, not day-ahead gas
- Interpretation: the control measures the expected near-term gas price level and not tomorrow's actual gas cost.
- Advantage: front-month futures react much less to next-day weather than day-ahead so the control is less likely
to absorb part of the wind/solar effect
- Limitations: short gas price spikes (e.g. cold snaps) are only partly captured, and the series jumps slightly
around month ends when it switches to the next contract

2. Price lag of 2 days (PRICE_LAG_DAYS = 2):
- The day-ahead power auction for delivery day D closes around noon on D-1, after the day-ahead wind/solar forecasts
for D are published. A gas or carbon close from D-1 is set after both, so it could react to the forecast (e.g. high
wind -> less gas burn -> lower gas price) and absorb part of the effect we are trying to estimate
- With a 2-day lag, day D gets the last price dated D-2 or earlier, which was set before the forecast for D existed.
Cost: a slightly staler control, mainly visible in 2022.
- Robustness check: rerun with PRICE_LAG_DAYS = 1. On 2019-2025 data the two gas series differ by EUR 1.46/MWh on
average (correlation 0.995)

3. Carbon price = EEX primary auction prices: contract T3PA only, successful auctions only, same-day auctions averaged.
Prices are carried forward between auctions; the longest gap is the Phase 3 to 4 break ending 29 January 2021.

4. Spain uses the MIBGAS-ES Index (daily, PVB), a spot price, because it is the price the gas cap formula uses and no 
free month-ahead history exists; this is a known difference from the TTF front-month used for the other zones.

5. Iberian gas cap: gas price = min(lagged MIBGAS, monthly reference price) from 15 Jun 2022 to 31 Dec 2023; plants 
were compensated for the difference, so this is the gas cost that drove power prices.'''

import argparse
import re

import pandas as pd

from data_pipeline import schema
from data_pipeline.clean import hourly_index
from zone_settings import get_external, get_zone

PRICE_LAG_DAYS = 2
FIRST_EUA_YEAR = 2017
MAX_FILL_WARN_DAYS = 7 # warn when a price is carried forward longer than this

# Iberian gas cap: monthly reference gas price (EUR/MWh) from RDL 10/2022 and RDL 3/2023 (extension)
IBERIAN_CAP_START = pd.Timestamp("2022-06-15")
IBERIAN_CAP_END = pd.Timestamp("2023-12-31") # inclusive
IBERIAN_CAP = {**{f"2022-{m:02d}": 40.0 for m in range(6, 13)},
    "2023-01": 45.0, "2023-02": 50.0, "2023-03": 55.0, "2023-04": 56.1, "2023-05": 57.2, "2023-06": 58.3,
    "2023-07": 59.4, "2023-08": 60.6, "2023-09": 61.7, "2023-10": 62.8, "2023-11": 63.9, "2023-12": 65.0}

# import and sort gas TTF file
def read_ttf(source):
    df = pd.read_csv(source, encoding = "utf-8-sig")
    dates = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return pd.Series(df["Price"].astype(float).values, index=dates).sort_index()

# import and clean each individual carbon prices file for further processing
def _read_eua_file(path):
    raw = pd.read_excel(path, header=None)
    header_row = raw.index[raw.iloc[:,1].astype(str).str.strip().eq("Date")][0] # find header row
    df = pd.read_excel(path, header=header_row)
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
    df = df[df["Contract"] == "T3PA"] # filter to only T3PA contracts
    if "Status" in df.columns: 
        df = df[df["Status"] == "successful"] # filter for successful auctions
    return df[["Date", "Auction Price €/tCO2"]]

# create final carbon prices series from all files
def read_eua(source_dir):
    frames = []
    for path in sorted(source_dir.glob("*-data.xls*")):
        year = int(re.search(r"report-(\d{4})-data", path.name).group(1)) # extract year from the file name
        if year >= FIRST_EUA_YEAR:
            frames.append(_read_eua_file(path))
    df = pd.concat(frames)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df.groupby("Date")["Auction Price €/tCO2"].mean().sort_index()

# read one MIBGAS annual file: daily MIBGAS-ES Index (PVB); sheet and column names change between years
def _read_mibgas_file(path):
    year = int(re.search(r"MIBGAS_Data_(\d{4})", path.name).group(1))
    sheet = "Indices" if year <= 2022 else "MIBGAS Indexes"
    df = pd.read_excel(path, sheet_name=sheet)
    if "Area" in df.columns:
        df = df[df["Area"] == "ES"] # 2021-2022 files also contain Portuguese rows
    col = next(c for c in df.columns if re.match(r"MIBGAS(-ES)? Index", c)) # "MIBGAS-ES Index" or "MIBGAS Index"
    s = pd.Series(df[col].astype(float).values, index=pd.to_datetime(df["Delivery day"]))
    return s[s.index.year == year]

# combine all MIBGAS annual files into one daily series
def read_mibgas(source_dir):
    s = pd.concat(_read_mibgas_file(p) for p in sorted(source_dir.glob("MIBGAS_Data_*.xlsx")))
    if s.index.duplicated().any():
        raise ValueError("MIBGAS: duplicate delivery days")
    return s.sort_index()

READERS = {"gas_ttf": read_ttf, "carbon_price": read_eua, "gas_mibgas": read_mibgas}

# create clean daily series from each external source (i.e. gas and carbon)
def daily_series(key, rebuild=False):
    ext = get_external(key)
    if ext["file"].exists() and not rebuild:
        return pd.read_csv(ext["file"], index_col=0, parse_dates=True).iloc[:,0]
    s = READERS[key](ext["source"]) # apply read function depending on the source
    s.index.name = "date"
    s.name = key
    ext["file"].parent.mkdir(parents = True, exist_ok = True)
    s.to_csv(ext["file"])
    return s

# give every hour in the regular data series the latest gas / carbon price dated at least PRICE_LAG_DAYS
# before its local date (just creates an index series converting daily data to hourly data)
def to_hours(daily, index, tz, name):
    daily = daily.sort_index()
    daily.index = daily.index + pd.Timedelta(days=PRICE_LAG_DAYS) # price of day D applies from d + lag
    hour_dates = index.tz_convert(tz).tz_localize(None).normalize() # local date of each hour
    needed = pd.DatetimeIndex(hour_dates.unique())
    if needed.min() < daily.index.min():
        raise ValueError(f"{name}: no price before {needed.min().date()}; source starts too late")
    combined = daily.reindex(daily.index.union(needed))
    source_date = pd.Series(combined.index, index=combined.index).where(combined.notna()).ffill()
    filled = combined.ffill().reindex(needed) # carry last price forward
    age = (needed - pd.DatetimeIndex(source_date.reindex(needed))).days
    if age.max() > MAX_FILL_WARN_DAYS: # warning if extending beyond warning boundary period
        worst = needed[age.argmax()].date()
        print(f"WARNING {name}: price carried forward {age.max()} days (up to {worst})")
    return pd.Series(filled.reindex(hour_dates).values, index=index, name=name)

# during the Iberian cap, gas plants bid as if gas cost at most the reference price of the delivery month
# applied after the lag: gas = min(lagged MIBGAS, cap of the hour's local month)
def apply_iberian_cap(gas, tz):
    local = gas.index.tz_convert(tz).tz_localize(None)
    cap = pd.Series(local.to_period("M").astype(str).map(IBERIAN_CAP), index=gas.index, dtype=float)
    active = (local >= IBERIAN_CAP_START) & (local < IBERIAN_CAP_END + pd.Timedelta(days=1))
    capped = gas.where(gas <= cap, cap)
    return gas.where(~active, capped)

# hourly series for gas and carbon created with previous function added to hourly ENTSO-E data for each zone
def external_for_zone(code, rebuild=False):
    zone = get_zone(code)
    index = hourly_index(zone)
    index.name = schema.TIMESTAMP
    gas = to_hours(daily_series(zone["gas_series"], rebuild), index, zone["timezone"], schema.GAS_PRICE)
    if zone["gas_cap"] == "iberian":
        gas = apply_iberian_cap(gas, zone["timezone"])
    co2 = to_hours(daily_series(zone["co2_series"], rebuild), index, zone["timezone"], schema.CO2_PRICE)
    return pd.concat([gas, co2], axis=1)

# same as other files - prevents from running immediately upon loading into notebook
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    parser.add_argument("--rebuild", action="store_true", help="rebuild daily files from the sources")
    args = parser.parse_args()
    out = external_for_zone(args.zone, args.rebuild)
    print(out.head(3), "\n", out.shape)
    print(out.isna().sum().to_string())