# adds calendar, daylight and penetration columns to a zone's hourly table
# run from repo root: uv run python -m data_pipeline.features --zone DE_LU

import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from astral import Observer
from astral.sun import daylight, elevation

from data_pipeline import schema
from zone_settings import get_zone

# extract hour, day of week (Monday = 0), month and year in the zone's local time
def calendar_features(index, tz):
    local = index.tz_convert(tz)
    return pd.DataFrame({
        schema.HOUR: local.hour,
        schema.DAY_OF_WEEK: local.dayofweek,
        schema.MONTH: local.month,
        schema.YEAR: local.year
    }, index=index)

# extract hours from sunrise to sunset on one date; 24 or 0 where the sun never sets or never rises
def _daylight_length(observer, date, tz):
    try:
        start, end = daylight(observer, date, tzinfo=tz)
        return (end-start).total_seconds() / 3600
    except ValueError: # polar day or polar night (e.g. northern Norway)
        noon = dt.datetime.combine(date, dt.time(12), tzinfo=tz)
        return 24.0 if elevation(observer, noon) >0 else 0.0

# calculate daylight hours from daylight length
def daylight_hours(index, zone):
    tz = ZoneInfo(zone["timezone"])
    loc = zone["daylight_location"]
    observer = Observer(latitude=loc["lat"], longitude=loc["lon"])
    local_dates = index.tz_convert(zone["timezone"]).date
    per_date = {d: _daylight_length(observer, d, tz) for d in sorted(set(local_dates))} # once per date
    return pd.Series([per_date[d] for d in local_dates], index=index, name=schema.DAYLIGHT_HOURS)

# calculation wind / solar penetration from forecast data
def penetration(df):
    return pd.DataFrame({
        schema.WIND_PENETRATION: df[schema.WIND_FORECAST] / df[schema.LOAD_FORECAST] * 100,
        schema.SOLAR_PENETRATION: df[schema.SOLAR_FORECAST] / df[schema.LOAD_FORECAST] * 100
    }, index=df.index)

# add all the features into the main dataset
def add_features(df, code):
    zone = get_zone(code)
    return pd.concat([
        df,
        penetration(df),
        calendar_features(df.index, zone["timezone"]),
        daylight_hours(df.index, zone)
    ], axis=1)

# same as other files - prevents from running immediately on initial import into notebook
if __name__ == "__main__":
    from data_pipeline.clean import clean_zone
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    args = parser.parse_args()
    out = add_features(clean_zone(args.zone), args.zone)
    cols = [schema.WIND_PENETRATION, schema.SOLAR_PENETRATION, schema.DAYLIGHT_HOURS]
    print(out[cols].describe().round(2))
    print(out.iloc[:3, -7:])