# reads config/zones.yaml and returns checked settings for one zone
# all scripts and notebooks get zone settings from here

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from data_pipeline import schema

ROOT = Path(__file__).resolve().parent  # goes to repo root when running
CONFIG_PATH = ROOT / "config" / "zones.yaml" # path to get to zones.yaml

ZONE_KEYS = {
    "name", "entsoe_code", "timezone", "start_date", "end_date",
    "daylight_location", "gas_series", "co2_series", "gas_cap",
    "extra_controls", "neighbours", "capacity_source"
}
GAS_CAPS = {None, "iberian"} # notes allowed values for gas_cap (only Iberian gas cap considered)
CAPACITY_SOURCES = {"entsoe", "energinet", "fingrid"}

# load zones.yaml file
def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# create list of all zones in the yaml file
def list_zones():
    return list(_load_config()["zones"])

# check settings for one zone in the file and format for further use

def get_zone(code):
    config = _load_config()
    zones = config["zones"]
    series = config["external_series"]

    if code not in zones:
        raise KeyError(f"Zone {code!r} not in zones.yaml. Available: {list(zones)}")
    zone = dict(zones[code])

    # check if every key present and there are no typos
    missing = ZONE_KEYS - zone.keys()
    unexpected = zone.keys() - ZONE_KEYS
    if missing or unexpected:
        raise ValueError(f"{code}: missing keys {sorted(missing)}, unexpected keys {sorted(unexpected)}")

    # check if values make sense
    ZoneInfo(zone["timezone"]) # raises an error if timezone is not a real timezone
    schema.zone_extra_cols(zone["extra_controls"]) # raises an error if an unknown column has been listed as an extra control
    for key in ("gas_series", "co2_series"):
        if zone[key] not in series:
            raise ValueError(f"{code}: {key} {zone[key]!r} not in external_series")
    if zone["gas_cap"] not in GAS_CAPS:
        raise ValueError(f"{code}: gas cap must be one of {GAS_CAPS}")
    if zone["capacity_source"] not in CAPACITY_SOURCES:
        raise ValueError(f"{code}: capacity_source must be one of {CAPACITY_SOURCES}")

    # conversions from basic entry format in yaml for further use
    tz = zone["timezone"]
    zone["start"] = pd.Timestamp(zone["start_date"], tz = tz) # converting start date to timestamp
    zone["end"] = pd.Timestamp(zone["end_date"], tz = tz) + pd.Timedelta(days=1) # converting end date to timestamp and making end date exclusive
    if zone["start"] >= zone["end"]:
        raise ValueError(f"{code}: start_date must be before end_date")
    zone["gas_file"] = ROOT / series[zone["gas_series"]]["file"] # creating path to clean daily gas file
    zone["co2_file"] = ROOT / series[zone["co2_series"]]["file"] # creating path to clean daily carbon price data
    zone["code"] = code
    return zone

# analog to previous function for external daily series (carbon prices and gas prices, for example)

def get_external(key):
    series = _load_config()["external_series"]
    if key not in series:
        raise KeyError(f"External series {key!r} not in zones.yaml. Available: {list(series)}")
    entry = dict(series[key])
    entry["file"] = ROOT / entry["file"]
    entry["source"] = ROOT / entry["source"]
    entry["key"] = key
    return entry