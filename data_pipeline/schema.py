# Standard column names for every processed zone dataset

# Dataset index
TIMESTAMP = "timestamp_utc" # start of the hour, UTC

# Market and system data (ENTSO-E) - all converted to 1 hour periods
PRICE = "price" # day ahead price in EUR per MWh (converted from local currency where needed)
LOAD_FORECAST = "load_forecast" # day-ahead load forecast in MW
WIND_FORECAST = "wind_forecast" # onshore + offshore wind forecast in MW
SOLAR_FORECAST = "solar_forecast" # solar forecast in MW
WIND_CAPACITY = "wind_capacity" # installed wind capacity in MW (annual value)
SOLAR_CAPACITY = "solar_capacity" # installed solar capacity in MW (annual value)

# Fuel and carbon (non-ENTSO-E) - daily values mapped to hours
GAS_PRICE = "gas_price" # EUR per MWh, zone's gas hub price or effective (capped) price where the zone has a gas cap in zones.yaml
CO2_PRICE = "co2_price" # EU carbon price in EUR per tonne

# Derived
WIND_PENETRATION = "wind_penetration" # wind_forecast / load_forecast * 100
SOLAR_PENETRATION = "solar_penetration" # solar_forecast / load_forecast * 100

# Calendar (local time of the zone)
HOUR = "hour" # 0-23
DAY_OF_WEEK = "day_of_week" # 0 = Monday ... 6 = Sunday
MONTH = "month" # 1-12
YEAR = "year"
DAYLIGHT_HOURS = "daylight_hours" # sunrise to sunset hours

# Core columns: every zone has these, in this order (excluding timestamp index)
COLUMNS = [
    PRICE,
    LOAD_FORECAST,
    WIND_FORECAST, SOLAR_FORECAST,
    WIND_CAPACITY, SOLAR_CAPACITY,
    GAS_PRICE, CO2_PRICE,
    WIND_PENETRATION, SOLAR_PENETRATION,
    HOUR, DAY_OF_WEEK, MONTH, YEAR, DAYLIGHT_HOURS
]

# Core inputs that must be present for an hour to be kept (calendar and penetration calculated from index and these)
REQUIRED = [
    PRICE, LOAD_FORECAST, WIND_FORECAST, SOLAR_FORECAST,
    WIND_CAPACITY, SOLAR_CAPACITY, GAS_PRICE, CO2_PRICE
]

# Optional controls depending on zones, as listed in zones.yaml
COAL_PRICE = "coal_price" # API2 coal in EUR per tonne (converted from USD) (mainly for Poland)
NUCLEAR_UNAVAILABLE = "nuclear_unavailable" # unavailable nuclear capacity in MW (mainly for France)
HYDRO_RESERVOIR = "hydro_reservoir" # reservoir filling in MWh (weekly, mapped to hours) (mainly for Sweden / Norway)
NEIGHBOUR_WIND = "neighbour_wind_forecast" # in MW, summed over neighbouring zones with data
NEIGHBOUR_SOLAR = "neighbour_solar_forecast" # in MW, summed over neighbouring zones with data
NEIGHBOUR_LOAD = "neighbour_load_forecast" # in MW, summed over neighbouring zones with data

OPTIONAL = [
    COAL_PRICE, NUCLEAR_UNAVAILABLE, HYDRO_RESERVOIR,
    NEIGHBOUR_WIND, NEIGHBOUR_SOLAR, NEIGHBOUR_LOAD
]

# catch typos when typing in extra columns for a zone
def _check_extras(extras):
    unknown = [c for c in extras if c not in OPTIONAL]
    if unknown:
        raise ValueError(f"Unknown optional columns: {unknown}")

# call extra columns for a zone
def zone_extra_cols(extras=()):
    _check_extras(extras)
    return COLUMNS + [c for c in OPTIONAL if c in extras]

# add extra columns to list of required columns for a zone
def zone_required_cols(extras=()):
    _check_extras(extras)
    return REQUIRED + [c for c in OPTIONAL if c in extras]

# check if dataset for a zone meets all requirements for columns
def check_columns(df, extras=()):
    expected = zone_extra_cols(extras)
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    if missing or unexpected:
        raise ValueError(f"Missing columns: {missing}. Unexpected columns: {unexpected}.")
    if list(df.columns) != expected:
        raise ValueError("Columns are present but not in the standard order.")

