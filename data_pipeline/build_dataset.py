# builds the final hourly dataset for one zone and saves it to data/processed/{zone}.csv
# run from repo root: uv run python -m data_pipeline.build_dataset --zone DE_LU

import argparse

import pandas as pd

from data_pipeline import schema
from data_pipeline.clean import clean_zone, missing_report
from data_pipeline.features import add_features
from data_pipeline.load_external import external_for_zone
from zone_settings import ROOT, get_zone

PROCESSED_DIR = ROOT / "data" / "processed"

# check date series to ensure that it is sorted, that there are no duplicates and no missing hours
def check_index(df, code):
    idx = df.index
    if idx.has_duplicates:
        raise ValueError(f"{code}: {idx.duplicated().sum()} duplicate timestamps")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{code}: timestamps are not sorted")
    steps = idx.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise ValueError(f"{code}: timestamps not exactly one hour apart: {steps}")

# build combined clean dataset for a zone
def build_zone(code, rebuild_external=False):
    zone = get_zone(code)
    extras = zone["extra_controls"]
    if extras:
        raise NotImplementedError(f"{code}: extra controls {extras} not implemented yet")

    # combine all parts on the same hourly UTC index
    entsoe = clean_zone(code)
    external = external_for_zone(code, rebuild_external)
    if not entsoe.index.equals(external.index):
        raise ValueError(f"{code}: ENTSO-E and external data have different time indices")
    df = add_features(pd.concat([entsoe, external], axis=1), code)
    check_index(df, code)

    # drop hours where a required input is missing 
    required = schema.zone_required_cols(extras)
    incomplete = df[required].isna().any(axis=1)
    print("Missing values before dropping (per local year):")
    print(missing_report(df[required], zone["timezone"]).to_string())
    print(f"Dropping {incomplete.sum()} of {len(df)} hours ({incomplete.mean():.2%})")
    df = df[~incomplete]

    # check columns use the standard names and order and that there is nothing missing
    df = df[schema.zone_extra_cols(extras)]
    schema.check_columns(df, extras)
    if df.isna().any().any():
        raise ValueError(f"{code}: missing values left after dropping: {df.isna().sum()[df.isna().sum()>0].to_dict()}")
    df.index.name = schema.TIMESTAMP
    return df

# save cleaned dataset to data folder
def save_zone(df, code):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{code}.csv"
    df.to_csv(path)
    return path

# same as others - prevents from running upon immediate loading into a notebook
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    parser.add_argument("--rebuild-external", action="store_true", help="rebuild gas/carbon daily files from sources")
    args = parser.parse_args()
    df = build_zone(args.zone, args.rebuild_external)
    path = save_zone(df, args.zone)
    print(f"Saved {len(df)} hours x {df.shape[1]} columns to {path.relative_to(ROOT)}")
    print(f"From {df.index.min()} to {df.index.max()} (UTC)")