import requests
import pandas as pd
from zone_settings import ROOT

URL = "https://api.energidataservice.dk/dataset/CapacityPerMunicipality"
OUT = ROOT / "data/external/source/energinet/capacity_per_municipality.csv"

def download(start="2018-01-01", end="2026-02-01"):
    r = requests.get(URL, params={"start": start, "end": end, "limit": 100000}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["records"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df["Month"].min(), df["Month"].max(), len(df))

if __name__ == "__main__":
    download()