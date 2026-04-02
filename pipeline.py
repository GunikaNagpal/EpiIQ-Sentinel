import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_jhu():
    confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    deaths = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    )

    def reshape(df, value_name):
        df = df.drop(columns=["Province/State", "Lat", "Long"])
        df = df.groupby("Country/Region").sum()
        df = df.T
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().melt(id_vars="index", var_name="iso3", value_name=value_name)
        df.rename(columns={"index": "date"}, inplace=True)
        return df

    confirmed = reshape(confirmed, "total_cases")
    deaths = reshape(deaths, "total_deaths")

    df = confirmed.merge(deaths, on=["iso3", "date"])
    return df

def compute_features(df):
    df = df.sort_values(["iso3", "date"])

    df["new_cases"] = df.groupby("iso3")["total_cases"].diff().clip(lower=0)
    df["new_deaths"] = df.groupby("iso3")["total_deaths"].diff().clip(lower=0)

    df["cases_smooth"] = df.groupby("iso3")["new_cases"].transform(lambda x: x.rolling(7).mean())

    df["growth_rate"] = df.groupby("iso3")["cases_smooth"].pct_change()

    df["Rt"] = df.groupby("iso3")["cases_smooth"].transform(
        lambda x: (x + 1) / (x.shift(7) + 1)
    )

    df["CFR"] = df["total_deaths"] / (df["total_cases"] + 1)

    df = df.fillna(0)

    df["risk_score"] = (
        0.4 * df["growth_rate"] +
        0.3 * df["Rt"] +
        0.3 * df["CFR"]
    ).clip(lower=0)

    return df

def run_pipeline():
    print("Running pipeline...")
    df = load_jhu()
    df = compute_features(df)

    df.to_csv(DATA_DIR / "risk.csv", index=False)
    df.to_csv(DATA_DIR / "outbreak.csv", index=False)

    print("Pipeline complete. Data saved.")

if __name__ == "__main__":
    run_pipeline()
