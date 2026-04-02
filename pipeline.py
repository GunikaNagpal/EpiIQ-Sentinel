"""
EpiIQ Sentinel — Data Pipeline
Run: python src/pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

FEATURES = ["Rt", "growth_rate", "CFR"]

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# DATA SOURCES
# ==============================

OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

JHU_CONFIRMED_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

JHU_DEATHS_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_deaths_global.csv"
)

# ==============================
# FETCH DATA
# ==============================

def fetch_owid():
    print("📥 Fetching OWID data...")
    try:
        df = pd.read_csv(OWID_URL, parse_dates=["date"])
        print("   ✅ OWID loaded")
        return df
    except Exception as e:
        print(f"   ⚠️ OWID failed: {e}")
        return pd.DataFrame()


def fetch_jhu():
    print("📥 Fetching JHU data...")

    conf = pd.read_csv(JHU_CONFIRMED_URL)
    deaths = pd.read_csv(JHU_DEATHS_URL)

    def melt(df, name):
        return (
            df.melt(
                id_vars=["Province/State", "Country/Region", "Lat", "Long"],
                var_name="date",
                value_name=name,
            )
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby(["Country/Region", "date"])[name]
            .sum()
            .reset_index()
        )

    conf = melt(conf, "total_confirmed")
    deaths = melt(deaths, "total_deaths")

    df = conf.merge(deaths, on=["Country/Region", "date"])

    print("   ✅ JHU loaded")
    return df


# ==============================
# PREPROCESS
# ==============================

def preprocess_owid(df):
    df = df.rename(columns={"iso_code": "iso3"})
    df = df.sort_values(["iso3", "date"])
    return df


# ==============================
# FEATURES
# ==============================

def compute_rt(df):
    df["Rt"] = df.groupby("iso3")["new_cases"].transform(
        lambda x: (x + 1) / (x.shift(7) + 1)
    )
    return df


def compute_growth(df):
    df["growth_rate"] = df.groupby("iso3")["new_cases"].pct_change()
    return df


def compute_cfr(df):
    df["CFR"] = df["total_deaths"] / (df["total_cases"] + 1)
    return df


# ==============================
# MAIN PIPELINE
# ==============================

def run_pipeline():

    print("\n==============================")
    print("🧬 Running Pipeline")
    print("==============================\n")

    owid = fetch_owid()
    jhu = fetch_jhu()

    # ==========================
    # USE OWID OR FALLBACK JHU
    # ==========================

    if not owid.empty:
        df = preprocess_owid(owid)

    elif not jhu.empty:
        df = jhu.rename(
            columns={
                "Country/Region": "iso3",
                "total_confirmed": "total_cases",
            }
        )

        # FIX iso3 safely
        if "iso3" not in df.columns:
            df["iso3"] = "UNKNOWN"

        if isinstance(df["iso3"], pd.DataFrame):
            df["iso3"] = df["iso3"].iloc[:, 0]

        df["iso3"] = df["iso3"].astype(str)

        # Compute daily values
        df["new_cases"] = df.groupby("iso3")["total_cases"].diff().clip(lower=0)
        df["new_deaths"] = df.groupby("iso3")["total_deaths"].diff().clip(lower=0)

    else:
        raise Exception("❌ No data available")

    # ==========================
    # FEATURE ENGINEERING
    # ==========================

    print("⚙️ Computing features...")

    df = compute_rt(df)
    df = compute_growth(df)
    df = compute_cfr(df)

    df = df.fillna(0)

    # 🔥 ADD THIS HERE
    df["risk_score"] = (
    0.4 * df["growth_rate"] +
    0.3 * df["Rt"] +
    0.3 * df["CFR"])

    # ==========================
    # SAVE OUTPUT
    # ==========================

    print("💾 Saving...")

    df.to_csv(PROC_DIR / "risk.csv", index=False)
    df.to_csv(PROC_DIR / "outbreak.csv", index=False)

    print("✅ DONE — files created in data/processed/")
    return df


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    run_pipeline()
