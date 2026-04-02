import pandas as pd

def compute_spatial_risk(df):
    latest = df.sort_values("date").groupby("iso3").tail(1)

    global_mean = latest["cases_smooth"].mean()

    latest["spatial_risk"] = latest["cases_smooth"] / (global_mean + 1)

    return latest[["iso3", "spatial_risk"]]
