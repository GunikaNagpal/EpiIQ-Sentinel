import pandas as pd
import numpy as np

MOBILITY_URL = (
    "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
)

CONTACT_MOBILITY_COLS = [
    "retail_and_recreation_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
]

RESIDENTIAL_COL = "residential_percent_change_from_baseline"


def load_mobility() -> pd.DataFrame:
    """
    Load Google Mobility at country level (national aggregate only).
    Computes a contact_index: high retail/transit/workplace → more contacts → transmission risk up.
    Residential mobility is inversely weighted (people staying home = less contact).
    """
    print("Loading Google Mobility data...")
    mob = pd.read_csv(
        MOBILITY_URL,
        usecols=[
            "country_region", "sub_region_1", "date",
            *CONTACT_MOBILITY_COLS, RESIDENTIAL_COL,
        ],
        parse_dates=["date"],
        low_memory=False,
    )

    mob = mob[mob["sub_region_1"].isna()].drop(columns=["sub_region_1"])
    mob = mob.rename(columns={"country_region": "country"})

    # Contact intensity: mean of out-of-home mobility minus residential buffer
    mob["contact_index"] = (
        mob[CONTACT_MOBILITY_COLS].mean(axis=1)
        - 0.5 * mob[RESIDENTIAL_COL]
    )

    mob["contact_index_smooth"] = (
        mob.groupby("country")["contact_index"]
        .transform(lambda x: x.rolling(7, min_periods=3).mean())
    )

    return mob[["country", "date", "contact_index", "contact_index_smooth",
                *CONTACT_MOBILITY_COLS, RESIDENTIAL_COL]]


def merge_mobility(epi_df: pd.DataFrame, mob_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge epi features with mobility lagged 14 days.
    14-day lag captures incubation period (~5d) + reporting delay (~9d).
    """
    mob_lagged = mob_df[["country", "date", "contact_index_smooth"]].copy()
    mob_lagged["date"] = mob_lagged["date"] + pd.Timedelta(days=14)
    mob_lagged = mob_lagged.rename(columns={"contact_index_smooth": "contact_index_lag14"})

    return epi_df.merge(mob_lagged, on=["country", "date"], how="left")


def compute_mobility_rt_correlation(merged_df: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """
    Per-country Pearson r between lagged contact index and Rt.
    A positive correlation confirms mobility as a leading transmission indicator for that country.
    """
    results = []
    for country, grp in merged_df.groupby("country"):
        valid = grp.dropna(subset=["contact_index_lag14", "Rt"])
        if len(valid) < min_obs:
            continue
        corr = valid["contact_index_lag14"].corr(valid["Rt"])
        results.append({
            "country": country,
            "mobility_rt_correlation": round(corr, 3),
            "n_obs": len(valid),
        })
    return pd.DataFrame(results).sort_values("mobility_rt_correlation", ascending=False)


def detect_hotspots(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Hotspot detection: countries where Rt AND growth rate are simultaneously elevated.
    Hotspot score = weighted combination of Rt signal + global growth percentile rank + risk percentile.
    Binary flag requires Rt > 1 AND growth in top quartile globally.
    """
    latest = df.sort_values("date").groupby("country").tail(1).copy()

    latest["growth_pct_rank"] = latest["growth_rate"].rank(pct=True)
    latest["risk_pct_rank"] = latest["risk_score"].rank(pct=True)

    rt_signal = ((latest["Rt"] - 1) / 2).clip(0, 1)
    latest["hotspot_score"] = (
        0.4 * rt_signal +
        0.35 * latest["growth_pct_rank"] +
        0.25 * latest["risk_pct_rank"]
    ).clip(0, 1)

    latest["is_hotspot"] = (
        (latest["Rt"] > 1.0) & (latest["growth_pct_rank"] > 0.75)
    )

    cols = ["country", "hotspot_score", "is_hotspot", "Rt", "growth_rate",
            "risk_score", "doubling_time", "cases_smooth", "new_cases"]
    available = [c for c in cols if c in latest.columns]

    return (
        latest[available]
        .sort_values("hotspot_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_mobility_adjusted_risk(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mobility-adjusted risk score.
    Contact index > 0 (above-baseline mobility) amplifies risk up to 1.5x.
    Contact index < 0 (lockdown-like) dampens risk down to 0.5x.
    Multiplier normalized so 0% mobility change = no adjustment.
    """
    if "contact_index_lag14" not in merged_df.columns:
        merged_df["mobility_adjusted_risk"] = merged_df["risk_score"]
        return merged_df

    ci = merged_df["contact_index_lag14"].fillna(0)
    multiplier = (1 + ci / 200).clip(0.5, 1.5)
    merged_df["mobility_adjusted_risk"] = (merged_df["risk_score"] * multiplier).clip(0, 1)
    return merged_df
