import pandas as pd
import numpy as np
from pathlib import Path

from spatial import load_mobility, merge_mobility, compute_mobility_adjusted_risk, detect_hotspots

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)

# OWID is a Track C recommended source — provides population, vaccination, and testing.
OWID_URL  = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OWID_COLS = [
    "location", "iso_code", "date", "population",
    "people_vaccinated_per_hundred",
    "people_fully_vaccinated_per_hundred",
    "positive_rate",
    "new_tests_smoothed_per_thousand",
]


# ── OWID loaders ──────────────────────────────────────────────────────────────

def load_owid() -> pd.DataFrame:
    """
    Load OWID COVID dataset. Strips continental/global aggregates
    (iso_codes starting with OWID_). Returns country-level time series.
    """
    print("Loading OWID data (population, vaccination, testing)...")
    df = pd.read_csv(OWID_URL, usecols=OWID_COLS, parse_dates=["date"], low_memory=False)
    df = df[~df["iso_code"].str.startswith("OWID_", na=True)]
    return df.rename(columns={"location": "country"})


def extract_population(owid_df: pd.DataFrame) -> pd.Series:
    """Latest non-null population per country. Returns Series indexed by country name."""
    return (
        owid_df.dropna(subset=["population"])
               .sort_values("date")
               .groupby("country")["population"]
               .last()
    )


def extract_vaccination(owid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vaccination coverage time series (% of population).
    Forward-filled within country — vaccination figures are reported sporadically.
    """
    vax = owid_df[["country", "date",
                   "people_vaccinated_per_hundred",
                   "people_fully_vaccinated_per_hundred"]].copy()
    vax = vax.rename(columns={
        "people_vaccinated_per_hundred":       "vacc_one_dose",
        "people_fully_vaccinated_per_hundred": "vacc_fully",
    }).sort_values(["country", "date"])
    vax[["vacc_one_dose", "vacc_fully"]] = (
        vax.groupby("country")[["vacc_one_dose", "vacc_fully"]]
           .transform(lambda x: x.ffill())
    )
    return vax


def extract_testing(owid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test positivity rate and testing volume per thousand.
    WHO threshold: positivity > 5% indicates inadequate testing.
    """
    return (
        owid_df[["country", "date", "positive_rate", "new_tests_smoothed_per_thousand"]]
               .rename(columns={
                   "positive_rate":                   "test_positivity_rate",
                   "new_tests_smoothed_per_thousand": "tests_per_thousand",
               })
    )


# ── JHU loader ────────────────────────────────────────────────────────────────

def load_jhu() -> pd.DataFrame:
    confirmed = pd.read_csv(JHU_BASE + "time_series_covid19_confirmed_global.csv")
    deaths    = pd.read_csv(JHU_BASE + "time_series_covid19_deaths_global.csv")

    def reshape(raw, value_name):
        raw = raw.drop(columns=["Province/State", "Lat", "Long"])
        raw = raw.groupby("Country/Region").sum()
        raw = raw.T
        raw.index = pd.to_datetime(raw.index)
        raw = raw.reset_index().melt(id_vars="index", var_name="country", value_name=value_name)
        raw.rename(columns={"index": "date"}, inplace=True)
        return raw

    return reshape(confirmed, "total_cases").merge(
        reshape(deaths, "total_deaths"), on=["country", "date"]
    )


# ── Epidemiological functions ─────────────────────────────────────────────────

def compute_rt(cases_smooth: pd.Series, serial_interval: int = 7) -> pd.Series:
    """Ratio-based Rt. SI=7 days. Clipped [0, 5]."""
    return ((cases_smooth + 0.5) / (cases_smooth.shift(serial_interval) + 0.5)).clip(0, 5)


def compute_doubling_time(growth_rate: pd.Series) -> pd.Series:
    """Exact doubling time: ln(2)/ln(1+g). NaN when growth <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(growth_rate > 0, np.log(2) / np.log(1 + growth_rate), np.nan)
    return pd.Series(dt, index=growth_rate.index)


def compute_rt_trend(rt_series: pd.Series, window: int = 7) -> pd.Series:
    """
    Rolling linear slope of Rt over `window` days via OLS.
    Positive = accelerating. Negative = decelerating/stabilising.
    """
    def slope(w):
        if len(w) < 3:
            return np.nan
        t   = np.arange(len(w))
        cov = np.cov(t, w.values)
        return cov[0, 1] / (np.var(t) + 1e-9)

    return rt_series.rolling(window, min_periods=4).apply(slope, raw=False)


def detect_wave_breaks(cases_smooth: pd.Series, threshold: float = 25.0) -> pd.Series:
    """
    CUSUM structural break detection.
    Accumulates positive z-score deviations from a 28-day rolling baseline.
    Flags the onset of a new regime: wave start, variant surge, policy shock.
    Returns 1 at each transition onset, 0 otherwise.
    """
    roll_mean = cases_smooth.rolling(28, min_periods=14).mean()
    roll_std  = cases_smooth.rolling(28, min_periods=14).std().replace(0, np.nan)
    z         = ((cases_smooth - roll_mean) / roll_std).fillna(0)

    cusum = np.zeros(len(z))
    for i in range(1, len(z)):
        cusum[i] = max(0.0, cusum[i - 1] + z.iloc[i])

    cusum_s = pd.Series(cusum, index=cases_smooth.index)
    return (cusum_s > threshold).astype(int).diff().clip(lower=0).fillna(0).astype(int)


def death_case_lag_corr(g: pd.DataFrame) -> pd.Series:
    """
    Rolling 21-day Pearson r between deaths[t] and cases[t-14].
    Validates the 14-day lag assumption per country.
    Drops when surveillance quality changes or variant behaviour shifts.
    """
    g = g.copy()
    g["cases_lag14"] = g["cases_smooth"].shift(14)
    corr = g[["deaths_smooth", "cases_lag14"]].rolling(21, min_periods=14).corr()
    try:
        return corr.unstack()["deaths_smooth"]["cases_lag14"]
    except Exception:
        return pd.Series(np.nan, index=g.index)


# ── Main feature computation ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, pop_map: pd.Series) -> pd.DataFrame:
    df  = df.sort_values(["country", "date"]).copy()
    grp = df.groupby("country")

    df["new_cases"]  = grp["total_cases"].diff().clip(lower=0)
    df["new_deaths"] = grp["total_deaths"].diff().clip(lower=0)

    df["cases_smooth"]  = grp["new_cases"].transform(lambda x: x.rolling(7, min_periods=3).mean())
    df["deaths_smooth"] = grp["new_deaths"].transform(lambda x: x.rolling(7, min_periods=3).mean())

    # Incidence per 100k — WHO standard; >50 = high transmission
    df["population"]         = df["country"].map(pop_map)
    df["incidence_per_100k"] = np.where(
        df["population"].notna() & (df["population"] > 0),
        df["cases_smooth"] / df["population"] * 100_000,
        np.nan,
    )

    # Week-over-week growth (day-over-day is too noisy for surveillance data)
    df["growth_rate"] = grp["cases_smooth"].transform(
        lambda x: x.pct_change(periods=7).clip(-0.9, 5)
    )

    df["Rt"]       = grp["cases_smooth"].transform(compute_rt)
    df["Rt_trend"] = grp["Rt"].transform(compute_rt_trend)

    # Transmission momentum = Rt × growth — captures speed + intensity simultaneously
    df["transmission_momentum"] = (df["growth_rate"].clip(0) * df["Rt"]).clip(0, 10)

    # CFR with 14-day lag — unlagged CFR systematically underestimates during growth phases
    df["CFR"] = (
        grp.apply(lambda g: g["total_deaths"] / (g["total_cases"].shift(14) + 1))
           .reset_index(level=0, drop=True)
           .clip(0, 1)
    )

    # Healthcare pressure = cases × CFR, normalised to country's own historical peak
    df["healthcare_pressure"]      = df["cases_smooth"] * df["CFR"]
    pressure_peak                  = grp["healthcare_pressure"].transform("max")
    df["healthcare_pressure_norm"] = (df["healthcare_pressure"] / (pressure_peak + 1e-9)).clip(0, 1)

    # Death acceleration: week-over-week growth in death series
    # Rising death acceleration is an early warning ~14 days before case trends reverse
    df["death_acceleration"] = grp["deaths_smooth"].transform(
        lambda x: x.pct_change(periods=7).clip(-0.9, 5)
    )

    # Death-case lag correlation — validates the 14-day lag assumption
    df["death_case_lag_corr"] = (
        grp.apply(death_case_lag_corr)
           .reset_index(level=0, drop=True)
           .clip(-1, 1)
    )

    df["doubling_time"]      = grp["growth_rate"].transform(compute_doubling_time)
    df["wave_break"]         = grp["cases_smooth"].transform(detect_wave_breaks)
    df["relative_incidence"] = df["cases_smooth"] / (grp["cases_smooth"].transform("max") + 1)

    df = df.fillna(0)

    # Composite risk score (0–1)
    df["risk_score"] = (
        0.30 * ((df["Rt"] - 1) / 2).clip(0, 1)            +
        0.20 * (df["growth_rate"] / 2).clip(0, 1)          +
        0.20 * df["CFR"].clip(0, 1)                        +
        0.10 * df["relative_incidence"].clip(0, 1)         +
        0.10 * (df["transmission_momentum"] / 3).clip(0, 1) +
        0.10 * df["healthcare_pressure_norm"].clip(0, 1)
    ).clip(0, 1)

    return df


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(include_mobility: bool = True):
    print("Loading JHU data...")
    df = load_jhu()

    # OWID: population, vaccination, testing
    pop_map = pd.Series(dtype=float)
    vax_df  = None
    test_df = None

    try:
        owid_df = load_owid()
        pop_map = extract_population(owid_df)
        vax_df  = extract_vaccination(owid_df)
        test_df = extract_testing(owid_df)
        print(f"OWID loaded: {len(pop_map)} countries with population data.")
    except Exception as e:
        print(f"OWID load failed ({e}). Incidence/100k, vaccination and testing unavailable.")

    print("Computing epidemiological features...")
    df = compute_features(df, pop_map)

    # Merge vaccination coverage
    if vax_df is not None:
        df = df.merge(vax_df, on=["country", "date"], how="left")
        for col in ["vacc_one_dose", "vacc_fully"]:
            df[col] = df.groupby("country")[col].transform(lambda x: x.ffill().fillna(0))
    else:
        df["vacc_one_dose"] = np.nan
        df["vacc_fully"]    = np.nan

    # Merge test positivity
    if test_df is not None:
        df = df.merge(test_df, on=["country", "date"], how="left")
        df["test_positivity_rate"] = df["test_positivity_rate"].clip(0, 1)
        df["tests_per_thousand"]   = df["tests_per_thousand"].clip(lower=0)
    else:
        df["test_positivity_rate"] = np.nan
        df["tests_per_thousand"]   = np.nan

    # Mobility integration
    if include_mobility:
        try:
            mob = load_mobility()
            df  = merge_mobility(df, mob)
            df  = compute_mobility_adjusted_risk(df)
            mob.to_csv(DATA_DIR / "mobility.csv", index=False)
            print("Mobility data integrated.")
        except Exception as e:
            print(f"Mobility load failed ({e}). Continuing without mobility.")
            df["contact_index_lag14"]    = np.nan
            df["mobility_adjusted_risk"] = df["risk_score"]
    else:
        df["contact_index_lag14"]    = np.nan
        df["mobility_adjusted_risk"] = df["risk_score"]

    df.to_csv(DATA_DIR / "risk.csv",     index=False)
    df.to_csv(DATA_DIR / "outbreak.csv", index=False)

    hotspots = detect_hotspots(df)
    hotspots.to_csv(DATA_DIR / "hotspots.csv", index=False)

    print(f"Pipeline complete. {df['country'].nunique()} countries processed.")


if __name__ == "__main__":
    run_pipeline()
