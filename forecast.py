"""
EpiIQ Sentinel — Forecasting Module
Prophet 14-day forecasts + SHAP values + walk-forward validation metrics.

Run: python src/forecast.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROPHET FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def prophet_forecast(df: pd.DataFrame, iso3: str, horizon: int = 14) -> pd.DataFrame:
    """
    Fit a Prophet model for a single country and return 14-day forecasts
    with 95% prediction intervals.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: pip install prophet")

    grp = (df[df["iso3"] == iso3]
             .sort_values("date")[["date", "new_cases"]]
             .rename(columns={"date": "ds", "new_cases": "y"})
             .dropna())

    grp["y"] = grp["y"].clip(lower=0)

    if len(grp) < 30:
        return pd.DataFrame()

    m = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15,
        interval_width=0.95,
    )
    m.fit(grp)

    future  = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
    out = out.rename(columns={
        "ds": "forecast_date",
        "yhat": "forecast",
        "yhat_lower": "lower_95",
        "yhat_upper": "upper_95",
    })
    out["iso3"] = iso3
    out["forecast"]  = out["forecast"].clip(lower=0)
    out["lower_95"]  = out["lower_95"].clip(lower=0)
    out["upper_95"]  = out["upper_95"].clip(lower=0)
    return out[["iso3", "forecast_date", "forecast", "lower_95", "upper_95"]]


def run_forecasts(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Run Prophet forecasts for the top-N most populous countries."""
    print(f"🔮 Running Prophet forecasts for up to {top_n} countries...")

    # Prioritise countries with most recent data and enough history
    valid_isos = (df.groupby("iso3")
                    .filter(lambda g: len(g) >= 60)["iso3"]
                    .unique())

    # Sort by latest case count (proxy for importance)
    latest = df.sort_values("date").groupby("iso3").last()
    priority = (latest.loc[latest.index.isin(valid_isos), "new_cases"]
                      .sort_values(ascending=False)
                      .index[:top_n])

    all_forecasts = []
    failed = []
    for i, iso3 in enumerate(priority):
        try:
            fc = prophet_forecast(df, iso3)
            if not fc.empty:
                all_forecasts.append(fc)
                if (i + 1) % 10 == 0:
                    print(f"   {i+1}/{len(priority)} done...")
        except Exception as e:
            failed.append(iso3)

    if failed:
        print(f"   ⚠️ Forecast failed for: {', '.join(failed[:10])}")

    if not all_forecasts:
        return pd.DataFrame()

    result = pd.concat(all_forecasts, ignore_index=True)
    print(f"   ✅ Forecasts complete: {result['iso3'].nunique()} countries")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(df: pd.DataFrame, iso3: str,
                           horizon: int = 14, n_splits: int = 4) -> dict:
    """
    Walk-forward cross-validation for a single country.
    Returns RMSE, MAE, MAPE.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {}

    grp = (df[df["iso3"] == iso3]
             .sort_values("date")[["date", "new_cases"]]
             .rename(columns={"date": "ds", "new_cases": "y"})
             .dropna())
    grp["y"] = grp["y"].clip(lower=0)

    if len(grp) < 60 + horizon:
        return {}

    rmses, maes, mapes = [], [], []
    step = (len(grp) - 60 - horizon) // n_splits

    for k in range(n_splits):
        train_end = 60 + k * step
        train = grp.iloc[:train_end]
        test  = grp.iloc[train_end: train_end + horizon]

        if len(test) < horizon // 2:
            continue

        try:
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.15, interval_width=0.95)
            m.fit(train)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future).tail(horizon)

            y_pred = forecast["yhat"].clip(lower=0).values[:len(test)]
            y_true = test["y"].values

            rmses.append(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            maes.append(np.mean(np.abs(y_true - y_pred)))
            denom = np.where(y_true == 0, 1, y_true)
            mapes.append(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
        except Exception:
            continue

    if not rmses:
        return {}

    return {
        "iso3":  iso3,
        "model": "Prophet",
        "rmse":  np.mean(rmses),
        "mae":   np.mean(maes),
        "mape":  np.mean(mapes),
        "n_splits": len(rmses),
    }


def run_validation(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Run walk-forward validation across top-N countries."""
    print(f"📐 Walk-forward validation for up to {top_n} countries...")
    valid_isos = (df.groupby("iso3")
                    .filter(lambda g: len(g) >= 90)["iso3"]
                    .unique())

    latest = df.sort_values("date").groupby("iso3").last()
    priority = (latest.loc[latest.index.isin(valid_isos), "new_cases"]
                      .sort_values(ascending=False)
                      .index[:top_n])

    results = []
    for iso3 in priority:
        m = walk_forward_validate(df, iso3)
        if m:
            results.append(m)

    if not results:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(results)
    print(f"   ✅ Validated {len(metrics_df)} countries | "
          f"Avg RMSE: {metrics_df['rmse'].mean():.1f} | "
          f"Avg MAPE: {metrics_df['mape'].mean():.1f}%")
    return metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GBM risk classifier and compute SHAP values on the latest snapshot.
    """
    try:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor
        from pipeline import FEATURES
    except ImportError as e:
        print(f"   ⚠️ SHAP/sklearn not available: {e}")
        return pd.DataFrame()

    print("🧠 Computing SHAP values...")
    latest = df.sort_values("date").groupby("iso3").last().reset_index()
    feat_cols = [f for f in FEATURES if f in latest.columns]
    X = latest[feat_cols].fillna(0)
    y = latest["risk_score"].fillna(0)

    if len(X) < 10:
        return pd.DataFrame()

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=feat_cols, index=latest.index)
    shap_df["iso3"] = latest["iso3"].values
    print(f"   ✅ SHAP values computed for {len(shap_df)} countries")
    return shap_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("🔮 EpiIQ Sentinel — Forecast Module")
    print("="*60 + "\n")

    risk_path = PROC_DIR / "risk.csv"
    if not risk_path.exists():
        print("❌ risk.csv not found. Run pipeline.py first.")
        return

    df = pd.read_csv(risk_path, parse_dates=["date"])

    # Forecasts
    forecasts = run_forecasts(df, top_n=50)
    if not forecasts.empty:
        forecasts.to_csv(PROC_DIR / "forecasts.csv", index=False)
        print(f"   💾 forecasts.csv saved")

    # Validation metrics
    metrics = run_validation(df, top_n=30)
    if not metrics.empty:
        metrics.to_csv(PROC_DIR / "model_metrics.csv", index=False)
        print(f"   💾 model_metrics.csv saved")

    # SHAP values
    shap_df = compute_shap_values(df)
    if not shap_df.empty:
        shap_df.to_csv(PROC_DIR / "shap_values.csv", index=False)
        print(f"   💾 shap_values.csv saved")

    print("\n✅ Forecast module complete.\n")


if __name__ == "__main__":
    main()
