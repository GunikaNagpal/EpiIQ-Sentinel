import numpy as np
import pandas as pd


def _damped_growth_rate(recent_rates: pd.Series) -> float:
    """
    Compute a damped growth estimate using exponentially weighted mean.
    Applies a damping factor so long forecasts don't explode unrealistically.
    """
    weights = np.exp(np.linspace(-1, 0, len(recent_rates)))
    weighted = np.average(recent_rates.clip(-0.5, 2), weights=weights)
    return float(np.clip(weighted, -0.3, 1.5))


def forecast_cases(
    df: pd.DataFrame,
    country: str,
    horizon: int = 14,
    damping: float = 0.92,
) -> pd.DataFrame:
    """
    14-day damped exponential forecast with uncertainty bands.

    Damping factor < 1 pulls growth rate toward zero over time,
    reflecting natural epidemic deceleration (intervention, immunity).

    Returns:
        DataFrame with columns: date, predicted, lower_80, upper_80
    """
    data = df[df["country"] == country].dropna(subset=["cases_smooth"]).copy()

    if len(data) < 14:
        return pd.DataFrame()

    last_date = data["date"].max()
    last_cases = data["cases_smooth"].iloc[-1]

    recent_growth = data["growth_rate"].tail(14)
    base_growth = _damped_growth_rate(recent_growth)

    # Estimate uncertainty from recent volatility (std of growth)
    growth_std = float(recent_growth.std())

    dates, predicted, lower_80, upper_80 = [], [], [], []

    current = last_cases
    g = base_growth

    for i in range(1, horizon + 1):
        g = g * damping  # dampen each step
        current = max(current * (1 + g), 0)

        # 80% interval: ±1.28σ propagated over i steps
        margin = current * (growth_std * np.sqrt(i)) * 1.28

        dates.append(last_date + pd.Timedelta(days=i))
        predicted.append(current)
        lower_80.append(max(current - margin, 0))
        upper_80.append(current + margin)

    return pd.DataFrame({
        "date": dates,
        "predicted": predicted,
        "lower_80": lower_80,
        "upper_80": upper_80,
    })
