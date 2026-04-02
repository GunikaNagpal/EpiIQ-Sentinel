import pandas as pd

def simple_forecast(df, country):
    data = df[df["iso3"] == country].copy()

    last_cases = data["cases_smooth"].iloc[-1]
    growth = data["growth_rate"].iloc[-1]

    if growth < 0:
        growth = 0.01

    predictions = []
    current = last_cases

    for i in range(14):
        current = current * (1 + growth)
        predictions.append(current)

    forecast_df = pd.DataFrame({
        "day": range(1, 15),
        "predicted_cases": predictions
    })

    return forecast_df
