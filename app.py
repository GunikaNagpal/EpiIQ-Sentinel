import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "risk.csv"

st.set_page_config(layout="wide")

st.title("🧠 EpiIQ Sentinel")
st.subheader("AI-Powered Epidemic Intelligence")

if not DATA_PATH.exists():
    st.error("Run pipeline first")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

country = st.selectbox("Select Country", sorted(df["iso3"].unique()))
country_df = df[df["iso3"] == country]

st.subheader("Epidemiological Signals")

col1, col2, col3 = st.columns(3)

col1.metric("Rt", round(country_df["Rt"].iloc[-1], 2))
col2.metric("Growth", round(country_df["growth_rate"].iloc[-1], 2))
col3.metric("Risk", round(country_df["risk_score"].iloc[-1], 2))

st.subheader("Case Trend")
st.line_chart(country_df.set_index("date")["cases_smooth"])

st.subheader("Risk Trend")
st.line_chart(country_df.set_index("date")["risk_score"])

st.subheader("Forecast")

growth = max(country_df["growth_rate"].iloc[-1], 0.01)
last = country_df["cases_smooth"].iloc[-1]

pred = []
current = last

for i in range(14):
    current = current * (1 + growth)
    pred.append(current)

forecast_df = pd.DataFrame({"day": range(1, 15), "cases": pred})
st.line_chart(forecast_df.set_index("day"))

st.subheader("Global Risk Map")

latest = df.sort_values("date").groupby("iso3").tail(1)

fig = px.choropleth(
    latest,
    locations="iso3",
    locationmode="country names",
    color="risk_score",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Risk Countries")

top = latest.sort_values("risk_score", ascending=False).head(10)
st.dataframe(top[["iso3", "risk_score", "Rt", "growth_rate"]])

st.subheader("Outbreak Status")

rt = country_df["Rt"].iloc[-1]
growth = country_df["growth_rate"].iloc[-1]

if rt > 1.2 and growth > 0.1:
    st.error("Rapid spread detected")
elif rt > 1:
    st.warning("Growing transmission")
else:
    st.success("Under control")
