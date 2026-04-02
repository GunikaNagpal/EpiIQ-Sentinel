import streamlit as st
import pandas as pd
from pathlib import Path

# ==============================
# LOAD DATA
# ==============================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "risk.csv"

st.set_page_config(page_title="EpiIQ Sentinel", layout="wide")

st.title("🧠 EpiIQ Sentinel")
st.subheader("AI-Powered Epidemic Intelligence System")

# ==============================
# CHECK DATA
# ==============================

if not DATA_PATH.exists():
    st.error("❌ Run pipeline first: python src/pipeline.py")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# ==============================
# COUNTRY SELECT
# ==============================

country = st.selectbox("🌍 Select Country", sorted(df["iso3"].unique()))
country_df = df[df["iso3"] == country]

# ==============================
# METRICS
# ==============================

st.subheader("📊 Epidemiological Signals")

col1, col2, col3 = st.columns(3)

col1.metric("Rt", round(country_df["Rt"].iloc[-1], 2))
col2.metric("Growth Rate", round(country_df["growth_rate"].iloc[-1], 2))
col3.metric("Risk Score", round(country_df["risk_score"].iloc[-1], 2))

# ==============================
# CASE TREND
# ==============================

st.subheader("📈 Case Trend")
st.line_chart(country_df.set_index("date")["new_cases"])

# ==============================
# RISK TREND
# ==============================

st.subheader("⚠️ Risk Trend")
st.line_chart(country_df.set_index("date")["risk_score"])

# ==============================
# FORECAST (OPTIONAL SIMPLE)
# ==============================

st.subheader("🔮 Forecast (Simple Projection)")

last_cases = country_df["new_cases"].iloc[-1]
growth = country_df["growth_rate"].iloc[-1]

forecast_values = []
current = last_cases

for i in range(14):
    current = current * (1 + growth)
    forecast_values.append(current)

forecast_df = pd.DataFrame({
    "day": range(1, 15),
    "predicted_cases": forecast_values
})

st.line_chart(forecast_df.set_index("day"))

# ==============================
# GLOBAL RISK TABLE
# ==============================

st.subheader("🌍 Top Risk Countries")

latest = df.sort_values("date").groupby("iso3").tail(1)
top = latest.sort_values("risk_score", ascending=False).head(10)

st.dataframe(top[["iso3", "risk_score", "Rt", "growth_rate"]])

# ==============================
# ALERT SYSTEM
# ==============================

st.subheader("🚨 Alert")

if country_df["Rt"].iloc[-1] > 1 and country_df["growth_rate"].iloc[-1] > 0.05:
    st.error("⚠️ Outbreak expanding")
elif country_df["Rt"].iloc[-1] < 1:
    st.success("✅ Under control")
else:
    st.warning("⚠️ Monitor situation")