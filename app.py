import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EpiIQ Sentinel", layout="wide")

st.title("🧠 EpiIQ Sentinel")
st.subheader("AI-Powered Epidemic Intelligence System")

# =========================
# LOAD DATA (ROBUST)
# =========================
try:
    df = pd.read_csv("data/processed/risk.csv")
except:
    try:
        df = pd.read_csv("risk.csv")
    except:
        st.error("❌ Data not found. Upload risk.csv properly.")
        st.stop()

# =========================
# FIX DATE (IMPORTANT)
# =========================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Controls")

country = st.sidebar.selectbox(
    "Select Country",
    sorted(df["country"].unique())
)

forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 14)

cdf = df[df["country"] == country]

# =========================
# METRICS
# =========================
st.subheader("📊 Epidemiological Signals")

col1, col2, col3 = st.columns(3)

col1.metric("Rt", round(cdf["Rt"].iloc[-1], 2))
col2.metric("Growth", round(cdf["growth_rate"].iloc[-1], 2))
col3.metric("Risk", round(cdf["risk_score"].iloc[-1], 2))

# =========================
# CASE TREND
# =========================
st.subheader("📈 Case Trend")
st.plotly_chart(px.line(cdf, x="date", y="cases_smooth"), use_container_width=True)

# =========================
# RISK TREND
# =========================
st.subheader("⚠️ Risk Trend")
st.plotly_chart(px.line(cdf, x="date", y="risk_score"), use_container_width=True)

# =========================
# RT TREND
# =========================
st.subheader("📉 Effective Rt")

fig_rt = go.Figure()
fig_rt.add_trace(go.Scatter(x=cdf["date"], y=cdf["Rt"], name="Rt"))

fig_rt.update_layout(
    yaxis=dict(range=[0, min(float(cdf["Rt"].max()) + 0.5, 5)])
)

st.plotly_chart(fig_rt, use_container_width=True)

# =========================
# FORECAST (ARIMA-LIKE)
# =========================
st.subheader("🔮 Forecast")

series = cdf["cases_smooth"].dropna()

# Simple ARIMA-like projection
if len(series) > 10:
    growth = np.mean(series.pct_change().dropna())
else:
    growth = 0.05

current = series.iloc[-1] if len(series) > 0 else 0

forecast_vals = []
for _ in range(forecast_days):
    current *= (1 + growth)
    forecast_vals.append(current)

forecast_df = pd.DataFrame({
    "day": range(1, forecast_days + 1),
    "cases": forecast_vals
})

st.plotly_chart(px.line(forecast_df, x="day", y="cases"), use_container_width=True)

# =========================
# GLOBAL MAP (SPATIAL)
# =========================
st.subheader("🌍 Global Risk Map")

latest = df.sort_values("date").groupby("country").tail(1)

fig_map = px.choropleth(
    latest,
    locations="country",
    locationmode="country names",
    color="risk_score",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# SPATIAL CLUSTER VIEW
# =========================
st.subheader("🧭 Spatial Risk Clusters")

fig_phase = px.scatter(
    latest,
    x="growth_rate",
    y="Rt",
    size="risk_score",
    color="risk_score",
    hover_name="country"
)

fig_phase.add_hline(y=1)
fig_phase.add_vline(x=0)

st.plotly_chart(fig_phase, use_container_width=True)

# =========================
# TOP COUNTRIES
# =========================
st.subheader("🌐 Top Risk Countries")

top = latest.sort_values("risk_score", ascending=False).head(10)
st.dataframe(top[["country", "risk_score", "Rt", "growth_rate"]])

# =========================
# OUTBREAK ALERT
# =========================
st.subheader("🚨 Outbreak Status")

rt = cdf["Rt"].iloc[-1]
growth = cdf["growth_rate"].iloc[-1]

if rt > 1.2 and growth > 0.1:
    st.error("High Risk: Rapid spread")
elif rt > 1:
    st.warning("Moderate Risk: Growing")
else:
    st.success("Low Risk: Controlled")

# =========================
# SUMMARY
# =========================
st.subheader("🧾 Summary")

st.write(f"""
Rt: {rt:.2f}  
Growth Rate: {growth:.2f}  
CFR: {cdf["CFR"].iloc[-1]:.3f}

This indicates the outbreak is {'expanding' if rt > 1 else 'controlled'}.
""")

# =========================
# ANIMATION (FIXED)
# =========================
st.subheader("🎞️ Outbreak Spread Over Time")

df_anim = df.copy()
df_anim["date"] = pd.to_datetime(df_anim["date"], errors="coerce")
df_anim = df_anim.dropna(subset=["date"])

df_anim["date_str"] = df_anim["date"].dt.strftime("%Y-%m-%d")

fig_anim = px.scatter(
    df_anim,
    x="growth_rate",
    y="Rt",
    animation_frame="date_str",
    size="risk_score",
    color="risk_score",
    hover_name="country"
)

st.plotly_chart(fig_anim, use_container_width=True)
