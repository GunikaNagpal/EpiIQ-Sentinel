import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PATH SETUP
# =========================
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="EpiIQ Sentinel", layout="wide")

# =========================
# CUSTOM STYLING (PREMIUM UI)
# =========================
st.markdown("""
<style>
body { background-color: #0a0e1a; }
.main { background-color: #0a0e1a; }
h1, h2, h3, h4 { color: #e2e8f0; }
[data-testid="stMetric"] {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 EpiIQ Sentinel")
st.subheader("AI-Powered Epidemic Intelligence System")

# =========================
# LOAD DATA (WITH LOADER)
# =========================
try:
    df = pd.read_csv("data/processed/risk.csv")
except:
    try:
        df = pd.read_csv("risk.csv")
    except:
        st.error("Data not found. Please check upload.")
        st.stop()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙️ Controls")

country = st.sidebar.selectbox("🌍 Select Country", sorted(df["country"].unique()))
forecast_days = st.sidebar.slider("📅 Forecast Days", 7, 30, 14)
show_heatmap = st.sidebar.checkbox("Show Heatmap", True)
show_animation = st.sidebar.checkbox("Show Animated Map", True)

cdf = df[df["country"] == country]

if cdf.empty:
    st.error("No data available")
    st.stop()

# =========================
# METRICS
# =========================
st.markdown("### 📊 Key Indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Rt", f"{cdf['Rt'].iloc[-1]:.2f}")
col2.metric("Growth", f"{cdf['growth_rate'].iloc[-1]:.2f}")
col3.metric("Risk", f"{cdf['risk_score'].iloc[-1]:.2f}")

st.markdown("---")

# =========================
# CASE TREND
# =========================
st.subheader("📈 Case Trend")

fig_cases = px.line(
    cdf,
    x="date",
    y="cases_smooth",
    hover_data=["Rt", "growth_rate", "risk_score"]
)

st.plotly_chart(fig_cases, use_container_width=True)

# =========================
# RISK TREND
# =========================
st.subheader("⚠️ Risk Trend")

fig_risk = px.line(
    cdf,
    x="date",
    y="risk_score",
    hover_data=["Rt", "growth_rate"]
)

st.plotly_chart(fig_risk, use_container_width=True)

# =========================
# RT TREND
# =========================
st.subheader("🔥 Effective Rt")

fig_rt = go.Figure()
fig_rt.add_trace(go.Scatter(x=cdf["date"], y=cdf["Rt"], name="Rt"))

fig_rt.update_layout(
    title="Effective Reproduction Number",
    yaxis=dict(range=[0, min(float(cdf["Rt"].max()) + 0.5, 5)])
)

st.plotly_chart(fig_rt, use_container_width=True)

# =========================
# FORECAST
# =========================
st.subheader("🔮 Forecast")

growth = max(cdf["growth_rate"].iloc[-1], 0.01)
current = cdf["cases_smooth"].iloc[-1]

forecast_vals = []
for _ in range(forecast_days):
    current *= (1 + growth)
    forecast_vals.append(current)

forecast_df = pd.DataFrame({
    "day": range(1, forecast_days + 1),
    "cases": forecast_vals
})

fig_forecast = px.line(forecast_df, x="day", y="cases")
st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown("---")

# =========================
# GLOBAL MAP
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

fig_map.update_layout(
    geo=dict(showframe=False, showcoastlines=False)
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# ANIMATED MAP
# =========================
if show_animation:
    st.subheader("🌍 Outbreak Spread Over Time")

    df_anim = df.copy()
    df_anim["date_str"] = df_anim["date"].dt.strftime("%Y-%m-%d")

    fig_anim = px.choropleth(
        df_anim,
        locations="country",
        locationmode="country names",
        color="cases_smooth",
        animation_frame="date_str",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig_anim, use_container_width=True)

# =========================
# TOP COUNTRIES
# =========================
st.subheader("🏆 Top Risk Countries")

top = latest.sort_values("risk_score", ascending=False).head(10)
st.dataframe(top[["country", "risk_score", "Rt", "growth_rate"]])

# =========================
# PHASE SPACE
# =========================
st.subheader("📊 Epidemic Phase Space")

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
# ALERT + AI INSIGHT
# =========================
st.subheader("🚨 Outbreak Status")

rt = cdf["Rt"].iloc[-1]
growth = cdf["growth_rate"].iloc[-1]

if rt > 1.2 and growth > 0.1:
    st.error("High Risk: Rapid spread")
elif rt > 1:
    st.warning("Moderate Risk: Growing")
else:
    st.success("Low Risk: Under control")

st.subheader("🧠 AI Insight")

if rt > 1.2:
    st.info("Transmission is accelerating. Immediate intervention needed.")
elif rt > 1:
    st.info("Transmission is increasing. Monitor closely.")
else:
    st.info("Transmission is under control.")

# =========================
# EXPLANATION
# =========================
st.subheader("📌 Summary")

st.write(f"""
- Rt: {rt:.2f}  
- Growth: {growth:.2f}  
- CFR: {cdf["CFR"].iloc[-1]:.2f}  

This indicates the outbreak is {'expanding' if rt > 1 else 'controlled'}.
""")

# =========================
# RISK BREAKDOWN
# =========================
st.subheader("🧠 Risk Breakdown")

components = {
    "Growth": abs(growth),
    "Rt": rt,
    "CFR": cdf["CFR"].iloc[-1]
}

fig_break = px.bar(
    x=list(components.keys()),
    y=list(components.values())
)

st.plotly_chart(fig_break, use_container_width=True)

# =========================
# HEATMAP
# =========================
if show_heatmap:
    st.subheader("🔥 Global Heatmap")

    pivot = df.pivot_table(
        values="cases_smooth",
        index="country",
        columns="date"
    )

    fig_heat = px.imshow(pivot)
    st.plotly_chart(fig_heat, use_container_width=True)
