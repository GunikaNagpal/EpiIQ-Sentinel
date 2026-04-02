"""
EpiIQ Sentinel — Spatial Analysis Module
Moran's I global autocorrelation + LISA cluster detection (HH/LL/HL/LH)
using PySAL (esda + libpysal).

Dependencies: pip install esda libpysal
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"

# ── Approximate lat/lon centroids for ISO3 codes ──────────────────────────────
# (Subset of key countries — extend as needed)
ISO3_COORDS = {
    "USA": (37.09, -95.71), "GBR": (55.38, -3.44),  "IND": (20.59, 78.96),
    "BRA": (-14.24, -51.93),"RUS": (61.52, 105.32),  "CHN": (35.86, 104.19),
    "FRA": (46.23, 2.21),   "DEU": (51.17, 10.45),   "ITA": (41.87, 12.57),
    "ESP": (40.46, -3.75),  "AUS": (-25.27, 133.78), "CAN": (56.13, -106.35),
    "ZAF": (-30.56, 22.94), "NGA": (9.08, 8.68),     "EGY": (26.82, 30.80),
    "PAK": (30.38, 69.35),  "IDN": (-0.79, 113.92),  "MEX": (23.63, -102.55),
    "ARG": (-38.42, -63.62),"TUR": (38.96, 35.24),   "IRN": (32.43, 53.69),
    "KOR": (35.91, 127.77), "JPN": (36.20, 138.25),  "THA": (15.87, 100.99),
    "VNM": (14.06, 108.28), "PHL": (12.88, 121.77),  "MYS": (4.21, 101.97),
    "COL": (4.57, -74.30),  "PER": (-9.19, -75.02),  "CHL": (-35.68, -71.54),
    "SDN": (12.86, 30.22),  "ETH": (9.14, 40.49),    "KEN": (-0.02, 37.91),
    "TZA": (-6.37, 34.89),  "UKR": (48.38, 31.17),   "POL": (51.92, 19.15),
    "NLD": (52.13, 5.29),   "BEL": (50.50, 4.47),    "CHE": (46.82, 8.23),
    "SWE": (60.13, 18.64),  "NOR": (60.47, 8.47),    "DNK": (56.26, 9.50),
    "FIN": (61.92, 25.75),  "PRT": (39.40, -8.22),   "GRC": (39.07, 21.82),
    "ISR": (31.05, 34.85),  "SAU": (23.89, 45.08),   "ARE": (23.42, 53.85),
    "QAT": (25.35, 51.18),  "KWT": (29.31, 47.48),   "IRQ": (33.22, 43.68),
    "SYR": (34.80, 38.99),  "JOR": (30.59, 36.24),   "LBN": (33.85, 35.86),
    "BGD": (23.68, 90.36),  "NPL": (28.39, 84.12),   "LKA": (7.87, 80.77),
    "MMR": (21.91, 95.96),  "KHM": (12.57, 104.99),  "LAO": (19.86, 102.50),
    "SGP": (1.35, 103.82),  "NZL": (-40.90, 174.89), "CIV": (7.54, -5.55),
    "GHA": (7.95, -1.02),   "CMR": (3.85, 11.50),    "SEN": (14.50, -14.45),
    "DZA": (28.03, 1.66),   "MAR": (31.79, -7.09),   "TUN": (33.89, 9.54),
    "LBY": (26.34, 17.23),  "AGO": (-11.20, 17.87),  "MOZ": (-18.67, 35.53),
    "ZMB": (-13.13, 27.85), "ZWE": (-19.02, 29.15),  "BOL": (-16.29, -63.59),
    "PRY": (-23.44, -58.44),"URY": (-32.52, -55.77), "VEN": (6.42, -66.59),
    "ECU": (-1.83, -78.18), "GTM": (15.78, -90.23),  "HND": (15.20, -86.24),
    "CRI": (9.75, -83.75),  "PAN": (8.54, -80.78),   "CUB": (21.52, -79.37),
    "DOM": (18.74, -70.16), "HTI": (18.97, -72.29),  "JAM": (18.11, -77.30),
    "UZB": (41.38, 63.97),  "KAZ": (48.02, 66.92),   "GEO": (42.32, 43.36),
    "AZE": (40.14, 47.58),  "ARM": (40.07, 45.04),   "BGR": (42.73, 25.49),
    "ROU": (45.94, 24.97),  "HUN": (47.16, 19.50),   "CZE": (49.82, 15.47),
    "SVK": (48.67, 19.70),  "HRV": (45.10, 15.20),   "SRB": (44.02, 21.01),
    "MKD": (41.61, 21.75),  "ALB": (41.15, 20.17),
}


def build_spatial_weights(iso3_list: list, k: int = 5) -> "libpysal.weights.W":
    """
    Build KNN spatial weights matrix from lat/lon centroids.
    Falls back to queen contiguity if coordinates are missing.
    """
    import libpysal.weights as lpw

    coords = [(ISO3_COORDS[iso][1], ISO3_COORDS[iso][0])  # (lon, lat)
              for iso in iso3_list if iso in ISO3_COORDS]
    valid_isos = [iso for iso in iso3_list if iso in ISO3_COORDS]

    if len(valid_isos) < k + 1:
        raise ValueError(f"Not enough countries with known coordinates (need >{k}, got {len(valid_isos)})")

    coords_arr = np.array(coords)
    w = lpw.KNN.from_array(coords_arr, k=k)
    w.id_order = valid_isos
    w.transform = "R"   # Row-standardise
    return w, valid_isos


def compute_morans_i(latest_df: pd.DataFrame, variable: str = "risk_score") -> dict:
    """
    Compute Global Moran's I for spatial autocorrelation of risk scores.
    Returns: {'I': float, 'EI': float, 'z_score': float, 'p_value': float}
    """
    try:
        from esda.moran import Moran
    except ImportError:
        raise ImportError("Run: pip install esda libpysal")

    df = latest_df.dropna(subset=["iso3", variable])
    w, valid_isos = build_spatial_weights(df["iso3"].tolist())

    y = df.set_index("iso3").loc[valid_isos, variable].values

    mi = Moran(y, w)
    return {
        "I":           round(mi.I, 4),
        "EI":          round(mi.EI, 4),
        "z_score":     round(mi.z_norm, 4),
        "p_value":     round(mi.p_norm, 4),
        "n_countries": len(valid_isos),
    }


def compute_lisa(latest_df: pd.DataFrame, variable: str = "risk_score",
                  significance: float = 0.05) -> pd.DataFrame:
    """
    Compute Local Moran's I (LISA) and classify into HH/LL/HL/LH/NS clusters.
    """
    try:
        from esda.moran import Moran_Local
    except ImportError:
        raise ImportError("Run: pip install esda libpysal")

    df = latest_df.dropna(subset=["iso3", variable]).copy()
    w, valid_isos = build_spatial_weights(df["iso3"].tolist())

    df_valid = df.set_index("iso3").loc[valid_isos].reset_index()
    y = df_valid[variable].values

    lm = Moran_Local(y, w)

    # LISA quadrants: 1=HH, 2=LH, 3=LL, 4=HL
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    clusters = []
    for i, iso3 in enumerate(valid_isos):
        sig = lm.p_sim[i] < significance
        cluster = quad_map.get(lm.q[i], "NS") if sig else "NS"
        clusters.append({"iso3": iso3, "lisa_cluster": cluster,
                          "lisa_I": lm.Is[i], "lisa_p": lm.p_sim[i]})

    lisa_df = pd.DataFrame(clusters)
    # Merge back with original columns
    result = df_valid.merge(lisa_df, on="iso3", how="left")
    result["lisa_cluster"] = result["lisa_cluster"].fillna("NS")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Spatial risk score (neighbourhood influence)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spatial_risk(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Add spatial_risk column: weighted average of k-nearest neighbours' risk scores.
    Computed on the latest snapshot, then merged back.
    """
    try:
        import libpysal.weights as lpw
    except ImportError:
        df["spatial_risk"] = 0.0
        return df

    latest = df.sort_values("date").groupby("iso3").last().reset_index()
    valid_isos = [iso for iso in latest["iso3"].tolist() if iso in ISO3_COORDS]

    if len(valid_isos) < k + 1:
        df["spatial_risk"] = 0.0
        return df

    w, valid_isos = build_spatial_weights(valid_isos, k=k)
    score_map = latest.set_index("iso3")["risk_score"].to_dict()
    y = np.array([score_map.get(iso, 0) for iso in valid_isos])

    # Spatial lag: W * y
    spatial_lag = np.array([
        sum(w.weights[i][j] * y[w.neighbors[i][j]]
            for j in range(len(w.neighbors[i])))
        for i in range(len(valid_isos))
    ])

    spatial_df = pd.DataFrame({"iso3": valid_isos, "spatial_risk": spatial_lag})
    df = df.merge(spatial_df, on="iso3", how="left")
    df["spatial_risk"] = df["spatial_risk"].fillna(0.0)
    return df


if __name__ == "__main__":
    # Quick sanity check
    risk_path = PROC_DIR / "risk.csv"
    if risk_path.exists():
        df = pd.read_csv(risk_path, parse_dates=["date"])
        latest = df.sort_values("date").groupby("iso3").last().reset_index()
        print("Running Moran's I...")
        result = compute_morans_i(latest)
        print(f"Moran's I = {result['I']:.4f}, p = {result['p_value']:.4f}")
        print("Running LISA...")
        lisa = compute_lisa(latest)
        print(lisa[["iso3", "lisa_cluster"]].value_counts("lisa_cluster"))
    else:
        print("Run pipeline.py first.")
