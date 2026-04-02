"""
EpiIQ Sentinel — risk.py
Re-exports FEATURES, FEATURE_LABELS, and simulate_scenario
so app.py can do: from risk import FEATURES, FEATURE_LABELS, simulate_scenario
"""

from pipeline import (   # noqa: F401  (re-export)
    FEATURES,
    FEATURE_LABELS,
    simulate_scenario,
    assign_risk_tier,
)
