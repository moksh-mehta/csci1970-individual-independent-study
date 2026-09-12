"""
Minimal Streamlit dashboard for the CMAPSS RUL predictor.

Run with:
    pip install streamlit
    streamlit run src/dashboard.py

The dashboard lets a maintenance engineer pick one of the held-out test
engines, see a plot of its sensor traces and the model's RUL prediction,
and read the recommended maintenance action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import LightGBMPredictor, warning_level   # noqa: E402
from data_loader import (   # noqa: E402
    OP_SETTING_COLS, SENSOR_COLS, load_subset
)


st.set_page_config(page_title="CMAPSS RUL Dashboard", layout="wide")

st.title("Turbofan engine RUL prediction")
st.write(
    "Pick a held-out test engine to see the model's prediction of its "
    "remaining useful life (in operating cycles) and the maintenance "
    "recommendation."
)

# ---- sidebar ----
subset = st.sidebar.selectbox("CMAPSS subset", ["FD001"])
predictor = LightGBMPredictor(subset=subset)
_, test, y_true = load_subset(ROOT / "data" / "CMAPSSData", subset)
engine_ids = sorted(test["unit"].unique())
unit_id = st.sidebar.selectbox(
    "Engine ID", engine_ids, index=engine_ids.index(1))

# ---- main panel ----
eng = test[test["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)
last_cycle = int(eng["cycle"].max())
true_rul = float(y_true[unit_id - 1])

input_df = eng[["unit", "cycle", *OP_SETTING_COLS, *SENSOR_COLS]]
pred_rul = predictor.predict(input_df)
level, action = warning_level(pred_rul)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cycles observed", last_cycle)
col2.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
col3.metric("True RUL", f"{true_rul:.0f} cycles")
col4.metric("Warning level", level)

st.markdown(f"**Recommended action:** {action}")

st.subheader("Sensor traces")
plot_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                "sensor_11", "sensor_15"]
fig, axes = plt.subplots(2, 3, figsize=(10, 4.5))
for ax, s in zip(axes.flat, plot_sensors):
    ax.plot(eng["cycle"], eng[s], color="#3a6fa5", linewidth=1.0)
    ax.set_title(s); ax.set_xlabel("cycle")
fig.tight_layout()
st.pyplot(fig)
