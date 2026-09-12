"""
Render a static mock-up of the dashboard for inclusion in the report.
We draw the layout with matplotlib rather than capture a real screenshot
because the environment does not have Streamlit / a display server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import LightGBMPredictor, warning_level   # noqa: E402
from data_loader import OP_SETTING_COLS, SENSOR_COLS, load_subset


FIG = ROOT / "figures"


def render(unit_id: int = 90):
    predictor = LightGBMPredictor(subset="FD001")
    _, test, y_true = load_subset(ROOT / "data" / "CMAPSSData", "FD001")
    eng = test[test["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)
    pred = predictor.predict(eng[["unit", "cycle", *OP_SETTING_COLS, *SENSOR_COLS]])
    true_rul = float(y_true[unit_id - 1])
    lvl, action = warning_level(pred)

    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.5, 0.7, 3])

    # Header bar
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor("#2c3e50")
    ax_head.text(0.01, 0.5, " Turbofan engine RUL dashboard",
                 transform=ax_head.transAxes,
                 fontsize=14, fontweight="bold", color="white",
                 va="center")
    ax_head.text(0.99, 0.5,
                 f"Subset: FD001 | Engine: {unit_id:>3d} | Model: LightGBM-FD001 ",
                 transform=ax_head.transAxes, color="white", ha="right",
                 va="center", fontsize=9)
    ax_head.set_xticks([]); ax_head.set_yticks([])
    for s in ax_head.spines.values():
        s.set_visible(False)

    # Metric tiles
    color_map = {"OK": "#27ae60", "WATCH": "#f1c40f",
                 "PLAN_MAINTENANCE": "#e67e22", "CRITICAL": "#c0392b"}
    metrics = [
        ("Cycles observed", f"{int(eng['cycle'].max())}", "#34495e"),
        ("Predicted RUL", f"{pred:5.1f} cyc", "#3a6fa5"),
        ("True RUL", f"{true_rul:5.1f} cyc", "#7f8c8d"),
        ("Warning level", lvl, color_map[lvl]),
    ]
    for i, (label, value, c) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i * 3 // 2:(i * 3 // 2) + 2])
        # Actually use whole row width / 4
    # Re-do the metric tile layout: 4 boxes side-by-side spanning full width
    for spc in [ax for ax in fig.axes if ax not in (ax_head,)]:
        spc.remove()

    for i, (label, value, c) in enumerate(metrics):
        left = 0.02 + i * 0.245
        width = 0.225
        ax = fig.add_axes([left, 0.55, width, 0.12])
        ax.set_facecolor("#ecf0f1")
        ax.text(0.5, 0.7, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#2c3e50")
        ax.text(0.5, 0.30, value, transform=ax.transAxes,
                ha="center", va="center", fontsize=15, color=c,
                fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Recommended action banner
    ax_action = fig.add_axes([0.02, 0.43, 0.96, 0.06])
    ax_action.set_facecolor(color_map[lvl])
    ax_action.text(0.01, 0.5, "Recommended action:",
                   transform=ax_action.transAxes,
                   color="white", fontsize=10, fontweight="bold",
                   va="center")
    ax_action.text(0.22, 0.5, action,
                   transform=ax_action.transAxes,
                   color="white", fontsize=9.5, va="center")
    ax_action.set_xticks([]); ax_action.set_yticks([])
    for s in ax_action.spines.values():
        s.set_visible(False)

    # Sensor traces grid - bottom half of figure
    plot_sensors = ["sensor_2", "sensor_3", "sensor_4",
                    "sensor_7", "sensor_11", "sensor_15"]
    # Bottom region: y from 0.04 to 0.38, two rows of three plots
    row_h = 0.13
    row_gap = 0.06
    for i, s in enumerate(plot_sensors):
        r, c = divmod(i, 3)
        left = 0.06 + c * 0.31
        bottom = 0.04 + (1 - r) * (row_h + row_gap)
        ax = fig.add_axes([left, bottom, 0.26, row_h])
        ax.plot(eng["cycle"], eng[s], color="#3a6fa5", linewidth=1.0)
        ax.set_title(s, fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("")
    fig.savefig(FIG / "fig_dashboard_mockup.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mock-up for engine {unit_id}: predicted RUL = {pred:.1f}, "
          f"true RUL = {true_rul:.1f}, level = {lvl}")


if __name__ == "__main__":
    # Engine 90 is near end-of-life (true RUL = 28) so the dashboard nicely
    # shows a PLAN_MAINTENANCE recommendation.
    render(unit_id=90)
