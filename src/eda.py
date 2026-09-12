"""
Exploratory Data Analysis for the C-MAPSS dataset.

Produces all figures used in the report.  Each figure is saved as a
separate PDF in `figures/` and indexed in figures/MANIFEST.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (
    OP_SETTING_COLS, SENSOR_COLS, load_subset, apply_piecewise_rul
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CMAPSSData"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Publication-quality matplotlib defaults
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})


def summary_table():
    """Compact descriptive table of the four CMAPSS subsets."""
    rows = []
    for sub, conds, faults in [("FD001", 1, 1), ("FD002", 6, 1),
                                ("FD003", 1, 2), ("FD004", 6, 2)]:
        tr, te, y = load_subset(DATA, sub)
        eng_train = tr.groupby("unit")["cycle"].max()
        eng_test = te.groupby("unit")["cycle"].max()
        rows.append({
            "Subset": sub,
            "Conditions": conds,
            "Fault Modes": faults,
            "Train Engines": tr["unit"].nunique(),
            "Test Engines": te["unit"].nunique(),
            "Train Cycles": len(tr),
            "Min Train Life": int(eng_train.min()),
            "Mean Train Life": round(float(eng_train.mean()), 1),
            "Max Train Life": int(eng_train.max()),
            "Mean Test Life": round(float(eng_test.mean()), 1),
            "Mean True RUL": round(float(y.mean()), 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "summary_table.csv", index=False)
    print(df.to_string(index=False))
    return df


def fig_engine_lifetimes():
    """Histogram of run-to-failure lengths across all four subsets."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
    for ax, sub in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
        tr, _, _ = load_subset(DATA, sub)
        lives = tr.groupby("unit")["cycle"].max().values
        ax.hist(lives, bins=22, color="#3a6fa5", edgecolor="white")
        ax.set_title(f"{sub} (n={len(lives)})")
        ax.set_xlabel("cycles to failure")
        ax.axvline(lives.mean(), color="crimson", linestyle="--", linewidth=1,
                   label=f"mean={lives.mean():.0f}")
        ax.legend(frameon=False)
    axes[0].set_ylabel("number of engines")
    fig.savefig(FIG / "fig_engine_lifetimes.pdf")
    plt.close(fig)


def fig_op_conditions():
    """How the three operational settings cluster in FD001 vs FD002."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, sub in zip(axes, ["FD001", "FD002"]):
        tr, _, _ = load_subset(DATA, sub)
        ax.scatter(tr["op_setting_1"], tr["op_setting_2"],
                   s=2, alpha=0.18, c="#2c5f8a")
        ax.set_title(f"{sub} -- operational settings")
        ax.set_xlabel("op_setting_1 (altitude proxy)")
        ax.set_ylabel("op_setting_2 (Mach proxy)")
    fig.savefig(FIG / "fig_op_conditions.pdf")
    plt.close(fig)


def fig_op_clusters():
    """K-means recovers six discrete flight regimes in FD002/FD004."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, sub in zip(axes, ["FD002", "FD004"]):
        tr, _, _ = load_subset(DATA, sub)
        X = tr[OP_SETTING_COLS].values
        km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        sc = ax.scatter(tr["op_setting_1"], tr["op_setting_2"],
                        c=labels, cmap="tab10", s=2, alpha=0.5)
        ax.set_title(f"{sub} -- six recovered regimes")
        ax.set_xlabel("op_setting_1")
        ax.set_ylabel("op_setting_2")
    fig.savefig(FIG / "fig_op_clusters.pdf")
    plt.close(fig)


def fig_sensor_variance():
    """Per-sensor standard deviation across FD001 (highlights constant ones)."""
    tr, _, _ = load_subset(DATA, "FD001")
    stds = tr[SENSOR_COLS].std().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#d65a31" if s < 1e-4 else "#3a6fa5" for s in stds.values]
    ax.barh(stds.index, stds.values, color=colors)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("standard deviation (log scale)")
    ax.set_title("Sensor variance on FD001 (red = effectively constant)")
    fig.savefig(FIG / "fig_sensor_variance.pdf")
    plt.close(fig)


def fig_sensor_signals():
    """Raw sensor traces for one engine vs the averaged degradation pattern."""
    tr, _, _ = load_subset(DATA, "FD001")
    # Pick a representative engine
    engine_id = 24
    sample = tr[tr["unit"] == engine_id]

    # Six informative sensors based on prior CMAPSS literature
    selected = ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                "sensor_11", "sensor_15"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6))
    for ax, s in zip(axes.flat, selected):
        # Faint trajectories from a few engines + the highlighted engine
        for uid in [3, 19, 41, 67, 82]:
            sub_eng = tr[tr["unit"] == uid]
            ax.plot(sub_eng["cycle"], sub_eng[s], color="grey",
                    alpha=0.35, linewidth=0.8)
        ax.plot(sample["cycle"], sample[s], color="#d65a31",
                linewidth=1.6, label=f"engine {engine_id}")
        ax.set_title(s)
        ax.set_xlabel("cycle")
    axes[0, 0].legend(loc="best", frameon=False)
    fig.suptitle("Sensor signals across engine lifetime (FD001)",
                 fontsize=11)
    fig.savefig(FIG / "fig_sensor_signals.pdf")
    plt.close(fig)


def fig_correlation():
    """Correlation of each sensor with RUL across subsets."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, sub in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
        tr, _, _ = load_subset(DATA, sub)
        rul = tr["RUL"]
        corrs = {s: tr[s].corr(rul) for s in SENSOR_COLS}
        corr_s = pd.Series(corrs).sort_values()
        colors = ["#d65a31" if abs(v) < 0.05 else "#3a6fa5"
                  for v in corr_s.values]
        ax.barh(corr_s.index, corr_s.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(sub)
        ax.set_xlabel("Pearson r with RUL")
    axes[0].set_ylabel("sensor")
    fig.savefig(FIG / "fig_correlation.pdf")
    plt.close(fig)


def fig_rul_target():
    """Comparison of linear vs piecewise-linear RUL targets."""
    tr, _, _ = load_subset(DATA, "FD001")
    eng = tr[tr["unit"] == 24].copy()
    eng["RUL_clipped"] = apply_piecewise_rul(eng["RUL"], cap=125)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(eng["cycle"], eng["RUL"], label="linear RUL", color="#888",
            linewidth=1.2)
    ax.plot(eng["cycle"], eng["RUL_clipped"], label="piecewise (cap=125)",
            color="#d65a31", linewidth=1.6)
    ax.axhline(125, color="#d65a31", linestyle=":", linewidth=0.8)
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Linear vs piecewise-linear RUL target (FD001 engine 24)")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "fig_rul_target.pdf")
    plt.close(fig)


def fig_normalisation_effect():
    """Sensor signal before and after condition-based normalisation on FD002.

    The same sensor (s2 -- LPC outlet temperature) is plotted for a single
    engine.  Raw values jump between flight regimes; after we standardise
    *within each regime* the underlying degradation trend becomes visible.
    """
    tr, _, _ = load_subset(DATA, "FD002")

    # Cluster operational settings into six regimes
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(
        tr[OP_SETTING_COLS].values)
    tr = tr.copy()
    tr["regime"] = km.labels_

    # Per-regime standardisation
    scalers = {}
    for r, sub in tr.groupby("regime"):
        scaler = StandardScaler().fit(sub[SENSOR_COLS])
        scalers[r] = scaler
    norm_vals = np.empty(len(tr))
    norm_vals[:] = np.nan
    for r, sub in tr.groupby("regime"):
        norm_vals[sub.index] = scalers[r].transform(
            sub[SENSOR_COLS])[:, SENSOR_COLS.index("sensor_2")]
    tr["sensor_2_norm"] = norm_vals

    eng = tr[tr["unit"] == 5]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    axes[0].plot(eng["cycle"], eng["sensor_2"], color="#3a6fa5", linewidth=0.8)
    axes[0].set_title("sensor 2 (raw)")
    axes[0].set_xlabel("cycle"); axes[0].set_ylabel("raw value")
    axes[1].plot(eng["cycle"], eng["sensor_2_norm"], color="#d65a31",
                 linewidth=0.8)
    # 30-cycle moving average
    ma = eng["sensor_2_norm"].rolling(30, min_periods=1).mean()
    axes[1].plot(eng["cycle"], ma, color="black", linewidth=1.4,
                 label="30-cycle MA")
    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_title("sensor 2 (regime-standardised)")
    axes[1].set_xlabel("cycle"); axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False)
    fig.suptitle("Effect of regime-aware normalisation on FD002 engine 5",
                 fontsize=11)
    fig.savefig(FIG / "fig_normalisation.pdf")
    plt.close(fig)


def main():
    print("== Summary table ==")
    summary_table()
    print("== Generating figures ==")
    for fn in (fig_engine_lifetimes, fig_op_conditions, fig_op_clusters,
               fig_sensor_variance, fig_sensor_signals, fig_correlation,
               fig_rul_target, fig_normalisation_effect):
        print(f"  - {fn.__name__}")
        fn()
    print("Done. Figures saved to", FIG)


if __name__ == "__main__":
    main()
