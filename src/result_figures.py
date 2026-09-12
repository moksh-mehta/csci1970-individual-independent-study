"""
Final result visualisations:

  - fig_predictions.pdf : scatter of true vs predicted RUL on each subset
  - fig_per_engine.pdf  : per-engine error bars for the test set of FD001
  - fig_training_curves.pdf : LSTM learning curves on FD001-FD004
  - fig_model_comparison.pdf : bar chart of RMSE for every (model, subset)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

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


def fig_predictions():
    """LSTM scatter: predicted vs true RUL for FD001/FD002/FD003/FD004."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, sub in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
        try:
            d = pd.read_csv(RES / f"lstm_predictions_{sub}.csv")
        except FileNotFoundError:
            ax.set_visible(False); continue
        # Use *raw* RUL for the scatter (more informative than capped)
        y_true = d["true_RUL_raw"].values
        y_pred = d["pred_RUL"].values
        ax.scatter(y_true, y_pred, s=18, alpha=0.55, color="#3a6fa5",
                   edgecolor="white", linewidth=0.5)
        lim = max(y_true.max(), y_pred.max()) + 5
        ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(125, color="#d65a31", linestyle=":", linewidth=0.7,
                   label="cap = 125")
        rmse = float(np.sqrt(((y_pred - np.minimum(y_true, 125)) ** 2).mean()))
        ax.set_title(f"{sub}  (RMSE={rmse:.2f})")
        ax.set_xlabel("true RUL (raw, cycles)")
        ax.set_ylabel("predicted RUL (cycles)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.suptitle("LSTM predictions vs ground truth (test set, last cycle per engine)",
                 fontsize=11)
    fig.savefig(FIG / "fig_predictions.pdf")
    plt.close(fig)


def fig_per_engine():
    """Sorted per-engine error for FD001."""
    d = pd.read_csv(RES / "lstm_predictions_FD001.csv")
    d["err"] = d["pred_RUL"] - np.minimum(d["true_RUL_raw"], 125)
    d = d.sort_values("true_RUL_raw").reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#d65a31" if e > 0 else "#3a6fa5" for e in d["err"]]
    ax.bar(x, d["err"], color=colors, width=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("test engine (sorted by true RUL)")
    ax.set_ylabel("prediction error = pred - true")
    ax.set_title("Per-engine LSTM prediction error on FD001 "
                 "(red = late prediction, blue = early)")
    fig.savefig(FIG / "fig_per_engine.pdf")
    plt.close(fig)


def fig_training_curves():
    """LSTM training and validation curves across subsets."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, sub in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
        try:
            h = pd.read_csv(RES / f"lstm_history_{sub}.csv")
        except FileNotFoundError:
            ax.set_visible(False); continue
        ax.plot(h["epoch"], np.sqrt(h["train_loss"]), label="train RMSE",
                color="#3a6fa5", linewidth=1.4)
        ax.plot(h["epoch"], h["val_rmse"], label="val RMSE",
                color="#d65a31", linewidth=1.4)
        ax.set_title(sub)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("RMSE (cycles)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("LSTM training and validation RMSE", fontsize=11)
    fig.savefig(FIG / "fig_training_curves.pdf")
    plt.close(fig)


def fig_model_comparison():
    """Bar chart: RMSE for every (model, subset) pair."""
    d = pd.read_csv(RES / "all_results.csv")
    pivot = d.pivot(index="model", columns="subset", values="RMSE")
    order = ["Mean", "Ridge", "RandomForest", "XGBoost", "LightGBM", "LSTM"]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    cols = ["FD001", "FD002", "FD003", "FD004"]
    x = np.arange(len(pivot.index))
    width = 0.2
    colors = ["#2c5f8a", "#3a8a6f", "#d6a531", "#d65a31"]
    for i, c in enumerate(cols):
        vals = pivot[c].values
        ax.bar(x + (i - 1.5) * width, vals, width, label=c, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("Test RMSE (cycles)")
    ax.set_title("Model comparison on the four CMAPSS subsets")
    ax.legend(frameon=False, ncol=4)
    ax.set_ylim(0, 50)
    # Annotate best per subset
    fig.savefig(FIG / "fig_model_comparison.pdf")
    plt.close(fig)


def fig_feature_importance():
    """LightGBM feature importance for FD001."""
    fi = pd.read_csv(RES / "feature_importance_lightgbm_FD001.csv")
    top = fi.head(20)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top["feature"].values[::-1], top["importance"].values[::-1],
            color="#3a6fa5")
    ax.set_xlabel("importance (LightGBM)")
    ax.set_title("Top-20 features by importance, LightGBM on FD001")
    fig.savefig(FIG / "fig_feature_importance.pdf")
    plt.close(fig)


def main():
    print("Generating result figures...")
    for fn in (fig_predictions, fig_per_engine, fig_training_curves,
               fig_model_comparison, fig_feature_importance):
        print(f"  - {fn.__name__}")
        fn()
    print("Done.")


if __name__ == "__main__":
    main()
