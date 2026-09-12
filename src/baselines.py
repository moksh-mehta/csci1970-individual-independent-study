"""
Baseline regression models for RUL prediction on the CMAPSS dataset.

Models trained:
  - mean predictor (sanity check)
  - linear regression with rolling features
  - random forest
  - XGBoost
  - LightGBM

All models predict the *piecewise-linear* RUL (capped at 125) following
Heimes (2008) and Ramasso (2014); we then report metrics against:
  - the (capped) RUL ground truth on the per-engine last-cycle prediction
  - the raw RUL ground truth, where we additionally compute the PHM08 score
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (
    add_test_rul, apply_piecewise_rul, load_subset, phm08_score
)
from features import (
    INFORMATIVE_SENSORS, build_tabular_features
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CMAPSSData"
RES = ROOT / "results"
MODELS = ROOT / "models"
RES.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Build train / test arrays
# ---------------------------------------------------------------------------
def get_xy_for_subset(subset: str, cap: int = 125):
    """Tabular features + capped RUL labels for one CMAPSS subset.

    Returns:
        Xtr, ytr             : full training set (one row per cycle)
        Xte_last, yte_last   : one row per test engine (last cycle), RUL = true value capped
    """
    train, test, y_true = load_subset(DATA, subset)
    tr_f, te_f = build_tabular_features(train, test, subset=subset)

    feature_cols = [c for c in tr_f.columns
                    if c not in ("unit", "cycle", "RUL")]

    Xtr = tr_f[feature_cols].values.astype(np.float32)
    ytr = apply_piecewise_rul(tr_f["RUL"].values, cap=cap).astype(np.float32)

    # For test set we keep only the last cycle per engine - this is what the
    # competition asks us to predict.
    last = te_f.groupby("unit").tail(1).copy().reset_index(drop=True)
    last = last.sort_values("unit").reset_index(drop=True)
    Xte_last = last[feature_cols].values.astype(np.float32)
    yte_last = apply_piecewise_rul(y_true, cap=cap).astype(np.float32)
    yte_uncapped = y_true.astype(np.float32)

    return (Xtr, ytr, Xte_last, yte_last, yte_uncapped,
            feature_cols)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit_mean_baseline(Xtr, ytr):
    mean = float(np.mean(ytr))
    return lambda X: np.full(X.shape[0], mean, dtype=np.float32)


def fit_ridge(Xtr, ytr):
    m = Ridge(alpha=1.0, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict


def fit_random_forest(Xtr, ytr, n_estimators=80):
    # Use n_jobs=1 to avoid joblib/loky process spawn issues in containers
    m = RandomForestRegressor(n_estimators=n_estimators, max_depth=12,
                              min_samples_leaf=10, n_jobs=1, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict, m


def fit_xgb(Xtr, ytr):
    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=0, n_jobs=1, tree_method="hist",
        objective="reg:squarederror",
    )
    m.fit(Xtr, ytr)
    return m.predict, m


def fit_lightgbm(Xtr, ytr):
    m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=-1, num_leaves=63,
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=0, n_jobs=1,
        verbosity=-1,
    )
    m.fit(Xtr, ytr)
    return m.predict, m


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(predict_fn, Xte, yte_capped, yte_uncapped):
    """Standard RMSE / MAE on capped labels + PHM08 score on raw labels."""
    yhat = predict_fn(Xte)
    yhat = np.clip(yhat, 0, None)  # negative RUL impossible
    rmse = float(np.sqrt(mean_squared_error(yte_capped, yhat)))
    mae = float(mean_absolute_error(yte_capped, yhat))
    # PHM08 score uses raw RUL on test set's last cycle
    score = phm08_score(yte_uncapped, yhat)
    return {"RMSE": rmse, "MAE": mae, "PHM08": score, "n": int(Xte.shape[0])}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_all(subsets=("FD001", "FD002", "FD003", "FD004")):
    rows = []
    importances = {}
    for sub in subsets:
        print(f"\n=== {sub} ===", flush=True)
        t0 = time.time()
        Xtr, ytr, Xte, yte_c, yte_u, cols = get_xy_for_subset(sub)
        print(f"  features built in {time.time()-t0:.1f}s. "
              f"X={Xtr.shape}, test={Xte.shape}", flush=True)

        for name, fit_fn in [
            ("Mean", fit_mean_baseline),
            ("Ridge", fit_ridge),
            ("RandomForest", fit_random_forest),
            ("XGBoost", fit_xgb),
            ("LightGBM", fit_lightgbm),
        ]:
            t1 = time.time()
            print(f"    fitting {name}...", flush=True)
            result = fit_fn(Xtr, ytr)
            if isinstance(result, tuple):
                pred, model = result
            else:
                pred, model = result, None
            train_time = time.time() - t1
            metrics = evaluate(pred, Xte, yte_c, yte_u)
            metrics.update({"subset": sub, "model": name,
                            "train_time_s": round(train_time, 2)})
            rows.append(metrics)
            print(f"  {name:>13}: RMSE={metrics['RMSE']:.2f}  "
                  f"MAE={metrics['MAE']:.2f}  PHM08={metrics['PHM08']:.1f}  "
                  f"({train_time:.1f}s)", flush=True)

            # Save the LightGBM model + feature importances for FD001
            if model is not None and name == "LightGBM" and sub == "FD001":
                joblib.dump(model, MODELS / "lightgbm_FD001.joblib")
                fi = pd.DataFrame({
                    "feature": cols,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi.to_csv(RES / f"feature_importance_lightgbm_{sub}.csv",
                          index=False)
                importances[sub] = fi
            if model is not None and name == "XGBoost":
                joblib.dump(model, MODELS / f"xgboost_{sub}.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "baseline_results.csv", index=False)
    print("\nResults written to results/baseline_results.csv")
    return df, importances


if __name__ == "__main__":
    df, importances = run_all()
    print("\n=== Summary ===")
    print(df.pivot(index="model", columns="subset", values="RMSE").round(2))
