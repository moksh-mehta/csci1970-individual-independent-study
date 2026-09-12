"""
Feature engineering for C-MAPSS RUL prediction.

Two main entry points:

  build_tabular_features(...)   -> dataframe with one row per (engine, cycle)
                                   suitable for classical ML
  build_sequence_dataset(...)   -> 3-D array (n, window, features) suitable
                                   for recurrent / convolutional models

Both can operate in "single regime" mode (FD001/FD003) or "multi-regime"
mode (FD002/FD004), where features are standardised within each of the
six operational regimes recovered by k-means on the op-setting columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_loader import (
    CONSTANT_SENSORS, OP_SETTING_COLS, SENSOR_COLS
)

# Sensors that retain useful variance once constants are removed
INFORMATIVE_SENSORS = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS]


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
@dataclass
class RegimeNormalizer:
    """Cluster operating conditions and standardise sensors per cluster.

    Fits on training data; ``transform`` may be applied to test data.
    """
    n_regimes: int = 6
    feature_cols: Iterable[str] = tuple(SENSOR_COLS)

    def fit(self, df: pd.DataFrame):
        self.km_ = KMeans(n_clusters=self.n_regimes, n_init=10,
                          random_state=0).fit(df[OP_SETTING_COLS].values)
        regimes = self.km_.predict(df[OP_SETTING_COLS].values)
        self.scalers_ = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            scaler = StandardScaler()
            scaler.fit(df.loc[mask, list(self.feature_cols)].values)
            self.scalers_[r] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        regimes = self.km_.predict(df[OP_SETTING_COLS].values)
        out = df.copy()
        # Pre-allocate the destination columns to avoid dtype churn
        for col in self.feature_cols:
            out[col] = 0.0
        cols = list(self.feature_cols)
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            out.loc[mask, cols] = self.scalers_[r].transform(
                df.loc[mask, cols].values)
        out["regime"] = regimes
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def standardise_single(train: pd.DataFrame, test: pd.DataFrame,
                       cols=INFORMATIVE_SENSORS):
    """Plain StandardScaler fit on train, applied to test (single-regime)."""
    scaler = StandardScaler().fit(train[cols].values)
    train = train.copy(); test = test.copy()
    train[cols] = scaler.transform(train[cols].values)
    test[cols] = scaler.transform(test[cols].values)
    return train, test, scaler


# ---------------------------------------------------------------------------
# Rolling statistical features
# ---------------------------------------------------------------------------
def add_rolling_stats(df: pd.DataFrame, cols=INFORMATIVE_SENSORS,
                      windows=(5, 20)) -> pd.DataFrame:
    """Per-engine rolling mean/std/trend over the past ``window`` cycles.

    Trend is the slope of a least-squares line fit through the window;
    we compute it via the analytic closed-form for evenly-spaced x to
    avoid the cost of polyfit inside ``rolling.apply``.
    """
    out = df.copy()
    g = out.groupby("unit", sort=False)
    for w in windows:
        means = g[cols].rolling(w, min_periods=1).mean().reset_index(0, drop=True)
        stds = g[cols].rolling(w, min_periods=1).std().reset_index(0, drop=True)
        for c in cols:
            out[f"{c}_mean_w{w}"] = means[c]
            out[f"{c}_std_w{w}"] = stds[c].fillna(0.0)
        # trend (slope) via cumulative running fit
        for c in cols:
            out[f"{c}_trend_w{w}"] = (
                g[c].transform(lambda s: _rolling_slope(s.values, w))
            )
    return out


def _rolling_slope(x: np.ndarray, w: int) -> np.ndarray:
    """Closed-form OLS slope of x[t-w+1:t+1] vs t (zero-indexed).

    Fully vectorised using cumulative-sum tricks: O(n) instead of O(n*w).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n)
    if n < 2:
        return out

    csum_v = np.concatenate(([0.0], np.cumsum(x)))
    csum_w = np.concatenate(([0.0], np.cumsum(np.arange(n) * x)))

    end = np.arange(n)
    start = np.maximum(0, end - w + 1)
    k = end - start + 1                  # window length at each position

    sum_v = csum_v[end + 1] - csum_v[start]
    sum_wv = csum_w[end + 1] - csum_w[start]
    sum_iv = sum_wv - start * sum_v
    i_mean = (k - 1) / 2.0
    numer = sum_iv - i_mean * sum_v
    denom = k * (k ** 2 - 1) / 12.0
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def add_lifetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle counters.

    NOTE: We deliberately do *not* add ``cycle_norm = cycle / max_cycle``
    here, even though it looks tempting.  In the training set every engine
    runs to failure so ``cycle_norm == 1`` is perfectly correlated with
    RUL == 0.  In the test set every engine is truncated *before* failure,
    so ``cycle_norm == 1`` corresponds to a wide range of true RUL values.
    A model that uses this feature catastrophically over-fits the training
    distribution and predicts RUL near 0 for healthy test engines.

    The raw ``cycle`` column carries similar information but with a much
    weaker signal, so it is kept and the cycle_norm column is omitted.
    """
    return df


# ---------------------------------------------------------------------------
# Tabular feature pipeline
# ---------------------------------------------------------------------------
def build_tabular_features(train: pd.DataFrame, test: pd.DataFrame,
                           subset: str = "FD001",
                           windows=(5, 20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_train, X_test) feature dataframes for tabular ML.

    Steps:
      1. Drop near-constant sensors.
      2. Standardise (regime-aware for FD002/FD004; global otherwise).
      3. Add rolling mean / std / trend over multiple window sizes.
      4. Add cycle counter.

    The returned dataframes still carry the ``unit`` and ``cycle`` columns
    (and ``RUL`` on the training side if present) so the caller can split
    by engine.
    """
    if subset in ("FD002", "FD004"):
        normalizer = RegimeNormalizer(n_regimes=6,
                                      feature_cols=INFORMATIVE_SENSORS)
        train_n = normalizer.fit_transform(train)
        test_n = normalizer.transform(test)
    else:
        train_n, test_n, _ = standardise_single(train, test,
                                                cols=INFORMATIVE_SENSORS)

    train_f = add_rolling_stats(train_n, cols=INFORMATIVE_SENSORS,
                                windows=windows)
    test_f = add_rolling_stats(test_n, cols=INFORMATIVE_SENSORS,
                               windows=windows)
    train_f = add_lifetime_features(train_f)
    test_f = add_lifetime_features(test_f)
    return train_f, test_f


# ---------------------------------------------------------------------------
# Sequence dataset for deep models
# ---------------------------------------------------------------------------
def build_sequence_dataset(train_f: pd.DataFrame, test_f: pd.DataFrame,
                           feature_cols: Iterable[str],
                           y_test: np.ndarray,
                           window: int = 30, cap: int = 125):
    """Convert per-cycle frames into (n, window, k) arrays for recurrent nets.

    For each cycle ``t`` in each engine we take the preceding ``window``
    cycles as one sample.  Engines with fewer than ``window`` cycles are
    front-padded with zeros (which after standardisation correspond to
    the mean value).

    Returns:
        X_train, y_train: (N, window, k), (N,)
        X_test, y_test:   (n_test_engines, window, k), (n_test_engines,)
            Test set uses only the *last* window per engine, paired with
            the supplied true RUL.
    """
    feature_cols = list(feature_cols)
    Xtr, ytr = [], []
    for _, eng in train_f.groupby("unit", sort=False):
        arr = eng[feature_cols].values.astype(np.float32)
        rul = eng["RUL"].values.astype(np.float32)
        n = len(arr)
        for t in range(n):
            start = max(0, t - window + 1)
            seg = arr[start:t + 1]
            if seg.shape[0] < window:
                pad = np.zeros((window - seg.shape[0], len(feature_cols)),
                               dtype=np.float32)
                seg = np.vstack([pad, seg])
            Xtr.append(seg)
            ytr.append(min(rul[t], cap))
    Xtr = np.stack(Xtr)
    ytr = np.asarray(ytr, dtype=np.float32)

    Xte, yte = [], []
    units_sorted = sorted(test_f["unit"].unique())
    for i, u in enumerate(units_sorted):
        eng = test_f[test_f["unit"] == u]
        arr = eng[feature_cols].values.astype(np.float32)
        n = len(arr)
        start = max(0, n - window)
        seg = arr[start:n]
        if seg.shape[0] < window:
            pad = np.zeros((window - seg.shape[0], len(feature_cols)),
                           dtype=np.float32)
            seg = np.vstack([pad, seg])
        Xte.append(seg)
        yte.append(min(float(y_test[i]), cap))
    Xte = np.stack(Xte)
    yte = np.asarray(yte, dtype=np.float32)
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    from data_loader import load_subset
    base = Path(__file__).resolve().parents[1] / "data" / "CMAPSSData"
    tr, te, y = load_subset(base, "FD001")
    print("Building tabular features for FD001 (this may take ~20s)...")
    tr_f, te_f = build_tabular_features(tr, te, subset="FD001",
                                        windows=(5, 20))
    print("train features shape:", tr_f.shape)
    print("test features shape: ", te_f.shape)
    print("new columns sample:", [c for c in tr_f.columns
                                  if c.endswith("_w20")][:6])
