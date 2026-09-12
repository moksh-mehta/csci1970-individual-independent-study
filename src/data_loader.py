"""
C-MAPSS Data Loader
-------------------
Loads NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
turbofan engine degradation dataset and computes Remaining Useful Life (RUL).

Reference:
    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
    modeling for aircraft engine run-to-failure simulation. PHM 2008.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Column names following the C-MAPSS convention
ID_COLS = ["unit", "cycle"]
OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]   # 21 sensors (s1..s21)
ALL_COLS = ID_COLS + OP_SETTING_COLS + SENSOR_COLS

# Sensors that are constant or near-constant across all engines in FD001/FD003
# (selected through variance analysis - see notebooks/eda.py)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]


def load_subset(data_dir: str | Path, subset: str = "FD001"):
    """Load one of FD001, FD002, FD003, FD004.

    Returns (train_df, test_df, y_test) where:
        train_df : full run-to-failure trajectories with RUL column added
        test_df  : truncated trajectories (failure occurs after recorded end)
        y_test   : true RUL at the last recorded cycle for each test engine
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+",
                        header=None, engine="python")
    test = pd.read_csv(data_dir / f"test_{subset}.txt", sep=r"\s+",
                       header=None, engine="python")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+",
                      header=None, engine="python")

    train.columns = ALL_COLS
    test.columns = ALL_COLS
    rul.columns = ["RUL"]

    # Compute RUL for the training set as (max_cycle_for_unit - current_cycle).
    # Standard convention since Heimes (2008).
    max_cycles = train.groupby("unit")["cycle"].max().rename("max_cycle")
    train = train.join(max_cycles, on="unit")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    train = train.drop(columns=["max_cycle"])

    return train, test, rul["RUL"].values


def apply_piecewise_rul(rul_series, cap: int = 125):
    """Apply the piecewise-linear RUL target (Heimes, 2008).

    In the early phase the machine is healthy and its 'remaining useful life'
    is not really predictable from sensors - it just decreases linearly with
    time.  Capping RUL at some value (typically 125-130 for CMAPSS) forces
    the model to focus on the degradation phase, where the prediction problem
    is well-posed.
    """
    if isinstance(rul_series, pd.Series):
        return rul_series.clip(upper=cap)
    return np.minimum(rul_series, cap)


def add_test_rul(test_df: pd.DataFrame, y_test: np.ndarray,
                 cap: int | None = 125) -> pd.DataFrame:
    """Add a per-cycle RUL column to the test dataframe.

    For each engine, the provided y_test gives the true RUL at the *last*
    recorded cycle. We linearly back-fill it for earlier cycles.
    """
    test_df = test_df.copy()
    last_cycles = test_df.groupby("unit")["cycle"].max().to_dict()
    units_sorted = sorted(last_cycles.keys())
    rul_at_last = dict(zip(units_sorted, y_test))

    def _per_unit(group):
        u = group.name
        max_c = last_cycles[u]
        return rul_at_last[u] + (max_c - group["cycle"])

    test_df["RUL"] = test_df.groupby("unit", group_keys=False).apply(_per_unit)
    if cap is not None:
        test_df["RUL"] = apply_piecewise_rul(test_df["RUL"], cap=cap)
    return test_df


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric scoring function from the PHM 2008 challenge.

    Penalises late predictions (y_pred > y_true) more than early ones,
    because in maintenance, missing a failure is worse than a false alarm.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score = np.where(d < 0,
                     np.exp(-d / 13.0) - 1.0,
                     np.exp(d / 10.0) - 1.0)
    return float(score.sum())


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "data" / "CMAPSSData"
    for sub in ("FD001", "FD002", "FD003", "FD004"):
        tr, te, y = load_subset(base, sub)
        print(f"{sub}: train shape={tr.shape}, test shape={te.shape}, "
              f"n_train_engines={tr['unit'].nunique():>3}, "
              f"n_test_engines={te['unit'].nunique():>3}, "
              f"max RUL train={tr['RUL'].max()}")
