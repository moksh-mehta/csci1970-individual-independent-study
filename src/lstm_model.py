"""
LSTM-based RUL prediction for the C-MAPSS dataset.

We follow the now-standard architecture: a stack of two LSTM layers on
a sliding 30-cycle window of the regime-normalised sensor features,
followed by a fully-connected head that regresses the (capped) RUL.

References:
  - Zheng, S., Ristovski, K., Farahat, A., Gupta, C. (2017) "Long
    short-term memory network for remaining useful life estimation",
    ICPHM.
  - Heimes, F. (2008) "Recurrent neural networks for remaining useful
    life estimation", PHM 2008.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (
    add_test_rul, apply_piecewise_rul, load_subset, phm08_score
)
from features import (
    INFORMATIVE_SENSORS, build_sequence_dataset, build_tabular_features
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CMAPSSData"
RES = ROOT / "results"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class RULNet(nn.Module):
    """Two-layer LSTM + FC head.

    Input shape : (batch, seq, n_features)
    Output      : (batch,) -- predicted RUL
    """

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]      # the representation at t = window end
        y = self.head(last).squeeze(-1)
        return y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def fit_lstm(Xtr, ytr, Xva, yva,
             epochs=40, batch_size=512, lr=1e-3, weight_decay=1e-5,
             hidden=64, dropout=0.2, patience=8, verbose=True):
    n_feat = Xtr.shape[-1]
    model = RULNet(n_feat, hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(Xtr),
                              torch.from_numpy(ytr))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    yva_t = torch.from_numpy(yva).to(DEVICE)

    history = []
    best_val = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        with torch.no_grad():
            preds = model(Xva_t)
            val_rmse = torch.sqrt(((preds - yva_t) ** 2).mean()).item()

        sched.step(val_rmse)
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_rmse": val_rmse})
        if verbose:
            print(f"    epoch {ep:>3}: train_loss={train_loss:.3f}  "
                  f"val_rmse={val_rmse:.3f}", flush=True)
        if val_rmse < best_val - 1e-3:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep} (best val={best_val:.3f})",
                          flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Build sequences for one subset, including a train/validation split by engine
# ---------------------------------------------------------------------------
def build_data_for_subset(subset: str, window: int = 30, cap: int = 125,
                          val_frac: float = 0.15, seed: int = 0):
    train, test, y_true = load_subset(DATA, subset)
    tr_f, te_f = build_tabular_features(train, test, subset=subset)

    # LSTM input: only the standardised informative sensors.  We drop the
    # constant sensors (zero variance), the rolling statistics (the LSTM
    # learns them internally), the op_settings (already encoded by the
    # regime-aware normalisation), and of course the bookkeeping columns.
    feat_cols = list(INFORMATIVE_SENSORS)
    if "regime" in tr_f.columns:
        feat_cols.append("regime")

    rng = np.random.RandomState(seed)
    units = tr_f["unit"].unique()
    n_val = max(1, int(round(val_frac * len(units))))
    val_units = set(rng.choice(units, size=n_val, replace=False))
    tr_part = tr_f[~tr_f["unit"].isin(val_units)]
    va_part = tr_f[tr_f["unit"].isin(val_units)]

    Xtr, ytr, Xte, yte = build_sequence_dataset(
        tr_part, te_f, feat_cols, y_true, window=window, cap=cap)
    Xva, yva, _, _ = build_sequence_dataset(
        va_part, te_f, feat_cols, y_true, window=window, cap=cap)

    return Xtr, ytr, Xva, yva, Xte, yte, y_true, feat_cols


def run_lstm_for_subset(subset: str, window: int = 30, epochs: int = 30,
                        verbose: bool = True):
    print(f"\n=== LSTM {subset} ===", flush=True)
    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte_capped, yte_uncapped, feat_cols = \
        build_data_for_subset(subset, window=window)
    print(f"  data: train={Xtr.shape}, val={Xva.shape}, "
          f"test={Xte.shape}  ({time.time()-t0:.1f}s)", flush=True)

    model, history = fit_lstm(Xtr, ytr, Xva, yva, epochs=epochs,
                              verbose=verbose)

    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(Xte).to(DEVICE)
        yhat = model(Xte_t).cpu().numpy()
    yhat = np.clip(yhat, 0, None)

    rmse = float(np.sqrt(((yte_capped - yhat) ** 2).mean()))
    mae = float(np.abs(yte_capped - yhat).mean())
    score = phm08_score(yte_uncapped, yhat)

    print(f"  test: RMSE={rmse:.2f}  MAE={mae:.2f}  PHM08={score:.1f}",
          flush=True)

    # Save model + per-engine predictions
    torch.save(model.state_dict(), MODELS / f"lstm_{subset}.pt")
    pred_df = pd.DataFrame({
        "engine": np.arange(1, len(yhat) + 1),
        "true_RUL_capped": yte_capped,
        "true_RUL_raw": yte_uncapped,
        "pred_RUL": yhat,
    })
    pred_df.to_csv(RES / f"lstm_predictions_{subset}.csv", index=False)
    pd.DataFrame(history).to_csv(
        RES / f"lstm_history_{subset}.csv", index=False)

    return {
        "subset": subset, "model": "LSTM",
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "PHM08": round(score, 1),
        "n": int(len(yhat)),
        "n_features": Xtr.shape[-1],
        "n_train_seq": int(Xtr.shape[0]),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }, model, feat_cols


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    results = []
    for sub in ("FD001", "FD003", "FD002", "FD004"):
        res, model, cols = run_lstm_for_subset(sub, window=30, epochs=25,
                                                verbose=True)
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RES / "lstm_results.csv", index=False)
    print("\n=== LSTM summary ===")
    print(df.to_string(index=False))
