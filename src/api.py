"""
Deployment prototype: a thin Flask REST API that serves RUL predictions
from the saved LightGBM and LSTM models.

The API accepts a JSON payload describing a partial run of a turbofan
engine (a list of cycles, each cycle being a 24-element vector of the
three operational settings and 21 raw sensor readings) and returns:

  {
    "model": "...",            # name of the served model
    "subset": "...",           # CMAPSS subset the model was trained on
    "n_cycles_received": ...,
    "predicted_RUL": ...,      # in cycles
    "warning_level": "...",    # "OK" | "WATCH" | "CRITICAL"
    "recommended_action": "...",
  }

Run with:
    python src/api.py        # serves on http://localhost:5000

Test with:
    curl -X POST http://localhost:5000/predict \\
         -H "Content-Type: application/json" \\
         -d @sample_payload.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Flask is optional - the file also exposes a `predict` function that can
# be called directly from notebooks / Streamlit without a web server.
try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import OP_SETTING_COLS, SENSOR_COLS, ALL_COLS
from features import (
    INFORMATIVE_SENSORS, RegimeNormalizer, build_tabular_features,
    add_rolling_stats
)

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CMAPSSData"
MODELS = ROOT / "models"


# ---------------------------------------------------------------------------
# Warning levels - simple but actionable
# ---------------------------------------------------------------------------
def warning_level(rul: float) -> tuple[str, str]:
    """Translate a numerical RUL into a maintenance recommendation."""
    if rul > 80:
        return "OK", "Continue normal operation. Re-evaluate after next cycle."
    if rul > 30:
        return "WATCH", ("Add the asset to the elevated-attention list and "
                          "review sensor trends manually within the week.")
    if rul > 10:
        return "PLAN_MAINTENANCE", ("Schedule preventive maintenance within "
                                     "the next 10 operating cycles.")
    return "CRITICAL", ("Bring the asset offline at the next opportunity; "
                        "risk of unplanned failure is elevated.")


# ---------------------------------------------------------------------------
# Predictor classes
# ---------------------------------------------------------------------------
class LightGBMPredictor:
    """Wraps the saved LightGBM model + the feature pipeline."""

    def __init__(self, subset: str = "FD001"):
        from data_loader import load_subset
        self.subset = subset
        self.name = f"LightGBM-{subset}"
        self.model = joblib.load(MODELS / f"lightgbm_{subset}.joblib")
        # Re-fit the feature scaler on the training data (cheap; ~5s)
        # so we can transform fresh user input consistently.
        train, _, _ = load_subset(DATA, subset)
        self._train = train
        # We don't keep the rolling-stat history; the API receives a fresh
        # trajectory each call and recomputes.

    def predict(self, df_cycles: pd.DataFrame) -> float:
        """df_cycles must have columns 'unit', 'cycle', op_settings, sensors."""
        # Treat the input as a single "engine"
        df = df_cycles.copy()
        # Standardise the same way as in training
        tr_f, te_f = build_tabular_features(self._train, df,
                                            subset=self.subset)
        last = te_f.groupby("unit").tail(1).reset_index(drop=True)
        feat_cols = [c for c in te_f.columns
                     if c not in ("unit", "cycle", "RUL")]
        X = last[feat_cols].values.astype(np.float32)
        pred = float(np.clip(self.model.predict(X)[0], 0, 125))
        return pred


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(subset: str = "FD001"):
    """Build a Flask app whose /predict endpoint serves the chosen subset."""
    if not HAS_FLASK:
        raise RuntimeError("flask is not installed; install with `pip install flask`.")
    app = Flask("rul_api")
    predictor = LightGBMPredictor(subset=subset)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "service": "CMAPSS RUL prediction",
            "model": predictor.name,
            "endpoints": ["/predict (POST)", "/health (GET)"],
            "expected_payload": {
                "cycles": "list of cycle records (each is a list of 24 floats: "
                           "3 op_settings + 21 sensors) in chronological order",
            },
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": predictor.name})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            payload = request.get_json(force=True)
            cycles = payload["cycles"]
            arr = np.asarray(cycles, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 24:
                return jsonify({
                    "error": "Each cycle must be a list of 24 floats "
                             "(3 op_settings + 21 sensors). "
                             f"Got shape {arr.shape}."
                }), 400

            df = pd.DataFrame(arr, columns=OP_SETTING_COLS + SENSOR_COLS)
            df.insert(0, "cycle", np.arange(1, len(df) + 1))
            df.insert(0, "unit", 1)

            rul = predictor.predict(df)
            lvl, action = warning_level(rul)
            return jsonify({
                "model": predictor.name,
                "subset": predictor.subset,
                "n_cycles_received": int(len(df)),
                "predicted_RUL_cycles": round(rul, 2),
                "warning_level": lvl,
                "recommended_action": action,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


# ---------------------------------------------------------------------------
# Sample payload generator (for testing the API end-to-end)
# ---------------------------------------------------------------------------
def make_sample_payload(subset: str = "FD001", unit_id: int = 1,
                        n_cycles: int | None = None) -> dict:
    """Build a JSON-serialisable payload from one of the held-out test engines."""
    from data_loader import load_subset
    _, test, y_true = load_subset(DATA, subset)
    eng = test[test["unit"] == unit_id].sort_values("cycle")
    if n_cycles is not None:
        eng = eng.head(n_cycles)
    cycles_array = eng[OP_SETTING_COLS + SENSOR_COLS].values.tolist()
    return {
        "cycles": cycles_array,
        "_meta": {
            "subset": subset,
            "unit_id": int(unit_id),
            "n_cycles_supplied": int(len(cycles_array)),
            "true_RUL_at_end_of_recorded_run": float(y_true[unit_id - 1]),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--write-sample", type=str, default=None,
                        help="instead of serving, write a sample payload to "
                             "this path and exit")
    args = parser.parse_args()

    if args.write_sample:
        payload = make_sample_payload(args.subset, unit_id=1)
        Path(args.write_sample).write_text(json.dumps(payload, indent=2))
        print(f"sample payload for {args.subset} written to {args.write_sample}")
    else:
        app = create_app(args.subset)
        print(f"Serving CMAPSS RUL API on port {args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)
