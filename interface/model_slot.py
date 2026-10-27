"""
=====================================================================
  MODEL SLOT  —  this is where the team's trained model plugs in
=====================================================================

To connect the real model, the modelling lead only needs to:

  run save_model.py (see README). It trains the final random forest on the
  same participants and split as the report, saves model/model.joblib, writes
  the held-out test results to model/model_info.json and the training-set
  healthy ranges to reference_ranges.json.

  Any other scikit-learn classifier with .predict_proba() (class 1 = disease),
  trained on the columns in features.FEATURE_ORDER, also works.

Nothing else in the app needs to change. While model/model.joblib does not
exist, the app runs in DEMO MODE and shows clearly labelled placeholder
results, so the interface can be designed and reviewed before the model is ready.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_ORDER, compute_features, features_frame

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"
REFERENCE_PATH = Path(__file__).parent / "reference_ranges.json"

THRESHOLD = 0.5          # operating threshold reported in the manuscript
N_BOOTSTRAP = 200        # stride resamples for the per-participant uncertainty range


def load_model():
    """Return the trained model, or None if it has not been added yet (demo mode)."""
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def load_model_info():
    """Held-out test metrics for the reliability card (placeholders until filled in)."""
    if INFO_PATH.exists():
        with open(INFO_PATH) as f:
            return json.load(f)
    return None


def load_reference():
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _prob(model, feature_dict):
    return float(model.predict_proba(features_frame(feature_dict))[0, 1])


def predict(model, df, feature_dict, reference, seed=42):
    """Run the model on one participant.

    Returns a dict with:
      prob, prob_low, prob_high : probability of the disease-group pattern, with a
                                  95% range from re-analysing resampled strides
      drivers                   : list of (feature, contribution) sorted by impact
      demo                      : True if these are placeholder results
    """
    if model is None:
        return _demo_result()

    prob = _prob(model, feature_dict)

    # Uncertainty for THIS participant: resample their strides with replacement,
    # recompute the features and the prediction, and take the middle 95%.
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOTSTRAP):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        boot.append(_prob(model, compute_features(sample)))
    low, high = np.percentile(boot, [2.5, 97.5])

    # Per-participant explanation (works for any model): replace one feature at a
    # time with the healthy-control median and see how much the probability drops.
    # A big drop means that feature pushed this prediction towards "disease".
    drivers = []
    for name in FEATURE_ORDER:
        if pd.isna(feature_dict[name]):
            continue
        neutral = dict(feature_dict)
        neutral[name] = reference[name]["median"]
        drivers.append((name, prob - _prob(model, neutral)))
    drivers.sort(key=lambda d: abs(d[1]), reverse=True)

    return {"prob": prob, "prob_low": float(low), "prob_high": float(high),
            "drivers": drivers, "demo": False}


def _demo_result():
    """Placeholder output shown until model/model.joblib exists. NOT a real prediction."""
    illustrative = [0.21, 0.17, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
    order = ["left_stride_iqr_sec", "right_stride_iqr_sec", "left_stride_sd_sec", "right_stride_sd_sec",
             "left_stride_mean_sec", "right_stride_mean_sec", "left_stride_median_sec",
             "right_stride_median_sec"]
    return {"prob": 0.84, "prob_low": 0.71, "prob_high": 0.93,
            "drivers": list(zip(order, illustrative)), "demo": True}
