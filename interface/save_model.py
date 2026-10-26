"""
Train and save the final model for the interface.

Reproduces the team's final setup from the modelling script:
  - features: the 8 stride features in data/processed/gait_features.csv
  - participants: without hunt20 (faulty right-foot sensor), as in the report
  - split: stratified 80/20 train/test, random_state=42
  - model: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

It then writes three files the app reads:
  model/model.joblib       the trained random forest (trained on the 80% only)
  model/model_info.json    its results on the held-out 20% (the test set, used once)
  reference_ranges.json    healthy ranges from the TRAINING-set controls only

Usage (from inside the interface/ folder):
  python3 save_model.py
  python3 save_model.py --features-csv ../data/processed/gait_features.csv
  python3 save_model.py --data-dir ~/Downloads/gait-in-neurodegenerative-disease-database-1.0.0
      (--data-dir also checks that features.py reproduces the CSV values exactly)
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

from features import FEATURE_ORDER, compute_features, load_ts

HERE = Path(__file__).parent
AXES = {"mean": (0.8, 1.8), "median": (0.8, 1.8), "sd": (0.0, 0.3), "iqr": (0.0, 0.3)}

parser = argparse.ArgumentParser()
parser.add_argument("--features-csv", default=str(HERE.parent / "data" / "processed" / "gait_features.csv"))
parser.add_argument("--data-dir", default=None, help="folder of raw .ts files, to verify features.py")
parser.add_argument("--include-hunt20", action="store_true", help="train on all 64 participants instead")
args = parser.parse_args()

# ------------------------------------------------------------------ data ---
data = pd.read_csv(args.features_csv)
if data["participant_id"].duplicated().any():
    raise SystemExit("Duplicate participant IDs found.")
if not np.isfinite(data[FEATURE_ORDER].to_numpy()).all():
    raise SystemExit("Features contain missing or infinite values.")
if set(data["disease"].unique()) != {0, 1}:
    raise SystemExit("Expected disease labels 0 and 1.")
if not args.include_hunt20:
    data = data.loc[data["participant_id"] != "hunt20"]
data = data.reset_index(drop=True)
print(f"Participants used: {len(data)} ({int((data.disease == 0).sum())} controls)")

# Optional: check the app computes exactly the same features as the training CSV.
if args.data_dir:
    worst = 0.0
    for _, row in data.iterrows():
        path = Path(args.data_dir).expanduser() / f"{row.participant_id}.ts"
        df, _ = load_ts(path.read_bytes())
        app = compute_features(df)
        worst = max(worst, max(abs(app[k] - row[k]) for k in FEATURE_ORDER))
    status = "OK" if worst < 1e-9 else "MISMATCH: features.py differs from the training features"
    print(f"Feature check against raw files: largest difference {worst:.2e} -> {status}")

X, y = data[FEATURE_ORDER], data["disease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# ----------------------------------------------------------------- model ---
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Held-out test set: evaluated once.
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
auc = roc_auc_score(y_test, prob)

rng = np.random.default_rng(42)
boot = []
yt, pt = y_test.to_numpy(), prob
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt))
    if len(np.unique(yt[i])) == 2:
        boot.append(roc_auc_score(yt[i], pt[i]))
lo, hi = np.percentile(boot, [2.5, 97.5])

info = {
    "model_name": "Random forest (8 stride features)",
    "sensitivity_pct": round(100 * tp / (tp + fn)),
    "specificity_pct": round(100 * tn / (tn + fp)),
    "auroc": round(float(auc), 2),
    "auroc_ci": f"{lo:.2f}-{hi:.2f}",
    "auprc": round(float(average_precision_score(y_test, prob)), 2),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 2),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_test_controls": int((y_test == 0).sum()),
    "threshold": 0.5,
    "participants": "all 64" if args.include_hunt20 else "63 (hunt20 excluded)",
}

(HERE / "model").mkdir(exist_ok=True)
joblib.dump(model, HERE / "model" / "model.joblib")
with open(HERE / "model" / "model_info.json", "w") as f:
    json.dump(info, f, indent=2)

# ------------------------------------------- healthy reference ranges ------
controls = X_train[y_train == 0]
ref = {"_note": f"Healthy ranges from the {len(controls)} TRAINING-set controls "
                "(25th-75th percentile and median). Written by save_model.py."}
for k in FEATURE_ORDER:
    stat = k.split("_")[2]                       # mean / sd / median / iqr
    ref[k] = {"q25": round(float(controls[k].quantile(0.25)), 4),
              "q75": round(float(controls[k].quantile(0.75)), 4),
              "median": round(float(controls[k].median()), 4),
              "axis_min": AXES[stat][0], "axis_max": AXES[stat][1]}
with open(HERE / "reference_ranges.json", "w") as f:
    json.dump(ref, f, indent=2)

print(f"Train {info['n_train']} / test {info['n_test']} ({info['n_test_controls']} controls in test)")
print(f"TEST  AUROC {info['auroc']} (95% CI {info['auroc_ci']})  AUPRC {info['auprc']}  "
      f"sens {info['sensitivity_pct']}%  spec {info['specificity_pct']}%  bal.acc {info['balanced_accuracy']}")
print("Saved model/model.joblib, model/model_info.json and reference_ranges.json")
