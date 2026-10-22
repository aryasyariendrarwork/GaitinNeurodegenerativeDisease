from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/processed/gait_features.csv")

feature_columns = [
    "left_stride_mean_sec",
    "left_stride_sd_sec",
    "left_stride_median_sec",
    "left_stride_iqr_sec",
    "right_stride_mean_sec",
    "right_stride_sd_sec",
    "right_stride_median_sec",
    "right_stride_iqr_sec",
]

if data["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs found.")

if not np.isfinite(data[feature_columns].to_numpy()).all():
    raise ValueError("Features contain missing or infinite values.")

if set(data["disease"].unique()) != {0, 1}:
    raise ValueError("Expected disease labels 0 and 1.")

datasets = {
    "all_participants": data.copy(),
    "without_hunt20": data.loc[
        data["participant_id"] != "hunt20"
    ].copy(),
}

rows = []

for dataset_name, participants in datasets.items():
    participants = participants.reset_index(drop=True)
    X = participants[feature_columns]
    y = participants["disease"]

    print(f"\nEvaluating: {dataset_name}", flush=True)

    for repeat in range(20):
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42 + repeat,
        )

        models = {
            "baseline": DummyClassifier(strategy="prior"),
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
            "random_forest": RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
),
        }

        for model_name, model in models.items():
            probabilities = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict_proba",
            )[:, 1]

            predictions = (probabilities >= 0.5).astype(int)

            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "repeat": repeat + 1,
                "balanced_accuracy": balanced_accuracy_score(
                    y, predictions
                ),
                "roc_auc": roc_auc_score(y, probabilities),
                "sensitivity": recall_score(
                    y, predictions, pos_label=1
                ),
                "specificity": recall_score(
                    y, predictions, pos_label=0
                ),
            })

        if (repeat + 1) % 5 == 0:
            print(
                f"Completed {repeat + 1} of 20 repeats",
                flush=True,
            )

results = pd.DataFrame(rows)

metrics = [
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
]

summary = results.groupby(["dataset", "model"])[metrics].agg(
    ["mean", "std", "min", "max"]
)

summary.columns = [
    f"{metric}_{statistic}"
    for metric, statistic in summary.columns
]

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results.to_csv(
    output_folder / "repeated_cv_results.csv",
    index=False,
)

summary.to_csv(
    output_folder / "repeated_cv_summary.csv"
)

print("\nAverage results across 20 repeats:")
print(
    summary[
        [
            "balanced_accuracy_mean",
            "roc_auc_mean",
            "sensitivity_mean",
            "specificity_mean",
        ]
    ].round(3).to_string()
)

print("\nResults saved in the results folder.")
print("Variation across repeats is not a confidence interval.")