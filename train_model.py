from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/processed/gait_features.csv")

# Use only gait measurements as predictors.
feature_columns = [
    "left_stride_median_sec",
    "left_stride_iqr_sec",
    "right_stride_median_sec",
    "right_stride_iqr_sec",
]

if data["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs found.")

if not np.isfinite(data[feature_columns].to_numpy()).all():
    raise ValueError("Features contain missing or infinite values.")

if set(data["disease"].unique()) != {0, 1}:
    raise ValueError("Expected disease labels 0 and 1.")

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results = []

# Compare both datasets without changing the original CSV.
datasets = {
    "all_participants": data.copy(),
    "without_hunt20": data.loc[
        data["participant_id"] != "hunt20"
    ].copy(),
}

for dataset_name, participants in datasets.items():
    participants = participants.reset_index(drop=True)

    X = participants[feature_columns]
    y = participants["disease"]

    # Five groups: train on four, predict the remaining group.
    # Scaling is learned separately within each training group.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
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
    }

    print(f"\nDataset: {dataset_name}")
    print("Participants:", len(participants))

    predictions = participants[
        ["participant_id", "group", "disease"]
    ].copy()

    for model_name, model in models.items():
        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        predicted_labels = (probabilities >= 0.5).astype(int)

        balanced_accuracy = balanced_accuracy_score(
            y, predicted_labels
        )
        auc = roc_auc_score(y, probabilities)

        tn, fp, fn, tp = confusion_matrix(
            y, predicted_labels, labels=[0, 1]
        ).ravel()

        print(f"\nModel: {model_name}")
        print(
            "Balanced accuracy:",
            round(balanced_accuracy, 3),
        )
        print("ROC AUC:", round(auc, 3))
        print("Controls correctly predicted:", tn)
        print("Controls predicted as disease:", fp)
        print("Disease predicted as control:", fn)
        print("Disease correctly predicted:", tp)

        predictions[f"{model_name}_probability"] = probabilities
        predictions[f"{model_name}_prediction"] = predicted_labels

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "participants": len(participants),
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": auc,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

    predictions.to_csv(
        output_folder / f"{dataset_name}_predictions.csv",
        index=False,
    )

pd.DataFrame(results).to_csv(
    output_folder / "preliminary_model_results.csv",
    index=False,
)

print("\nResults saved in the results folder.")
print("These are preliminary cross-validation results.")