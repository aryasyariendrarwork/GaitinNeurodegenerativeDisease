from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/processed/gait_features.csv")

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

    print(f"\nChecking features: {dataset_name}", flush=True)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for fold, (train_indices, test_indices) in enumerate(
        cv.split(X, y), start=1
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        )

        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        importance = permutation_importance(
            model,
            X.iloc[test_indices],
            y.iloc[test_indices],
            scoring="balanced_accuracy",
            n_repeats=30,
            random_state=42 + fold,
        )

        for index, feature in enumerate(feature_columns):
            rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "feature": feature,
                "accuracy_drop": importance.importances_mean[index],
            })

        print(f"Completed fold {fold} of 5", flush=True)

results = pd.DataFrame(rows)

summary = (
    results.groupby(["dataset", "feature"])["accuracy_drop"]
    .agg(["mean", "std"])
    .reset_index()
)

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results.to_csv(
    output_folder / "feature_importance_by_fold.csv",
    index=False,
)
summary.to_csv(
    output_folder / "feature_importance_summary.csv",
    index=False,
)

labels = [
    "Left stride median",
    "Left stride IQR",
    "Right stride median",
    "Right stride IQR",
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

for ax, dataset_name in zip(axes, datasets):
    values = (
        summary.loc[summary["dataset"] == dataset_name]
        .set_index("feature")
        .loc[feature_columns]
    )

    ax.barh(
        labels,
        values["mean"] * 100,
        xerr=values["std"] * 100,
        color="steelblue",
        capsize=4,
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Balanced accuracy drop (percentage points)")
    ax.set_title(
        "All participants"
        if dataset_name == "all_participants"
        else "Without hunt20"
    )

    print(f"\nFeature importance: {dataset_name}")
    print(values.round(4).to_string())

fig.suptitle("Preliminary importance on held-out participants")
fig.tight_layout()
fig.savefig(
    output_folder / "feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

print("\nTables and graph saved in results.")
print("Error bars show variation across five folds, not confidence intervals.")
plt.show()