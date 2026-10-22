from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)
data = pd.read_csv("data/processed/gait_features.csv")

# Use the same dataset used in the report
data = data.loc[data["participant_id"] != "hunt20"].copy()

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

X = data[feature_columns]
y = data["disease"]

seeds = [42, 43, 44, 45, 46]

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

rows = []

for seed in seeds:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities,
        )
        auprc = average_precision_score(
    y_test,
    probabilities,
)
        predictions = (
    probabilities >= 0.5
).astype(int)
        balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions,
)
        tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)


        rows.append({
            "seed": seed,
            "model": model_name,
            "test_auc": auc,
            "test_auprc": auprc,
            "test_sensitivity": sensitivity,
            "test_specificity": specificity,
            "test_balanced_accuracy": balanced_accuracy,

        })

results = pd.DataFrame(rows)

print("\n===== AUROC BY SEED =====")
print(results)

summary = (
    results.groupby("model")
    .agg({
        "test_auc": ["mean", "std"],
        "test_auprc": ["mean", "std"],
        "test_sensitivity": ["mean", "std"],
        "test_specificity": ["mean", "std"],
        "test_balanced_accuracy": ["mean", "std"],
    })
)


print("\n===== MEAN ± SD =====")

for model in summary.index:

    print(f"\n{model}")

    print(
        f"AUROC: "
        f"{summary.loc[model, ('test_auc', 'mean')]:.3f} ± "
        f"{summary.loc[model, ('test_auc', 'std')]:.3f}"
    )

    print(
        f"AUPRC: "
        f"{summary.loc[model, ('test_auprc', 'mean')]:.3f} ± "
        f"{summary.loc[model, ('test_auprc', 'std')]:.3f}"
    )

    print(
        f"Sensitivity: "
        f"{summary.loc[model, ('test_sensitivity', 'mean')]:.3f} ± "
        f"{summary.loc[model, ('test_sensitivity', 'std')]:.3f}"
    )

    print(
        f"Specificity: "
        f"{summary.loc[model, ('test_specificity', 'mean')]:.3f} ± "
        f"{summary.loc[model, ('test_specificity', 'std')]:.3f}"
    )

    print(f"Balanced Accuracy: "
        f"{summary.loc[model, ('test_balanced_accuracy', 'mean')]:.3f} ± "
        f"{summary.loc[model, ('test_balanced_accuracy', 'std')]:.3f}"
    )

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results.to_csv(
    output_folder / "auroc_by_seed.csv",
    index=False,
)

summary.to_csv(
    output_folder / "auroc_summary.csv",
    index=False,
)

print("\nSaved:")
print("results/auroc_by_seed.csv")
print("results/auroc_summary.csv")