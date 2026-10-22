from pathlib import Path

from matplotlib.pylab import split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/processed/gait_features.csv")

# Use only gait measurements as predictors.
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

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

results = []
seeds = [42, 43, 44, 45, 46]
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
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

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

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
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
    X_trainval,
    y_trainval,
    cv=cv,
    method="predict_proba",
)[:, 1]

        predicted_labels = (probabilities >= 0.5).astype(int)
        model.fit(X_trainval, y_trainval)
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_predictions = (
    test_probabilities >= 0.5
).astype(int)
        test_auc = roc_auc_score(
    y_test,
    test_probabilities,
)
        test_auprc = average_precision_score(
    y_test,
    test_probabilities,
)
        test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions,
)
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()
        test_sensitivity = (
    test_tp / (test_tp + test_fn)
)
        test_specificity = (
    test_tn / (test_tn + test_fp)
)

        print("\nHELD-OUT TEST RESULTS")
        print("Test Balanced Accuracy:",
    round(test_balanced_accuracy, 3),
        )
        print("Test ROC AUC:", round(test_auc, 3))
        print("Test AUPRC:", round(test_auprc, 3))
        print("Test Sensitivity:", round(test_sensitivity, 3),
)
        print("Test Specificity:",round(test_specificity, 3))

        
        balanced_accuracy = balanced_accuracy_score(
             y_trainval, predicted_labels
        )
        auc = roc_auc_score(y_trainval, probabilities)
        auprc = average_precision_score(y_trainval, probabilities)

        tn, fp, fn, tp = confusion_matrix(
            y_trainval, predicted_labels, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f"\nModel: {model_name}")
        print(
            "Balanced accuracy:",
            round(balanced_accuracy, 3),
        )
        print("ROC AUC:", round(auc, 3))
        print("AUPRC:", round(auprc, 3))
        print("Controls correctly predicted:", tn)
        print("Controls predicted as disease:", fp)
        print("Disease predicted as control:", fn)
        print("Disease correctly predicted:", tp)
        print("Sensitivity:", round(sensitivity, 3))
        print("Specificity:", round(specificity, 3))

        #predictions[f"{model_name}_probability"] = probabilities
        #predictions[f"{model_name}_prediction"] = predicted_labels

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "participants": len(participants),
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": auc,
            "auprc": auprc,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })

        

    #predictions.to_csv(
        output_folder / f"{dataset_name}_predictions.csv",
        index=False,
    #)

pd.DataFrame(results).to_csv(
    output_folder / "preliminary_model_results.csv",
    index=False,
)


print("\nResults saved in the results folder.")
print("These are preliminary cross-validation results.")
