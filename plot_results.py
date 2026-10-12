from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

results_folder = Path("results")

datasets = {
    "all_participants": "All 64 participants",
    "without_hunt20": "Without hunt20 (63 participants)",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (name, title) in zip(axes, datasets.items()):
    predictions = pd.read_csv(
        results_folder / f"{name}_predictions.csv"
    )

    matrix = confusion_matrix(
        predictions["disease"],
        predictions["logistic_regression_prediction"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Control", "Disease"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        im_kw={"vmin": 0, "vmax": 48},
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted group")
    ax.set_ylabel("Actual group")

fig.suptitle(
    "Logistic regression: preliminary cross-validation predictions"
)
fig.tight_layout()

output_path = results_folder / "confusion_matrices.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Graph saved to {output_path}")
plt.show()