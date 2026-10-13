from pathlib import Path
import pandas as pd

files = sorted(Path("data/raw").glob("*.ts"))

if not files:
    raise FileNotFoundError("No .ts files found in data/raw.")

rows = []

for file in files:
    data = pd.read_csv(file, sep=r"\s+", header=None)

    if data.shape[1] != 13:
        raise ValueError(f"{file.name}: expected 13 columns.")

    # Each file represents one participant.
    row = {"participant_id": file.stem}

    for side, column in [("left", 1), ("right", 2)]:
        strides = data[column]

        row[f"{side}_stride_mean_sec"] = strides.mean()
        row[f"{side}_stride_sd_sec"] = strides.std()
        row[f"{side}_stride_median_sec"] = strides.median()
        row[f"{side}_stride_iqr_sec"] = (
            strides.quantile(0.75) - strides.quantile(0.25)
        )

    rows.append(row)

features = pd.DataFrame(rows)

# Add group labels and the binary classification target.
features["group"] = features["participant_id"].str.extract(
    r"^(als|park|hunt|control)", expand=False
)

if features["group"].isna().any():
    raise ValueError("Some participant IDs have an unknown group.")

features["disease"] = (features["group"] != "control").astype(int)

# Save the complete table.
output_path = Path("data/processed/gait_features.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(output_path, index=False)

print("Participants summarized:", len(features))
print("\nParticipants in each group:")
print(features["group"].value_counts())

print("\nFeatures for als12:")
print(
    features.loc[
        features["participant_id"] == "als12"
    ].to_string(index=False)
)

print(f"\nSaved to {output_path}")
print("Columns:", features.columns.tolist())