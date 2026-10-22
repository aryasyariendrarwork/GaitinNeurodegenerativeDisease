from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Change this ID whenever you want to inspect another participant.
participant_id = "hunt20"

# Column names, in the order provided by PhysioNet.
columns = [
    "elapsed_time_sec",
    "left_stride_sec",
    "right_stride_sec",
    "left_swing_sec",
    "right_swing_sec",
    "left_swing_pct",
    "right_swing_pct",
    "left_stance_sec",
    "right_stance_sec",
    "left_stance_pct",
    "right_stance_pct",
    "double_support_sec",
    "double_support_pct",
]

# Read the selected participant's original measurements.
data = pd.read_csv(
    f"data/raw/{participant_id}.ts",
    sep=r"\s+",
    header=None,
    names=columns,
)

stride_columns = [
    "elapsed_time_sec",
    "left_stride_sec",
    "right_stride_sec",
]

print(f"\nParticipant: {participant_id}")
print("First five rows of stride measurements:")
print(data[stride_columns].head().to_string(index=False))

print("\nMissing values in each column:")
print(data.isna().sum())

# Check the structure and missing values in all participant files.
print("\nChecking all participants:")

files = sorted(Path("data/raw").glob("*.ts"))
total_missing = 0

for file in files:
    participant = pd.read_csv(file, sep=r"\s+", header=None)

    if participant.shape[1] != len(columns):
        print("Unexpected column count:", file.name, participant.shape[1])

    missing = int(participant.isna().sum().sum())
    total_missing += missing

    if missing > 0:
        print("Missing values:", file.name, missing)

print("Participant files checked:", len(files))
print("Total missing values:", total_missing)

# Calculate preliminary statistics without removing any measurements.
print(f"\nExample features for participant {participant_id}:")

for column in ["left_stride_sec", "right_stride_sec"]:
    print(column)
    print("  Mean (seconds):", round(data[column].mean(), 4))
    print("  Standard deviation (seconds):", round(data[column].std(), 4))
    print("  Minimum (seconds):", round(data[column].min(), 4))
    print("  Maximum (seconds):", round(data[column].max(), 4))

print("\nFive longest left stride intervals:")
print(
    data.nlargest(5, "left_stride_sec")[stride_columns]
    .to_string(index=False)
)

# Plot the selected participant's stride intervals.
plt.figure(figsize=(10, 4))
plt.plot(
    data["elapsed_time_sec"],
    data["left_stride_sec"],
    label="Left",
)
plt.plot(
    data["elapsed_time_sec"],
    data["right_stride_sec"],
    label="Right",
)
plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Stride interval (seconds)")
plt.title(f"Stride intervals for participant {participant_id}")
plt.legend()
plt.tight_layout()

# Save the graph before displaying it.
Path("results").mkdir(exist_ok=True)
graph_path = f"results/{participant_id}_stride_intervals.png"
plt.savefig(graph_path, dpi=300)
print(f"\nGraph saved to {graph_path}")

plt.show()
print("\nMedian and IQR for participant", participant_id)

for column in ["left_stride_sec", "right_stride_sec"]:
    median = data[column].median()
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25)

    print(column)
    print("  Median (seconds):", round(median, 4))
    print("  IQR (seconds):", round(iqr, 4))