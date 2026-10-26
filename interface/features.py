"""
Gait feature extraction and input validation for the interface.

IMPORTANT: the model must be trained on features computed by EXACTLY this code.
If the team changes how features are computed for training, change this file
too (or import the training code here), otherwise the app will feed the model
different numbers from the ones it learned on.

The 8 features are the team's stride-time features (as in the team's
data/processed/gait_features.csv): mean, SD, median and IQR of the stride
interval, for the left and right foot separately, computed from the raw .ts
file without filtering.
"""

import io

import pandas as pd

COLUMNS = [
    "time_s",
    "left_stride_s", "right_stride_s",
    "left_swing_s", "right_swing_s",
    "left_swing_pct", "right_swing_pct",
    "left_stance_s", "right_stance_s",
    "left_stance_pct", "right_stance_pct",
    "double_support_s", "double_support_pct",
]

# Order matters: the model receives the features as columns in this order,
# with exactly these names (the same column names as gait_features.csv).
FEATURE_ORDER = [
    "left_stride_mean_sec",
    "left_stride_sd_sec",
    "left_stride_median_sec",
    "left_stride_iqr_sec",
    "right_stride_mean_sec",
    "right_stride_sd_sec",
    "right_stride_median_sec",
    "right_stride_iqr_sec",
]

# Human-readable labels for the interface: (name, description, unit, decimals).
FEATURE_INFO = {
    "left_stride_mean_sec": ("Left stride time (mean)", "Average duration of a left stride", " s", 2),
    "left_stride_sd_sec": ("Left stride variability (SD)", "Spread of left stride times", " s", 3),
    "left_stride_median_sec": ("Left stride time (median)", "Typical duration of a left stride", " s", 2),
    "left_stride_iqr_sec": ("Left stride variability (IQR)", "Spread of the middle 50% of left strides", " s", 3),
    "right_stride_mean_sec": ("Right stride time (mean)", "Average duration of a right stride", " s", 2),
    "right_stride_sd_sec": ("Right stride variability (SD)", "Spread of right stride times", " s", 3),
    "right_stride_median_sec": ("Right stride time (median)", "Typical duration of a right stride", " s", 2),
    "right_stride_iqr_sec": ("Right stride variability (IQR)", "Spread of the middle 50% of right strides", " s", 3),
}

MIN_STRIDES = 50            # fewer strides -> variability estimates are unreliable
MAX_FILE_BYTES = 1_000_000  # 1 MB; a 5-minute recording is ~20 KB
SENSOR_FAILURE_STRIDE_S = 3.0


class GaitFileError(ValueError):
    """Raised when an uploaded file cannot be analysed. The message is shown to the user."""


# ---------------------------------------------------------------- validation --
def load_ts(raw_bytes):
    """Parse and validate a .ts file. Returns (dataframe, list of warning strings)."""
    if len(raw_bytes) == 0:
        raise GaitFileError("The file is empty.")
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise GaitFileError(f"The file is too large ({len(raw_bytes) / 1e6:.1f} MB). "
                            "A gait .ts recording is usually under 100 KB.")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise GaitFileError("This is not a plain-text file. Please upload a gait .ts text file.")

    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    except Exception:
        raise GaitFileError("The file could not be read as a table of numbers.")

    if df.shape[1] != len(COLUMNS):
        raise GaitFileError(f"Unexpected format: this file has {df.shape[1]} columns per row; "
                            f"a gait .ts file has {len(COLUMNS)}. Check that you uploaded the right file.")
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad = int(df.isna().any(axis=1).sum())
        raise GaitFileError(f"{bad} row(s) contain text or missing values. Every value must be a number.")

    if len(df) < MIN_STRIDES:
        raise GaitFileError(f"Recording too short: only {len(df)} strides were found; "
                            f"at least {MIN_STRIDES} are needed for reliable variability measures.")

    warnings = []
    for foot in ["left", "right"]:
        if df[f"{foot}_stride_s"].median() > SENSOR_FAILURE_STRIDE_S:
            raise GaitFileError(f"The {foot}-foot sensor appears faulty (stride times above "
                                f"{SENSOR_FAILURE_STRIDE_S:.0f} s throughout). The model needs both feet, "
                                "so this recording cannot be analysed.")

    n_negative = int((df[COLUMNS[1:]] < 0).any(axis=1).sum())
    if n_negative > 0.2 * len(df):
        raise GaitFileError(f"{n_negative} of {len(df)} strides contain negative time values, "
                            "which suggests a recording problem.")
    if n_negative:
        warnings.append(f"{n_negative} stride(s) contain negative values; they are included "
                        "as recorded, as in the training data.")
    return df, warnings


# ------------------------------------------------------------------ features --
def _stats(x):
    """Mean, SD (n-1), median and IQR (linear quantiles), pandas defaults.
    save_model.py --data-dir checks these match the training CSV."""
    return x.mean(), x.std(), x.median(), x.quantile(0.75) - x.quantile(0.25)


def compute_features(df):
    """Return the 8 stride features for one participant as a dict."""
    feats = {}
    for foot in ["left", "right"]:
        mean, sd, median, iqr = _stats(df[f"{foot}_stride_s"])
        feats[f"{foot}_stride_mean_sec"] = mean
        feats[f"{foot}_stride_sd_sec"] = sd
        feats[f"{foot}_stride_median_sec"] = median
        feats[f"{foot}_stride_iqr_sec"] = iqr
    return feats


def features_frame(feature_dict):
    """One-row DataFrame in FEATURE_ORDER, ready for model.predict_proba()."""
    return pd.DataFrame([[feature_dict[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
