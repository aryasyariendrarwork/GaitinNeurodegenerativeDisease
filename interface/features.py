"""
Gait feature extraction and input validation for the interface.

IMPORTANT: the model must be trained on features computed by EXACTLY this code.
If the team changes how features are computed for training, change this file
too (or import the training code here), otherwise the app will feed the model
different numbers from the ones it learned on.

The 9 features are outlier-resistant (median / IQR based) summaries of one
participant's walk, as in robust_9_features.py.
"""

import io

import numpy as np
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

# Order matters: the model receives the features as columns in this order.
FEATURE_ORDER = [
    "stride_time_median_s",
    "stride_time_rcv",
    "swing_pct_median",
    "swing_time_rcv",
    "stance_time_rcv",
    "double_support_pct_median",
    "double_support_rcv",
    "stride_asymmetry_pct",
    "swing_asymmetry_pct",
]

# Human-readable labels for the interface.
FEATURE_INFO = {
    "stride_time_median_s": ("Stride time", "Typical duration of one stride", " s", 2),
    "stride_time_rcv": ("Stride time variability", "Stride-to-stride irregularity", "%", 1),
    "swing_pct_median": ("Swing phase", "Share of stride with the foot in the air", "%", 1),
    "swing_time_rcv": ("Swing time variability", "Irregularity of time in the air", "%", 1),
    "stance_time_rcv": ("Stance time variability", "Irregularity of time on the ground", "%", 1),
    "double_support_pct_median": ("Double support", "Share of stride with both feet down", "%", 1),
    "double_support_rcv": ("Double support variability", "Irregularity of both-feet-down time", "%", 1),
    "stride_asymmetry_pct": ("Left–right stride asymmetry", "Difference between left and right strides", "%", 1),
    "swing_asymmetry_pct": ("Left–right swing asymmetry", "Difference in left and right swing share", "%", 1),
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
    df, failed = remove_failed_sensor(df)
    for foot in failed:
        warnings.append(f"The {foot}-foot sensor appears faulty (stride times above "
                        f"{SENSOR_FAILURE_STRIDE_S:.0f} s throughout). The analysis uses the other "
                        "foot only, and asymmetry and double-support features are unavailable.")
    if len(failed) == 2:
        raise GaitFileError("Both foot sensors appear faulty, so this recording cannot be analysed.")

    n_negative = int((df[COLUMNS[1:]] < 0).any(axis=1).sum())
    if n_negative > 0.2 * len(df):
        raise GaitFileError(f"{n_negative} of {len(df)} strides contain negative time values, "
                            "which suggests a recording problem.")
    if n_negative:
        warnings.append(f"{n_negative} stride(s) contain negative values. The median-based "
                        "features limit their influence.")
    return df, warnings


def remove_failed_sensor(df):
    """A foot whose median stride is > 3 s has a broken sensor for the whole walk.
    Set that foot, and double support (which needs both feet), to missing."""
    df = df.copy()
    failed = []
    for foot in ["left", "right"]:
        if df[f"{foot}_stride_s"].median() > SENSOR_FAILURE_STRIDE_S:
            cols = [c for c in COLUMNS if c.startswith(foot)] + ["double_support_s", "double_support_pct"]
            df[cols] = np.nan
            failed.append(foot)
    return df, failed


# ------------------------------------------------------------------ features --
def _iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def _robust_cv(x):
    """IQR / median x 100: outlier-resistant measure of variability (%)."""
    if x.notna().sum() == 0:
        return np.nan
    return 100 * _iqr(x) / x.median()


def _both_feet(left, right):
    vals = [v for v in (left, right) if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def compute_features(df):
    """Return the 9 features for one participant as a dict (NaN where unavailable)."""
    stride = df[["left_stride_s", "right_stride_s"]].mean(axis=1)
    return {
        "stride_time_median_s": stride.median(),
        "stride_time_rcv": _both_feet(_robust_cv(df.left_stride_s), _robust_cv(df.right_stride_s)),
        "swing_pct_median": _both_feet(df.left_swing_pct.median(), df.right_swing_pct.median()),
        "swing_time_rcv": _both_feet(_robust_cv(df.left_swing_s), _robust_cv(df.right_swing_s)),
        "stance_time_rcv": _both_feet(_robust_cv(df.left_stance_s), _robust_cv(df.right_stance_s)),
        "double_support_pct_median": df.double_support_pct.median(),
        "double_support_rcv": _robust_cv(df.double_support_s),
        "stride_asymmetry_pct": 100 * ((df.left_stride_s - df.right_stride_s).abs() / stride).median(),
        "swing_asymmetry_pct": abs(df.left_swing_pct.median() - df.right_swing_pct.median()),
    }


def features_frame(feature_dict):
    """One-row DataFrame in FEATURE_ORDER, ready for model.predict_proba()."""
    return pd.DataFrame([[feature_dict[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
