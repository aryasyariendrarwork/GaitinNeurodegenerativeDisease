"""
Check that a trained model will work in the interface.

Usage (from inside the interface/ folder):
    python3 check_model.py                      # checks model/model.joblib
    python3 check_model.py path/to/some.ts      # also runs it on one recording
"""

import sys
from pathlib import Path

from features import FEATURE_ORDER, compute_features, features_frame, load_ts
from model_slot import MODEL_PATH, load_model

ok = True


def fail(msg):
    global ok
    ok = False
    print("  ✗", msg)


print(f"1. Loading {MODEL_PATH} ...")
if not MODEL_PATH.exists():
    sys.exit("  ✗ No model file found. Save it as model/model.joblib (see model_slot.py).")
try:
    model = load_model()
except Exception as e:
    sys.exit(f"  ✗ Could not load the model: {e}\n"
             "    Usually a scikit-learn version mismatch: install the same version used for training.")
print("  ✓ loaded:", type(model).__name__)

print("2. Checking it can output probabilities ...")
if hasattr(model, "predict_proba"):
    print("  ✓ has predict_proba")
else:
    fail("no predict_proba(). Use a classifier that outputs probabilities "
         "(e.g. LogisticRegression, RandomForestClassifier, or SVC(probability=True)).")

classes = getattr(model, "classes_", None)
print("3. Checking classes ...")
if classes is not None:
    print("  classes:", list(classes))
    if list(classes) != [0, 1]:
        fail("expected classes [0, 1] with 1 = disease. The app reads column 1 of predict_proba.")
    else:
        print("  ✓ [0, 1] (1 = disease)")

print("4. Checking the features match features.py ...")
n = getattr(model, "n_features_in_", None)
names = getattr(model, "feature_names_in_", None)
if n is not None and n != len(FEATURE_ORDER):
    fail(f"model expects {n} features, the app provides {len(FEATURE_ORDER)}: {FEATURE_ORDER}")
if names is not None:
    names = list(names)
    if names != FEATURE_ORDER:
        fail("feature names/order differ from features.FEATURE_ORDER")
        print("     model:", names)
        print("     app:  ", FEATURE_ORDER)
    else:
        print(f"  ✓ same {len(FEATURE_ORDER)} features, same order")
elif n == len(FEATURE_ORDER):
    print("  ? model was trained without column names. Make sure its column ORDER was:", FEATURE_ORDER)

if len(sys.argv) > 1:
    ts = Path(sys.argv[1])
    print(f"5. Test prediction on {ts.name} ...")
    try:
        df, warnings = load_ts(ts.read_bytes())
        p = model.predict_proba(features_frame(compute_features(df)))[0, 1]
        print(f"  ✓ probability of disease-group pattern: {p:.3f}")
    except Exception as e:
        fail(f"prediction failed: {e}")

print("\nRESULT:", "ready — run the app" if ok else "fix the ✗ items above before using the model in the app")
