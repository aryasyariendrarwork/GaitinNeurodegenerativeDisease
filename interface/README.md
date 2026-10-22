# Gait Pattern Analysis — interface (course prototype)

> **Course prototype for research and education only. Not a medical device. Not for clinical use or diagnosis.**

A Streamlit web app for the EXSU 500 team project. A user uploads one participant's
gait recording (`.ts` file from the PhysioNet Gait in Neurodegenerative Disease
Database), and the app shows:

1. **Prediction:** whether the walking pattern more closely resembles the
   neurodegenerative-disease group or healthy controls, as a probability with a
   95% uncertainty range (from resampling the participant's strides).
2. **Feature comparison:** each of the 9 gait features against the healthy-control range.
3. **Ranking:** which features drove this prediction.
4. **Model reliability and limitations.**

## Run it on your computer

```bash
cd interface
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Press **Ctrl + C** in the terminal to stop it.

## Files

| File | What it does |
|---|---|
| `app.py` | The website: the 3 screens (Welcome → Input → Results) |
| `features.py` | Validates uploaded files and computes the 9 gait features |
| `model_slot.py` | ⭐ **Where the trained model plugs in** (prediction, uncertainty, ranking) |
| `model/` | Put the trained model here as `model.joblib` |
| `reference_ranges.json` | Healthy-control ranges used in the comparison chart |
| `examples/` | Optional example `.ts` recordings for the dropdown |
| `.streamlit/config.toml` | Upload limit (1 MB) and colours |
| `requirements.txt` | Python packages |

## Connecting the model

Until `model/model.joblib` exists, the app runs in **demo mode**: the feature
values are real, but the probability, its range and the ranking are clearly
labelled placeholders. To connect the real model:

1. Train the final model on the 9 features produced by `features.compute_features()`,
   with columns in the order of `features.FEATURE_ORDER`. It must be a scikit-learn
   estimator or `Pipeline` with `.predict_proba()`, where class 1 = disease.
   If any feature can be missing (e.g. a failed sensor), include an imputer in the pipeline.
2. Save it:
   ```python
   import joblib
   joblib.dump(final_model, "interface/model/model.joblib")
   ```
3. Copy `model/model_info.json.example` to `model/model_info.json` and fill in the
   **held-out test-set** results. The example numbers are placeholders, not results.
4. Recompute `reference_ranges.json` from the **training-set controls** only.

**Important:** `features.py` must compute features exactly as the training code
does. If the team changes the feature code, update this file too.

## Example recordings

To enable the example dropdown, copy a few `.ts` files from the **held-out test set**
into `examples/`. Check the dataset's licence on PhysioNet before committing data
files; the course asks teams not to commit the dataset itself.

## Deploying (optional +2% bonus)

Push this folder to GitHub, then connect the repository to a free host that runs
Python apps (e.g. Streamlit Community Cloud or Hugging Face Spaces; check
myCourses for the recommended option), choosing `interface/app.py` as the entry
point. The deployed app must run the real trained model, show the prototype notice
(already on every screen) and stay online until at least May 1, 2027.

## Data source

Hausdorff JM et al. *Gait in Neurodegenerative Disease Database* (v1.0.0). PhysioNet.
https://physionet.org/content/gaitndd/1.0.0/
