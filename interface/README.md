# Gait Pattern Analysis — interface (course prototype)

> **Course prototype for research and education only. Not a medical device. Not for clinical use or diagnosis.**

A Streamlit web app for the EXSU 500 team project. A user uploads one participant's
gait recording (`.ts` file from the PhysioNet Gait in Neurodegenerative Disease
Database), and the app shows:

1. **Prediction:** whether the walking pattern more closely resembles the
   neurodegenerative-disease group or healthy controls, as a probability with a
   95% uncertainty range (from resampling the participant's strides).
2. **Feature comparison:** each of the 8 stride features against the healthy-control range.
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
| `features.py` | Validates uploaded files and computes the 8 stride features |
| `model_slot.py` | ⭐ **Where the trained model plugs in** (prediction, uncertainty, ranking) |
| `save_model.py` | Trains and saves the final random forest, its test results and healthy ranges |
| `check_model.py` | Checks a saved model fits the app |
| `model/` | Put the trained model here as `model.joblib` |
| `reference_ranges.json` | Healthy-control ranges used in the comparison chart |
| `examples/` | Optional example `.ts` recordings for the dropdown |
| `.streamlit/config.toml` | Upload limit (1 MB) and colours |
| `requirements.txt` | Python packages |

## Connecting the model

The app uses the team's final model: a **random forest on the 8 stride features**
(mean, SD, median and IQR of stride time for each foot), trained without hunt20
on the same stratified 80/20 split (`random_state=42`) as the report.

From inside `interface/`, run once:

```bash
python3 save_model.py --data-dir PATH/TO/gait-in-neurodegenerative-disease-database-1.0.0
python3 check_model.py PATH/TO/park1.ts
```

`save_model.py` reads `../data/processed/gait_features.csv`, trains the random forest
on the 80% training split, tests it **once** on the held-out 20%, and writes:
- `model/model.joblib`: the trained model
- `model/model_info.json`: the held-out test results shown on the reliability card
- `reference_ranges.json`: healthy ranges from the **training-set** controls only

`--data-dir` (optional) checks that `features.py` reproduces the CSV's features exactly.
`check_model.py` confirms the saved model fits the app. Without `model/model.joblib`
the app falls back to demo mode with labelled placeholders.

Files with a faulty foot sensor (like hunt20) are rejected with a message, because
the model was trained without missing values.

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
