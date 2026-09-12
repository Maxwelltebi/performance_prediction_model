# Student Performance Predictor

A single-page web app that predicts a student's final grade (**a**–**f**) from 13 profile,
study-habit and subject-score inputs, served by a FastAPI backend wrapping the Random
Forest classifier trained in [student_performance_model.ipynb](student_performance_model.ipynb).

Model: `RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt')`
— **88% test accuracy** over 25,000 students.

## Layout

```
backend/
  app.py             FastAPI app: /api/predict, /api/schema, /api/health + serves the UI
  features.py        Single source of truth for the model's input contract
  predictor.py       Loads artifacts, preprocesses input, predicts
  schemas.py         Pydantic request/response models
  test_predictor.py  Preprocessing smoke tests
frontend/
  index.html         The single-page interface (no build step, no dependencies)
models/
  best_random_forest_model.joblib
  scaler.joblib
main.py              Original notebook export — see "Known issues" below
```

## Running locally

```powershell
uv venv .venv-vercel --python 3.12
uv pip install --python .venv-vercel/Scripts/python.exe -r requirements.txt
.venv-vercel/Scripts/python.exe -m uvicorn app:app --app-dir backend --reload
```

Then open <http://127.0.0.1:8000>. Interactive API docs are at `/docs`.

Use Python 3.12 with these pinned dependencies. Without `uv`, create a virtual
environment using an installed Python 3.12 interpreter and install `requirements.txt`
with its pip. The older `run.ps1` helper uses `venv/`.

The existing environment can still be started with:

```powershell
venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --reload
```

## Tests

```powershell
uv pip install --python .venv-vercel/Scripts/python.exe -r requirements-dev.txt
.venv-vercel/Scripts/python.exe scripts/build_vercel.py
.venv-vercel/Scripts/python.exe -m pytest backend tests -q
node --test tests/frontend.test.cjs
```

## API

`POST /api/predict`

```json
{
  "age": 18, "gender": "male", "school_type": "private", "parent_education": "phd",
  "study_hours": 7.0, "attendance_percentage": 90.0, "internet_access": "yes",
  "travel_time": "<15 min", "extra_activities": "yes", "study_method": "notes",
  "math_score": 85.0, "science_score": 92.0, "english_score": 88.0
}
```

```json
{
  "grade": "b",
  "label": "Very good",
  "confidence": 0.68,
  "breakdown": [
    { "grade": "b", "label": "Very good", "probability": 0.68 },
    { "grade": "c", "label": "Good", "probability": 0.18 }
  ]
}
```

All 13 fields are required. Categories and numeric ranges are validated, so a bad value
returns `422` with a per-field message rather than a wrong prediction.

`GET /api/schema` returns the field spec the frontend renders its form from, so the form
and the model cannot drift apart. `GET /api/health` reports whether the artifacts loaded.

## Deploying

Follow [the Vercel deployment guide](VERCEL.md). Vercel serves the page and generated
form schema from its CDN, while its Python function runs the bundled model and scaler.
No Hugging Face account, API URL, token, or application environment variable is needed.

Both artifacts in `models/` are committed and included in the function; the forest
is about 70 MB. `requirements.txt` includes both web and ML dependencies. Python 3.12
and sklearn 1.6.1 are selected to run the existing artifacts without re-exporting them.
The first prediction can still incur a function cold start; the page remains static.

## Known issues

**1. `main.py` silently ignores five of the thirteen inputs.** Its `predict_student_grade`
calls `pd.get_dummies(..., drop_first=True)` on a *single-row* DataFrame. With one row each
nominal column holds exactly one category, so `get_dummies` emits one dummy column and
`drop_first` immediately removes it. The subsequent `reindex(..., fill_value=0)` then fills
`gender`, `school_type`, `internet_access`, `extra_activities` and `study_method` with
zeros — the model never sees them.

`backend/predictor.py` encodes those columns against the fixed training category lists in
`features.py` instead, which reproduces what training actually produced. On 300 randomly
generated students the two paths disagree on **10%** of predicted grades.
`main.py` is kept only as the notebook's record; the deployed app does not import it.

**2. The saved scaler was fitted on the test split.** Notebook cell 16 runs
`scaler.fit_transform(X_train)` and then `scaler.fit_transform(X_test)` — the second call
*refits* the same object, so `scaler.joblib` carries the test split's means and variances
rather than the training split's. The two splits are drawn from the same 25,000 rows so the
statistics are very close and predictions are barely affected, but it is a leak worth
correcting. Fixing it means changing the second call to `scaler.transform(X_test)`,
re-running the notebook, and re-exporting both artifacts.
