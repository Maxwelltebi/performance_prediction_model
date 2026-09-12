# Deploy the complete app on Vercel

The frontend, API, model, and scaler deploy together on Vercel. No Hugging Face
service, external prediction API, application environment variables, or tokens are required.

```text
Browser -> Vercel CDN: page and form schema
Browser -> /api/predict -> FastAPI -> scaler -> Random Forest -> JSON -> Browser
```

## Deployment configuration

- `server.py` exports FastAPI; `pyproject.toml` selects it instead of the notebook
  export in `main.py`.
- `.python-version` selects Python 3.12. `requirements.txt` includes the web and ML
  dependencies, with sklearn 1.6.1 matching the saved model and scaler.
- Both artifacts in `models/` are tracked by Git and included in the function.
  They are not copied into `public/` or exposed as website downloads.
- The build runs `scripts/build_vercel.py`, creating `public/index.html` and
  `public/form-schema.json`. Opening the form does not require model initialization.
- FastAPI loads the bundled artifacts once per function instance. There are no
  model downloads or remote prediction calls.
- `/api/health` returns 200 when the model is loaded and 503 if startup fails.
  Invalid prediction input returns 422.
- The function duration limit is 120 seconds. The UI displays a waiting message
  after five seconds and restores the submit button after failures or a 90-second timeout.

## Deploy

1. Commit and push the project, including `models/best_random_forest_model.joblib`
   and `models/scaler.joblib`, to your Git host. The model is about 70 MB.
2. Import the repository at <https://vercel.com/new> with these settings:

| Setting | Value |
| --- | --- |
| Framework Preset | FastAPI |
| Root Directory | Repository root |
| Build Command | Leave override off; uses `python scripts/build_vercel.py` |
| Install Command | Leave override off; uses `requirements.txt` |
| Output Directory | Leave override off; integration handles `public/` |
| Application environment variables | None |

3. If you previously entered `HF_ENDPOINT_URL` or `HF_TOKEN`, remove them from
   Vercel. They are no longer used. No HF repository or endpoint is needed.
4. Deploy with Fluid compute enabled. Vercel invokes `server:app`; there is no
   Uvicorn start command. Check dependency installation and static build logs.

## Verify the deployed app

- Open the production URL in a signed-out browser and check that deployment
  protection does not require recruiters to sign in.
- All 13 fields should appear. The page fetches `/form-schema.json`, not an API
  health check. Only submitting the form should invoke `/api/predict`.
- Load the sample and submit: the current artifacts return grade `b` and confidence
  `0.68`. Check `/api/health` for `model_loaded: true` if prediction fails.
- Revisit after inactivity and measure the first prediction. CDN delivery keeps
  the page independent of Python startup, but Vercel can still cold-start the API
  and reload the model. This does not promise instant first predictions.
- Check function size and memory in deployment output. The 70 MB model does not
  represent total bundle size or runtime RAM. The standard Python bundle limit is
  500 MB uncompressed; Vercel's cloud build is the final packaging check.

## Local environment and checks

Use Python 3.12 for the pinned dependencies. The previous Python 3.14 / sklearn 1.9
virtual environment is left untouched. With uv installed:

```powershell
uv venv .venv-vercel --python 3.12
uv pip install --python .venv-vercel/Scripts/python.exe -r requirements-dev.txt
.venv-vercel/Scripts/python.exe -m uvicorn app:app --app-dir backend --reload
```

Without uv, use an installed Python 3.12 interpreter to create a virtual environment
and install `requirements-dev.txt` with its pip. `requirements-local.txt` is an alias
of `requirements.txt` for older commands. The older `run.ps1` uses `venv/`.

```powershell
.venv-vercel/Scripts/python.exe scripts/build_vercel.py
.venv-vercel/Scripts/python.exe -m pytest backend tests -q
node --test tests/frontend.test.cjs
```

The model and scaler are unchanged. The known notebook scaler-fit issue remains;
switching hosting does not require retraining and does not correct model quality.
Local tests do not replace a Vercel cloud build and hosted prediction check.

Verified locally with Python 3.12.14 and sklearn 1.6.1: 14 Python tests and four
frontend tests pass, including a real prediction through the Vercel entrypoint
without HF configuration. No sklearn artifact-version warnings were emitted.
Linux x86_64 Python 3.12 wheels resolved successfully; their unpacked contents plus
both model artifacts total approximately 345.6 MB before source, bytecode and runtime
overhead. This is a packaging estimate, not a measured Vercel bundle or RAM usage.

Sources: [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi),
[Python runtime and packaging](https://vercel.com/docs/functions/runtimes/python).
