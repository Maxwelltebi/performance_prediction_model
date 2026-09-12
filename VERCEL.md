# Deploy the portfolio app on Vercel

Vercel serves the HTML and form schema from its CDN. Submitting the form calls a
lightweight FastAPI function, which sends validated inputs to an authenticated
Hugging Face Inference Endpoint. The model and scaler run on HF.

## What is prepared

- `server.py` exports the FastAPI application. `pyproject.toml` selects it explicitly,
  so Vercel does not run the original notebook export in `main.py`.
- `requirements.txt` includes only the web runtime. Use `requirements-local.txt`
  for local inference and `requirements-dev.txt` for all tests.
- `.python-version` selects Python 3.14 for the web function. HF's Python and
  sklearn versions are configured separately in its serving environment.
- The build command in `pyproject.toml` runs `scripts/build_vercel.py`, copying the
  page into `public/index.html` and generating `public/form-schema.json` from
  `backend/features.py`. These generated files do not require a running model.
- `vercel.json` routes `/` to the static page and gives the function 120 seconds.
  `.vercelignore` and function exclusions keep artifacts, virtual environments,
  training notebooks, and tests out of the deployment.
- On Vercel, missing HF configuration produces a 503; it never falls back to
  loading the local model. Local `run.ps1` still supports local inference.
- The UI shows a message after five seconds of waiting and aborts after 90 seconds.
  The HF client uses a 10-second connect timeout and a 60-second read timeout.
  The form recovers after failures and allows another submission.

## 1. Have an HF inference endpoint ready

Follow [HUGGING_FACE.md](HUGGING_FACE.md) for the model repository and endpoint.
Vercel needs the **inference endpoint URL**, not the model repository URL.
The endpoint must use this project's custom handler and return its grade response.

The existing model and scaler are unchanged. Local tests still warn that the
artifacts were saved with sklearn 1.6.1, while the local environment has 1.9.0.
Validate the HF serving environment separately as described in the HF guide.

## 2. Import the Git repository

Commit and push the prepared source to your Git host, then import the repository
at <https://vercel.com/new>. Use these settings:

| Setting | Value |
| --- | --- |
| Framework Preset | FastAPI |
| Root Directory | Repository root |
| Build Command | Leave the override off; uses `python scripts/build_vercel.py` |
| Install Command | Leave the override off; uses `requirements.txt` |
| Output Directory | Leave the override off; FastAPI integration handles `public/` |

There is no Uvicorn start command to configure on Vercel. The platform invokes
`server:app`. Keep Fluid compute enabled. The generated public directory is ignored
by Git intentionally: the Vercel build creates it from the source on every deploy.

## 3. Set server-side environment variables

In the Vercel project's Environment Variables settings, add:

| Name | Value |
| --- | --- |
| `HF_ENDPOINT_URL` | `https://YOUR-ENDPOINT.REGION.PROVIDER.endpoints.huggingface.cloud` |
| `HF_TOKEN` | A token authorized to call your protected HF endpoint |

Mark the token as sensitive when available. Set values for Production and, if you
want live predictions in preview deployments, Preview. Do not put tokens in the
frontend or commit them to Git. No browser-visible environment variable is needed.
Deploy/redeploy after setting the variables.

## 4. Verify before putting the URL on your portfolio

1. Check the build log for successful static file generation and Python deployment.
2. Open the site and confirm that all 13 fields appear. The browser should request
   `/form-schema.json` on page load and no `/api/*` endpoints until submitting.
3. Load the sample values and submit. Compare the result with the same inputs sent
   directly to HF. The browser must communicate with your Vercel domain only for
   predictions; the HF token must not appear in browser requests or page source.
4. `/api/health` reports web-client readiness, not live HF model availability.
   A real prediction verifies the full connection. Invalid input returns 422;
   HF timeouts, bad responses and unavailability return 503.
5. Test the site again after inactivity. CDN delivery avoids a Python startup wait
   for the page, but the Vercel function can still cold-start on prediction.
6. Ensure the production link opens in a signed-out/incognito browser. Review any
   Vercel deployment protection setting that would require a recruiter to log in.

If HF scales to zero, a recruiter can still wait for the model to start. For steady
prediction availability, disable HF scale-to-zero and retain at least one replica;
this incurs ongoing HF compute charges. No always-warm compute guarantee is implied
by using Vercel. Review request limits and spending settings before sharing publicly.

## Local checks

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe scripts/build_vercel.py
.\venv\Scripts\python.exe -m pytest backend tests -q
node --test tests/frontend.test.cjs
```

`run.ps1` serves the same frontend locally. The local FastAPI route
`/form-schema.json` supplies the schema without requiring a generated public build.
On Vercel the generated static file is served at that URL by the CDN.

These checks validate local code and build output. A Vercel cloud build, CDN routing,
and authenticated HF prediction must still be verified in the actual deployment.

Sources: [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi),
[Python runtime](https://vercel.com/docs/functions/runtimes/python),
[Fluid compute](https://vercel.com/docs/fluid-compute),
[HF autoscaling](https://huggingface.co/docs/inference-endpoints/guides/autoscaling).
