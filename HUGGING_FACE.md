# Hugging Face deployment walkthrough

This setup separates the web application from model inference:

For the current Vercel web deployment, use [VERCEL.md](VERCEL.md) after completing
the HF steps below. Render remains an alternative web host described in step 5.

```text
Browser -> Render FastAPI app -> Hugging Face Inference Endpoint
                                      |
                             pinned HF model repository
```

The Hub repository stores versioned files. The dedicated endpoint loads those files
and executes preprocessing and prediction. Uploading a joblib file alone does not
create an inference API. Render uses HTTPS and does not load the ML dependencies
when `HF_ENDPOINT_URL` is configured. Local mode remains available without that variable.

## 1. Prepare a release locally

From the project root, run:

```powershell
.\venv\Scripts\python.exe scripts/prepare_hf_release.py
```

This creates `.hf-release/` with exactly eight files: both artifacts in `models/`,
the shared predictor, features and schemas in `backend/`, plus `handler.py`,
`requirements.txt` and a model card at the root. It does not upload anything.
It refuses to overwrite an existing release directory; move a previous release
aside before preparing a replacement. Recreate the release after changing code or models.

## 2. Validate the inference environment

The existing artifacts were saved with sklearn 1.6.1, while the project's Windows
environment runs Python 3.14 / sklearn 1.9.0 and emits compatibility warnings.
The release requirements are a candidate Python 3.11 environment using sklearn
1.6.1; the rest of the original training versions are unknown. Do not install
these older requirements into the existing Python 3.14 environment.

Use a separate Python 3.11 environment matching the selected HF serving image and
install `.hf-release/requirements.txt`. Before paying for an endpoint, load
`EndpointHandler(path='.hf-release')` from `.hf-release/handler.py` and call it
with `{'inputs': <the README sample payload>}`. Check the grade/probability response
and dependency warnings. Local tests with Python 3.14 do not validate that image.

For a reliable model release, fix the notebook's `fit_transform(X_test)` to
`transform(X_test)`, rerun training and evaluation, export both artifacts, and pin
the actual environment used. Do not simply resave the old model under a newer
sklearn version. Training data is not included in this checkout.

## 3. Upload a private HF model repository

Create a **Model** repository at <https://huggingface.co/new>, for example
`YOUR_USERNAME/student-performance`, initially private. Install the HF CLI:

```powershell
.\venv\Scripts\python.exe -m pip install huggingface_hub
.\venv\Scripts\hf.exe auth login
.\venv\Scripts\hf.exe upload YOUR_USERNAME/student-performance .hf-release . --repo-type model
```

Enter a write-scoped HF token in the local login prompt, not in chat or source files.
Only the staged release is uploaded. Check its files in the browser, then copy the
full commit SHA from the repository history. This identifies the exact model/code
release to deploy. Existing Git-tracked models are left intact; you can remove
them from Git tracking after the remote deployment works.

## 4. Deploy inference on HF

Dedicated Inference Endpoints are billed resources. Review the current cost in
<https://huggingface.co/docs/inference-endpoints/pricing> before creating one.

In <https://endpoints.huggingface.co/>, select the model repository and the tested
commit revision. Use the custom handler serving option, authenticated (protected)
access, and CPU hardware for this sklearn forest. GPU hardware will not accelerate
this standard sklearn implementation. Select a serving image compatible with the
validated Python/dependency environment; check its build and startup logs.

Wait for the endpoint to become ready. Test it with the README sample wrapped in
`{"inputs": {...}}`; expect the same grade and probabilities as the local handler.
Record the endpoint HTTPS URL. Configure idle scale-to-zero if appropriate, and
pause/delete the endpoint when finished experimenting; review billing behavior in
the current HF dashboard. Cold starts can temporarily return 503 or exceed the
web app's timeout, in which case the UI asks the user to retry.

## 5. Connect Render

Configure one Python web service with the repository root as its root directory:

| Setting | Value |
| --- | --- |
| Build | `pip install -r requirements-web.txt` |
| Start | `python -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port $PORT --workers 1` |
| Python | Pin the validated web runtime (currently tested locally with 3.14.7) |
| Health check | `/api/health` |
| `HF_ENDPOINT_URL` | Your dedicated `https://....endpoints.huggingface.cloud` URL |
| `HF_TOKEN` | Secret token authorized to call that protected endpoint |

Use a separate narrowly scoped inference token for the web app; it does not need
upload rights. The browser calls your own `/api/predict`, so the HF token stays on
the server and frontend CORS changes are unnecessary. Avoid sharing the upload token.

`/api/health` checks web backend initialization, not remote model availability.
In remote mode it reports `model_loaded: null` and `remote_model_status: not_probed`.
This avoids waking a sleeping paid endpoint on every health probe. Check remote
availability with an actual sample prediction after deployments. Remote failures
and timeouts return a generic 503; the client does not retry automatically or expose
upstream response bodies to visitors. Local initialization failures also return 503.

## 6. Verify and release

- Run `venv/Scripts/python.exe -m pytest backend -q` locally.
- Verify the HF handler in its target environment and through the deployed endpoint.
- Verify the hosted page, schema, valid prediction, invalid input (422), and remote
  outage response (503). Check request latency and CPU/RAM before resizing.
- Before a widely shared launch, set a request budget/rate limit appropriate for
  the paid endpoint; a public prediction form can generate paid inference traffic.
- Future releases: retrain/evaluate, stage, upload, test a pinned revision, then
  update the endpoint. Record the old revision for rollback. Changing HF hardware
  later does not require moving ML libraries back into Render.

Hugging Face sources: [custom handlers](https://huggingface.co/docs/inference-endpoints/guides/custom_handler),
[uploads](https://huggingface.co/docs/huggingface_hub/guides/upload),
[endpoint configuration](https://huggingface.co/docs/inference-endpoints/guides/advanced).
