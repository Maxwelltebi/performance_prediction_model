"""FastAPI service exposing the student grade model, plus the single-page UI."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from features import form_schema
from remote_predictor import RemoteGradePredictor, RemotePredictionUnavailable
from schemas import PredictionResponse, StudentInput

logger = logging.getLogger("uvicorn.error")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize local inference or the remote client once per process."""
    state.update(predictor=None, error=None)
    state["mode"] = "remote" if os.getenv("HF_ENDPOINT_URL") else "local"
    try:
        if state["mode"] == "remote":
            state["predictor"] = RemoteGradePredictor(
                os.environ["HF_ENDPOINT_URL"], os.getenv("HF_TOKEN", "")
            )
        else:
            from predictor import GradePredictor

            state["predictor"] = GradePredictor()
        logger.info("Prediction backend initialized (%s).", state["mode"])
    except Exception as exc:  # keep serving /api/health so the UI can explain why
        state["error"] = "Prediction backend could not initialize. Check server configuration."
        logger.error("Prediction backend initialization failed (%s).", type(exc).__name__)
    try:
        yield
    finally:
        if isinstance(state["predictor"], RemoteGradePredictor):
            state["predictor"].close()
        state["predictor"] = None


app = FastAPI(
    title="Student Performance API",
    description="Predicts a student's final grade (a-f) from 13 profile, habit and score inputs.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_predictor():
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {state['error']}",
        )
    return predictor


@app.get("/api/health")
def health(response: Response) -> dict:
    ready = state["predictor"] is not None
    response.status_code = 200 if ready else 503
    return {
        "status": "ok" if ready else "degraded",
        "mode": state.get("mode", "local"),
        "model_loaded": ready if state.get("mode") != "remote" else None,
        "backend_ready": ready,
        "remote_model_status": "not_probed" if state.get("mode") == "remote" else None,
        "error": state["error"],
    }


@app.get("/api/schema")
def schema() -> dict:
    """Field spec the frontend renders its form from."""
    return form_schema()


@app.post("/api/predict", response_model=PredictionResponse)
def predict(student: StudentInput) -> PredictionResponse:
    predictor = _require_predictor()
    try:
        result = predictor.predict(student.model_dump())
    except RemotePredictionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.") from exc
    return PredictionResponse(**result)


# Serve the single-page UI from the same origin, so no CORS setup is needed.
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
