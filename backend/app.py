"""FastAPI service exposing the student grade model, plus the single-page UI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from features import form_schema
from predictor import GradePredictor
from schemas import PredictionResponse, StudentInput

logger = logging.getLogger("uvicorn.error")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the artifacts once at startup rather than per request."""
    try:
        state["predictor"] = GradePredictor()
        logger.info("Model and scaler loaded successfully.")
    except Exception as exc:  # keep serving /api/health so the UI can explain why
        state["error"] = str(exc)
        logger.error("Could not load model artifacts: %s", exc)
    yield


app = FastAPI(
    title="Student Performance API",
    description="Predicts a student's final grade (a-f) from 13 profile, habit and score inputs.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_predictor() -> GradePredictor:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {state['error']}",
        )
    return predictor


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if state["predictor"] else "degraded",
        "model_loaded": state["predictor"] is not None,
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
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result)


# Serve the single-page UI from the same origin, so no CORS setup is needed.
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
