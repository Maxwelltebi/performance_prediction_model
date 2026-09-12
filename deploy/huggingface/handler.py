"""HF custom handler; prepare_hf_release.py includes the shared backend code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from predictor import GradePredictor
from schemas import StudentInput


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        self.predictor = GradePredictor(Path(path or ".") / "models")

    def __call__(self, data: dict) -> dict:
        student = StudentInput.model_validate(data["inputs"])
        return self.predictor.predict(student.model_dump())
