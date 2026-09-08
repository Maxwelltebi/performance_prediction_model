"""Loads the trained artifacts and turns raw form input into a predicted grade.

Preprocessing mirrors the training notebook exactly: ordinal-encode
`parent_education` and `travel_time`, one-hot-encode the nominal columns with the
first category dropped, order the columns as the estimator saw them, then scale.

One deliberate deviation from `main.py`: the nominal columns are encoded against
the *known training categories* rather than by calling
`pd.get_dummies(..., drop_first=True)` on the single input row. On one row each
nominal column holds exactly one category, so get_dummies emits one dummy and
drop_first removes it -- every nominal feature silently collapses to 0 and the
model never sees gender, school type, internet access, extracurriculars or study
method. Encoding against the fixed category list reproduces what training
actually produced.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from features import (
    FEATURE_ORDER,
    GRADE_LABELS,
    ONE_HOT_BASELINE,
    ONE_HOT_CATEGORIES,
    PARENT_EDUCATION_ORDER,
    TRAVEL_TIME_ORDER,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_random_forest_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"


class ArtifactsMissing(RuntimeError):
    """Raised when the model or scaler file is not on disk."""


def _fitted_ordinal_encoder(order: list[str], column: str) -> OrdinalEncoder:
    encoder = OrdinalEncoder(categories=[order])
    encoder.fit(pd.DataFrame({column: order}))
    return encoder


class GradePredictor:
    def __init__(self) -> None:
        missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
        if missing:
            raise ArtifactsMissing(
                f"Missing artifact(s) in {MODEL_DIR}: {', '.join(missing)}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoder_parent_education = _fitted_ordinal_encoder(
            PARENT_EDUCATION_ORDER, "parent_education"
        )
        self.encoder_travel_time = _fitted_ordinal_encoder(TRAVEL_TIME_ORDER, "travel_time")

        n_expected = getattr(self.model, "n_features_in_", len(FEATURE_ORDER))
        if n_expected != len(FEATURE_ORDER):
            raise RuntimeError(
                f"Model expects {n_expected} features but FEATURE_ORDER defines "
                f"{len(FEATURE_ORDER)}. The artifacts and feature spec are out of sync."
            )

    def build_features(self, raw: dict) -> pd.DataFrame:
        """Raw form values -> the single-row, 18-column frame the scaler expects."""
        row: dict[str, float] = {
            "age": float(raw["age"]),
            "study_hours": float(raw["study_hours"]),
            "attendance_percentage": float(raw["attendance_percentage"]),
            "math_score": float(raw["math_score"]),
            "science_score": float(raw["science_score"]),
            "english_score": float(raw["english_score"]),
        }

        row["parent_education_encoded"] = float(
            self.encoder_parent_education.transform(
                pd.DataFrame({"parent_education": [raw["parent_education"]]})
            )[0, 0]
        )
        row["travel_time_encoded"] = float(
            self.encoder_travel_time.transform(
                pd.DataFrame({"travel_time": [raw["travel_time"]]})
            )[0, 0]
        )

        # One-hot against the training categories; the baseline category is all-zeros.
        for column, categories in ONE_HOT_CATEGORIES.items():
            value = raw[column]
            for category in categories:
                if category == ONE_HOT_BASELINE[column]:
                    continue
                row[f"{column}_{category}"] = int(value == category)

        return pd.DataFrame([row])[FEATURE_ORDER]

    def predict(self, raw: dict) -> dict:
        features = self.build_features(raw)
        scaled = self.scaler.transform(features)

        grade = str(self.model.predict(scaled)[0])
        probabilities = self.model.predict_proba(scaled)[0]
        breakdown = [
            {
                "grade": str(cls),
                "label": GRADE_LABELS.get(str(cls), ""),
                "probability": round(float(prob), 4),
            }
            for cls, prob in zip(self.model.classes_, probabilities)
        ]
        breakdown.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "grade": grade,
            "label": GRADE_LABELS.get(grade, ""),
            "confidence": round(float(np.max(probabilities)), 4),
            "breakdown": breakdown,
        }
