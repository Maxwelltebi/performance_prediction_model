"""Request and response models for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentInput(BaseModel):
    """The 13 raw values a user supplies. Mirrors the notebook's example input."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 18,
                "gender": "male",
                "school_type": "private",
                "parent_education": "phd",
                "study_hours": 7.0,
                "attendance_percentage": 90.0,
                "internet_access": "yes",
                "travel_time": "<15 min",
                "extra_activities": "yes",
                "study_method": "notes",
                "math_score": 85.0,
                "science_score": 92.0,
                "english_score": 88.0,
            }
        },
    )

    age: int = Field(ge=10, le=25)
    study_hours: float = Field(ge=0, le=24)
    attendance_percentage: float = Field(ge=0, le=100)
    math_score: float = Field(ge=0, le=100)
    science_score: float = Field(ge=0, le=100)
    english_score: float = Field(ge=0, le=100)

    gender: Literal["female", "male", "other"]
    school_type: Literal["private", "public"]
    parent_education: Literal[
        "no formal", "diploma", "high school", "graduate", "post graduate", "phd"
    ]
    study_method: Literal[
        "coaching", "group study", "mixed", "notes", "online videos", "textbook"
    ]
    internet_access: Literal["no", "yes"]
    extra_activities: Literal["no", "yes"]
    travel_time: Literal["<15 min", "15-30 min", "30-60 min", ">60 min"]


class GradeProbability(BaseModel):
    grade: str
    label: str
    probability: float


class PredictionResponse(BaseModel):
    grade: str
    label: str
    confidence: float
    breakdown: list[GradeProbability]
