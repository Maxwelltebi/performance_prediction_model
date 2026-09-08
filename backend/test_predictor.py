"""Smoke tests for the preprocessing contract.

The important one is `test_nominal_fields_reach_the_model`: it guards against the
`get_dummies(drop_first=True)`-on-one-row trap that silently zeroed every nominal
feature in the original notebook export.

Run with:  python -m pytest backend/test_predictor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from features import FEATURE_ORDER, ONE_HOT_BASELINE, ONE_HOT_CATEGORIES
from predictor import GradePredictor

SAMPLE = {
    "age": 18, "gender": "male", "school_type": "private", "parent_education": "phd",
    "study_hours": 7.0, "attendance_percentage": 90.0, "internet_access": "yes",
    "travel_time": "<15 min", "extra_activities": "yes", "study_method": "notes",
    "math_score": 85.0, "science_score": 92.0, "english_score": 88.0,
}


@pytest.fixture(scope="module")
def predictor():
    return GradePredictor()


def test_feature_frame_matches_training_layout(predictor):
    frame = predictor.build_features(SAMPLE)
    assert list(frame.columns) == FEATURE_ORDER
    assert frame.shape == (1, 18)


def test_ordinal_encoding_uses_training_order(predictor):
    frame = predictor.build_features(SAMPLE)
    assert frame.loc[0, "parent_education_encoded"] == 5.0  # 'phd' is last
    assert frame.loc[0, "travel_time_encoded"] == 0.0       # '<15 min' is first


def test_one_hot_matches_selected_category(predictor):
    frame = predictor.build_features(SAMPLE)
    assert frame.loc[0, "gender_male"] == 1
    assert frame.loc[0, "gender_other"] == 0
    assert frame.loc[0, "school_type_public"] == 0  # 'private' is the baseline
    assert frame.loc[0, "study_method_notes"] == 1


def test_baseline_category_is_all_zeros(predictor):
    values = dict(SAMPLE, gender=ONE_HOT_BASELINE["gender"])
    frame = predictor.build_features(values)
    assert frame.loc[0, "gender_male"] == 0
    assert frame.loc[0, "gender_other"] == 0


@pytest.mark.parametrize("column", list(ONE_HOT_CATEGORIES))
def test_nominal_fields_reach_the_model(predictor, column):
    """Changing any nominal input must change the feature vector."""
    baseline = predictor.build_features(SAMPLE)
    for category in ONE_HOT_CATEGORIES[column]:
        if category == SAMPLE[column]:
            continue
        other = predictor.build_features(dict(SAMPLE, **{column: category}))
        assert not other.equals(baseline), f"{column}={category} produced no change"


def test_predict_returns_a_valid_grade(predictor):
    result = predictor.predict(SAMPLE)
    assert result["grade"] in set("abcdef")
    assert 0 <= result["confidence"] <= 1
    assert len(result["breakdown"]) == 6
    assert sum(item["probability"] for item in result["breakdown"]) == pytest.approx(1.0, abs=1e-3)


def test_matches_notebook_example(predictor):
    """The notebook's worked example predicts grade 'b'."""
    assert predictor.predict(SAMPLE)["grade"] == "b"
