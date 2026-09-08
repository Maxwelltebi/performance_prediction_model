"""Single source of truth for the model's input contract.

Everything here is derived from the training notebook (student_performance_model.ipynb):
the ordinal category orders, the one-hot baseline categories that `drop_first=True`
removed, and the exact 18-column feature order the estimator was fitted on.

The API serves this same spec to the frontend so the form and the model can never
drift apart.
"""

# --- Column groups, as defined during training -------------------------------

NOMINAL_COLS = ["gender", "school_type", "internet_access", "extra_activities", "study_method"]
ORDINAL_COLS = ["parent_education", "travel_time"]

PARENT_EDUCATION_ORDER = ["no formal", "diploma", "high school", "graduate", "post graduate", "phd"]
TRAVEL_TIME_ORDER = ["<15 min", "15-30 min", "30-60 min", ">60 min"]

# `pd.get_dummies(..., drop_first=True)` sorts categories and drops the first one,
# so each nominal column has an implicit baseline that is represented by all-zeros.
ONE_HOT_CATEGORIES = {
    "gender": ["female", "male", "other"],
    "school_type": ["private", "public"],
    "internet_access": ["no", "yes"],
    "extra_activities": ["no", "yes"],
    "study_method": ["coaching", "group study", "mixed", "notes", "online videos", "textbook"],
}
ONE_HOT_BASELINE = {col: cats[0] for col, cats in ONE_HOT_CATEGORIES.items()}

# The exact order the RandomForest and StandardScaler were fitted on.
FEATURE_ORDER = [
    "age", "study_hours", "attendance_percentage", "math_score", "science_score",
    "english_score", "parent_education_encoded", "travel_time_encoded",
    "gender_male", "gender_other", "school_type_public", "internet_access_yes",
    "extra_activities_yes", "study_method_group study", "study_method_mixed",
    "study_method_notes", "study_method_online videos", "study_method_textbook",
]

# --- Form spec ---------------------------------------------------------------
# Drives Pydantic validation, the rendered form, and the sample payload.

NUMERIC_FIELDS = {
    "age":                   {"label": "Age",            "min": 10,  "max": 25,  "step": 1,   "default": 17,   "unit": "yrs", "group": "Profile"},
    "study_hours":           {"label": "Study hours",    "min": 0,   "max": 24,  "step": 0.1, "default": 4.0,  "unit": "hrs/day", "group": "Habits"},
    "attendance_percentage": {"label": "Attendance",     "min": 0,   "max": 100, "step": 0.1, "default": 85.0, "unit": "%", "group": "Habits"},
    "math_score":            {"label": "Math score",     "min": 0,   "max": 100, "step": 0.1, "default": 70.0, "unit": "/100", "group": "Scores"},
    "science_score":         {"label": "Science score",  "min": 0,   "max": 100, "step": 0.1, "default": 70.0, "unit": "/100", "group": "Scores"},
    "english_score":         {"label": "English score",  "min": 0,   "max": 100, "step": 0.1, "default": 70.0, "unit": "/100", "group": "Scores"},
}

CATEGORICAL_FIELDS = {
    "gender":           {"label": "Gender",            "options": ONE_HOT_CATEGORIES["gender"],           "default": "female",     "group": "Profile"},
    "school_type":      {"label": "School type",       "options": ONE_HOT_CATEGORIES["school_type"],      "default": "public",     "group": "Profile"},
    "parent_education": {"label": "Parent education",  "options": PARENT_EDUCATION_ORDER,                 "default": "high school","group": "Profile"},
    "study_method":     {"label": "Study method",      "options": ONE_HOT_CATEGORIES["study_method"],     "default": "notes",      "group": "Habits"},
    "extra_activities": {"label": "Extracurriculars",  "options": ONE_HOT_CATEGORIES["extra_activities"], "default": "yes",        "group": "Habits"},
    "internet_access":  {"label": "Internet access",   "options": ONE_HOT_CATEGORIES["internet_access"],  "default": "yes",        "group": "Habits"},
    "travel_time":      {"label": "Travel time",       "options": TRAVEL_TIME_ORDER,                      "default": "<15 min",    "group": "Habits"},
}

FIELD_GROUPS = ["Profile", "Habits", "Scores"]

GRADE_LABELS = {
    "a": "Excellent",
    "b": "Very good",
    "c": "Good",
    "d": "Satisfactory",
    "e": "Needs improvement",
    "f": "At risk",
}


def form_schema() -> dict:
    """The payload the frontend uses to render its form."""
    def numeric(name, spec):
        return {"name": name, "type": "number", **spec}

    def categorical(name, spec):
        return {"name": name, "type": "select", **spec}

    fields = [numeric(n, s) for n, s in NUMERIC_FIELDS.items()]
    fields += [categorical(n, s) for n, s in CATEGORICAL_FIELDS.items()]
    return {
        "groups": FIELD_GROUPS,
        "fields": fields,
        "grades": [{"grade": g, "label": lbl} for g, lbl in GRADE_LABELS.items()],
    }
