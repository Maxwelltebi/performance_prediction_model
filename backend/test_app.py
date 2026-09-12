"""The form stays available if local artifacts cannot initialize."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app
from predictor import GradePredictor
from schemas import StudentInput


def test_missing_artifacts_report_unavailable_but_keep_form(monkeypatch):
    def unavailable(self):
        raise FileNotFoundError("internal model path")

    monkeypatch.setattr(GradePredictor, "__init__", unavailable)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/form-schema.json').status_code == 200
        health = client.get('/api/health')
        assert health.status_code == 503
        assert health.json()['model_loaded'] is False
        assert 'internal model path' not in health.text
        response = client.post('/api/predict', json=StudentInput.model_config['json_schema_extra']['example'])
        assert response.status_code == 503
        assert 'internal model path' not in response.text
