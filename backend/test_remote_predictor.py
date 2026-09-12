"""Remote contract and failure tests; no network or HF account required."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app
from remote_predictor import RemoteGradePredictor, RemotePredictionUnavailable
from schemas import StudentInput

URL = "https://example.us-east-1.aws.endpoints.huggingface.cloud"
RESULT = {"grade": "b", "label": "Very good", "confidence": 0.68, "breakdown": []}
SAMPLE = StudentInput.model_config["json_schema_extra"]["example"]


def client_with_transport(handler):
    predictor = RemoteGradePredictor(URL, "test-token")
    predictor.client.close()
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-token"},
    )
    return predictor


def test_remote_request_contract():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        assert json.loads(request.content) == {"inputs": SAMPLE}
        return httpx.Response(200, json=RESULT)

    predictor = client_with_transport(handler)
    try:
        assert predictor.predict(SAMPLE) == RESULT
    finally:
        predictor.close()


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_remote_failure_hides_upstream_body(status):
    predictor = client_with_transport(lambda r: httpx.Response(status, text="private upstream details"))
    try:
        with pytest.raises(RemotePredictionUnavailable) as exc:
            predictor.predict(SAMPLE)
        assert "private upstream details" not in str(exc.value)
    finally:
        predictor.close()


def test_timeout_is_recoverable():
    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    predictor = client_with_transport(timeout)
    try:
        with pytest.raises(RemotePredictionUnavailable):
            predictor.predict(SAMPLE)
    finally:
        predictor.close()


def test_invalid_upstream_response():
    predictor = client_with_transport(lambda r: httpx.Response(200, json={"unexpected": True}))
    try:
        with pytest.raises(RemotePredictionUnavailable):
            predictor.predict(SAMPLE)
    finally:
        predictor.close()


@pytest.mark.parametrize("url", ["http://example.com", "https://example.com", URL + "?token=secret"])
def test_rejects_unexpected_token_destinations(url):
    with pytest.raises(ValueError):
        RemoteGradePredictor(url, "test-token")


def test_app_remote_mode_and_unavailable_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT_URL", URL)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(RemoteGradePredictor, "predict", lambda self, raw: RESULT)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is None
        assert health.json()["remote_model_status"] == "not_probed"
        assert client.post("/api/predict", json=SAMPLE).json() == RESULT
        assert client.post("/api/predict", json={}).status_code == 422

        def unavailable(self, raw):
            raise RemotePredictionUnavailable("Please try again shortly.")

        monkeypatch.setattr(RemoteGradePredictor, "predict", unavailable)
        assert client.post("/api/predict", json=SAMPLE).status_code == 503


def test_missing_token_fails_readiness_without_exposing_secrets(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT_URL", URL)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 503
        assert client.post("/api/predict", json=SAMPLE).status_code == 503
