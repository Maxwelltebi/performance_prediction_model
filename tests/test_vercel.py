"""Deployment boundaries: static bootstrap and bundled inference without secrets."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_vercel import build


def test_static_build_contains_full_form_without_api_bootstrap(tmp_path):
    build(tmp_path)
    schema = json.loads((tmp_path / "form-schema.json").read_text(encoding="utf-8"))
    assert len(schema["fields"]) == 13
    assert len({field["name"] for field in schema["fields"]}) == 13
    assert (tmp_path / "index.html").read_bytes() == (ROOT / "frontend/index.html").read_bytes()


def test_vercel_entrypoint_predicts_with_bundled_artifacts_and_no_secrets():
    script = '''
import sys
from fastapi.testclient import TestClient
from server import app
from schemas import StudentInput
with TestClient(app) as client:
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json()['model_loaded'] is True
    assert health.json()['mode'] == 'local'
    assert client.get('/form-schema.json').status_code == 200
    assert len(client.get('/form-schema.json').json()['fields']) == 13
    result = client.post('/api/predict', json=StudentInput.model_config['json_schema_extra']['example'])
    assert result.status_code == 200
    assert result.json()['grade'] == 'b'
    assert result.json()['confidence'] == 0.68
    assert len(result.json()['breakdown']) == 6
    assert client.post('/api/predict', json={}).status_code == 422
'''
    env = {key: value for key, value in os.environ.items()
           if key not in ("HF_TOKEN", "HF_ENDPOINT_URL")}
    env["VERCEL"] = "1"
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, check=True)
