"""Deployment boundaries: static bootstrap and a function without model imports."""

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


def test_vercel_entrypoint_never_imports_ml_even_with_missing_configuration():
    script = '''
import sys
from fastapi.testclient import TestClient
from server import app
with TestClient(app) as client:
    health = client.get('/api/health')
    assert health.status_code == 503
    assert health.json()['mode'] == 'remote'
    assert client.get('/form-schema.json').status_code == 200
    assert len(client.get('/form-schema.json').json()['fields']) == 13
assert not {'sklearn', 'pandas', 'numpy', 'joblib', 'predictor'} & sys.modules.keys()
'''
    env = {key: value for key, value in os.environ.items()
           if key not in ("HF_TOKEN", "HF_ENDPOINT_URL")}
    env["VERCEL"] = "1"
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, check=True)
