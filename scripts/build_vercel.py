"""Generate the CDN files without importing the API or model dependencies."""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from features import form_schema


def build(output_dir: Path = ROOT / "public") -> None:
    for filename in ("best_random_forest_model.joblib", "scaler.joblib"):
        if not (ROOT / "models" / filename).is_file():
            raise FileNotFoundError(f"Required deployment artifact missing: models/{filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "frontend" / "index.html", output_dir / "index.html")
    (output_dir / "form-schema.json").write_text(
        json.dumps(form_schema(), ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    build()
    print("Built public/index.html and public/form-schema.json for Vercel CDN.")
