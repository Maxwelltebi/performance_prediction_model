"""Stage only the artifacts and inference code for a Hugging Face model repo."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / ".hf-release"
FILES = {
    "models/best_random_forest_model.joblib": "models/best_random_forest_model.joblib",
    "models/scaler.joblib": "models/scaler.joblib",
    "backend/predictor.py": "backend/predictor.py",
    "backend/features.py": "backend/features.py",
    "backend/schemas.py": "backend/schemas.py",
    "deploy/huggingface/handler.py": "handler.py",
    "deploy/huggingface/requirements.txt": "requirements.txt",
    "deploy/huggingface/README.md": "README.md",
}


def main() -> None:
    # Refuse an existing directory so old or unrelated files cannot be uploaded.
    for source in FILES:
        if not (ROOT / source).is_file():
            raise FileNotFoundError(source)
    DEST.mkdir(exist_ok=False)
    for source, target in FILES.items():
        destination = DEST / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    print(f"Prepared {len(FILES)} files in {DEST}. Nothing uploaded.")


if __name__ == "__main__":
    main()
