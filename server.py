"""Vercel entrypoint; keep the original main.py notebook export out of serving."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from backend.app import app
