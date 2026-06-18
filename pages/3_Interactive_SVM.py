"""Interactive SVM page."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"

spec = importlib.util.spec_from_file_location("hw8_interactive_streamlit_app", APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load interactive app from {APP_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
