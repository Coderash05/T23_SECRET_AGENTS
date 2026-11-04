"""Repo-root launcher so `streamlit run app.py` works without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from termsheet_ai.ui.app import main  # noqa: E402

main()
