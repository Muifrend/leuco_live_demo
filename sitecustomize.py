"""Make the src package importable when running from the repo root.

This keeps `.venv/bin/python -m leuco_live_demo` working without modifying the
Jetson virtual environment.
"""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir():
    src_text = str(SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
