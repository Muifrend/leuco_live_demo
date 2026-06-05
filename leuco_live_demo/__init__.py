"""Repo-root import shim for the src package."""

from pathlib import Path

SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "leuco_live_demo"
if SRC_PACKAGE.is_dir():
    __path__.append(str(SRC_PACKAGE))

__version__ = "0.1.0"
