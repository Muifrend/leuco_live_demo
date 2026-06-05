from __future__ import annotations

import sys

from .app import DemoApp
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
        app = DemoApp(config)
        return app.run()
    except Exception as exc:  # noqa: BLE001 - convert startup failures to CLI errors
        print(f"leuco-live-demo: {exc}", file=sys.stderr)
        return 2
