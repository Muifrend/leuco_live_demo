from __future__ import annotations

import logging
import sys

from .app import DemoApp
from .config import load_config

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("leuco_live_demo").setLevel(getattr(logging, level))


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
        configure_logging(config.log_level)
        LOGGER.info("starting leuco live demo")
        app = DemoApp(config)
        return app.run()
    except Exception as exc:  # noqa: BLE001 - convert startup failures to CLI errors
        logging.basicConfig(level=logging.INFO)
        LOGGER.exception("leuco live demo failed")
        print(f"leuco-live-demo: {exc}", file=sys.stderr)
        return 2
