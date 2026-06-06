from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time
from typing import Any
from urllib.parse import urlsplit


LOGGER = logging.getLogger(__name__)


class SharedDemoState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.status: dict[str, Any] = {"message": "starting"}
        self.running = True

    def update(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self.condition:
            self.latest_jpeg = jpeg
            self.status = status
            self.condition.notify_all()

    def update_status(self, status: dict[str, Any]) -> None:
        with self.condition:
            self.status = status
            self.condition.notify_all()

    def snapshot(self) -> tuple[bytes | None, dict[str, Any]]:
        with self.condition:
            return self.latest_jpeg, dict(self.status)

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()


def create_server(host: str, port: int, state: SharedDemoState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LeucoLiveDemo/0.1"

        def log_message(self, fmt: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            LOGGER.debug("HTTP GET %s", path)
            if self._is_index_path(path):
                self._index()
            elif self._matches_path(path, "stream.mjpg"):
                self._stream()
            elif self._matches_path(path, "status"):
                self._status()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        @staticmethod
        def _is_index_path(path: str) -> bool:
            return path in {"", "/"} or path.endswith("/")

        @staticmethod
        def _matches_path(path: str, endpoint: str) -> bool:
            return path == f"/{endpoint}" or path.endswith(f"/{endpoint}")

        def _index(self) -> None:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _status(self) -> None:
            _, status = state.snapshot()
            body = json.dumps(status, sort_keys=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_sent: bytes | None = None
            try:
                while state.running:
                    with state.condition:
                        state.condition.wait(timeout=1.0)
                        jpeg = state.latest_jpeg
                    if jpeg is None or jpeg is last_sent:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    last_sent = jpeg
            except (BrokenPipeError, ConnectionResetError):
                return None

    return ThreadingHTTPServer((host, port), Handler)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leuco Live Demo</title>
  <style>
    html, body { margin: 0; min-height: 100%; background: #111; color: #f3f3ef; font-family: system-ui, sans-serif; }
    main { display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }
    header { padding: 12px 16px; background: #1d2224; display: flex; justify-content: space-between; gap: 12px; }
    img { width: 100%; height: calc(100vh - 52px); object-fit: contain; background: #050505; }
    code { color: #9be2ff; }
  </style>
</head>
<body>
  <main>
    <header><strong>Leuco Live Demo</strong><code id="status">status</code></header>
    <img src="stream.mjpg" alt="Leuco annotated live stream">
  </main>
  <script>
    async function tick() {
      try {
        const res = await fetch('status', {cache: 'no-store'});
        const data = await res.json();
        const message = data.message ? ` | ${data.message}` : '';
        document.getElementById('status').textContent = `${data.risk_state || 'starting'} | ${data.alert_state || 'idle'}${message}`;
      } catch (_) {}
    }
    setInterval(tick, 1000); tick();
  </script>
</body>
</html>
"""
