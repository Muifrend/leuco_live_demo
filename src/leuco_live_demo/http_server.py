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
STREAM_HEARTBEAT_SECONDS = 1.0


class DemoHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class SharedDemoState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.frame_id = 0
        self.status: dict[str, Any] = {"message": "starting"}
        self.running = True

    def update(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self.condition:
            self.latest_jpeg = jpeg
            self.frame_id += 1
            self.status = status
            self.condition.notify_all()

    def update_status(self, status: dict[str, Any]) -> None:
        with self.condition:
            self.status = status
            self.condition.notify_all()

    def snapshot(self) -> tuple[bytes | None, dict[str, Any]]:
        with self.condition:
            return self.latest_jpeg, dict(self.status)

    def wait_for_jpeg(self, last_frame_id: int, timeout: float) -> tuple[int, bytes | None, bool]:
        with self.condition:
            if self.running and self.frame_id == last_frame_id:
                self.condition.wait(timeout=timeout)
            return self.frame_id, self.latest_jpeg, self.running

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
            if self._matches_path(path, "stream.mjpg"):
                self._stream()
            elif self._matches_path(path, "status"):
                self._status()
            elif self._is_index_path(path):
                self._index()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        @staticmethod
        def _is_index_path(path: str) -> bool:
            if path in {"", "/"} or path.endswith("/"):
                return True
            last_segment = path.rsplit("/", maxsplit=1)[-1]
            return "." not in last_segment

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
            last_frame_id = -1
            last_sent_at = 0.0
            try:
                while True:
                    frame_id, jpeg, running = state.wait_for_jpeg(last_frame_id, timeout=STREAM_HEARTBEAT_SECONDS)
                    if not running:
                        break
                    if jpeg is None:
                        continue
                    if frame_id == last_frame_id and time.time() - last_sent_at < STREAM_HEARTBEAT_SECONDS:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last_frame_id = frame_id
                    last_sent_at = time.time()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                return None
            except OSError as exc:
                LOGGER.debug("MJPEG client disconnected: %s", exc)
                return None

    return DemoHTTPServer((host, port), Handler)


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
    <img id="stream" alt="Leuco annotated live stream">
  </main>
  <script>
    const basePath = window.location.pathname.endsWith('/') ? window.location.pathname : `${window.location.pathname}/`;
    function endpoint(name) {
      return new URL(name, `${window.location.origin}${basePath}`).toString();
    }
    function connectStream() {
      document.getElementById('stream').src = `${endpoint('stream.mjpg')}?t=${Date.now()}`;
    }
    async function tick() {
      try {
        const res = await fetch(endpoint('status'), {cache: 'no-store'});
        const data = await res.json();
        const message = data.message ? ` | ${data.message}` : '';
        document.getElementById('status').textContent = `${data.risk_state || 'starting'} | ${data.alert_state || 'idle'}${message}`;
      } catch (_) {}
    }
    document.getElementById('stream').addEventListener('error', () => setTimeout(connectStream, 1000));
    connectStream();
    setInterval(tick, 1000); tick();
  </script>
</body>
</html>
"""
