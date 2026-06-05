from __future__ import annotations

import json
import threading
import time
import unittest
from urllib.request import urlopen

import cv2
import numpy as np

from leuco_live_demo.http_server import SharedDemoState, create_server


class HttpServerTests(unittest.TestCase):
    def test_status_index_and_stream_endpoints(self) -> None:
        state = SharedDemoState()
        server = create_server("127.0.0.1", 0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        state.update(encoded.tobytes(), {"risk_state": "normal", "alert_state": "idle"})

        try:
            with urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["risk_state"], "normal")

            with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                self.assertIn(b"Leuco Live Demo", response.read())

            with urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=2) as response:
                chunk = response.read(256)
            self.assertIn(b"--frame", chunk)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
        finally:
            state.stop()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
