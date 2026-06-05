from __future__ import annotations

from collections import deque
import threading
import time

import cv2

from .alerts import AlertManager
from .config import AppConfig
from .http_server import SharedDemoState, create_server
from .inference import create_backend
from .models import AlertState, DecisionState, RiskMetrics
from .overlay import build_status, draw_overlay
from .pool_gate import PoolGate
from .risk import RiskEngine
from .sources import create_source
from .tracker import OnePersonTracker


class RateMeter:
    def __init__(self, max_samples: int = 60) -> None:
        self.samples: deque[float] = deque(maxlen=max_samples)

    def tick(self, timestamp: float) -> float:
        self.samples.append(timestamp)
        if len(self.samples) < 2:
            return 0.0
        elapsed = self.samples[-1] - self.samples[0]
        return (len(self.samples) - 1) / elapsed if elapsed > 0 else 0.0


class DemoApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.shared_state = SharedDemoState()
        self.source = create_source(config)
        self.backend = create_backend(config)
        self.tracker = OnePersonTracker()
        self.pool_gate = PoolGate(config.pool_gate)
        self.risk = RiskEngine(config)
        self.alerts = AlertManager(config)
        self.capture_rate = RateMeter()
        self.http_server = create_server(config.http_host, config.http_port, self.shared_state)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)

    def run(self) -> int:
        self.http_thread.start()
        print(f"Leuco live demo listening on http://{self.config.http_host}:{self.config.http_port}/")
        last_ai_at = 0.0
        min_ai_interval = 1.0 / max(0.1, self.config.ai_fps)
        track = None
        pool = self.pool_gate.evaluate(None)
        decision = self._empty_decision()
        alert = self.alerts.snapshot()
        message = ""

        try:
            while True:
                frame = self.source.read()
                if frame is None:
                    message = "source returned no frame"
                    self._publish_status(decision, alert, track, 0.0, message)
                    time.sleep(0.1)
                    continue

                capture_fps = self.capture_rate.tick(frame.timestamp)
                if frame.timestamp - last_ai_at >= min_ai_interval:
                    result = self.backend.infer(frame)
                    track = self.tracker.update(result.detections)
                    pool = self.pool_gate.evaluate(track)
                    decision = self.risk.process(frame, track, pool)
                    alert = self.alerts.maybe_send(decision)
                    last_ai_at = frame.timestamp
                    message = ""

                status = build_status(self.config, decision, alert, track, capture_fps, message)
                annotated = draw_overlay(frame.image, status, track, pool)
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    self.shared_state.update(encoded.tobytes(), status.as_dict())
                else:
                    self.shared_state.update_status({**status.as_dict(), "message": "jpeg encode failed"})
        except KeyboardInterrupt:
            return 0
        finally:
            self.shared_state.stop()
            self.http_server.shutdown()
            self.source.close()

    def _publish_status(
        self,
        decision: DecisionState,
        alert: AlertState,
        track,
        capture_fps: float,
        message: str,
    ) -> None:
        status = build_status(self.config, decision, alert, track, capture_fps, message)
        self.shared_state.update_status(status.as_dict())

    def _empty_decision(self) -> DecisionState:
        return DecisionState(
            person_detected=False,
            in_pool=False,
            risk_active=False,
            risk_state="normal",
            high_risk_frames=0,
            window_size=self.config.window_size,
            window_seconds=self.config.decision_window_seconds,
            ai_fps=self.config.ai_fps,
            should_alert=False,
            metrics=RiskMetrics(),
        )
