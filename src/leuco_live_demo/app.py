from __future__ import annotations

from collections import deque
import logging
import threading
import time

import cv2
import numpy as np

from .alerts import AlertManager
from .config import AppConfig
from .http_server import SharedDemoState, create_server
from .inference import InferenceBackend, create_backend
from .models import AlertState, DecisionState, PoolState, RiskMetrics, Track
from .overlay import build_status, draw_overlay
from .pool_gate import PoolGate
from .risk import RiskEngine
from .sources import VideoSource, create_source
from .tracker import OnePersonTracker

LOGGER = logging.getLogger(__name__)
ERROR_LOG_INTERVAL_SECONDS = 5.0
PLACEHOLDER_INTERVAL_SECONDS = 1.0
STATUS_LOG_INTERVAL_SECONDS = 5.0


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
        self.backend: InferenceBackend | None = None
        self.source: VideoSource | None = None
        self.tracker = OnePersonTracker()
        self.pool_gate = PoolGate(config.pool_gate)
        self.risk = RiskEngine(config)
        self.alerts = AlertManager(config)
        self.capture_rate = RateMeter()
        self.http_server = create_server(config.http_host, config.http_port, self.shared_state)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self._last_ai_at = 0.0
        self._min_ai_interval = 1.0 / max(0.1, self.config.ai_fps)
        self._read_failures = 0
        self._last_no_frame_log_at = 0.0
        self._last_source_error_log_at = 0.0
        self._last_inference_error_log_at = 0.0
        self._last_placeholder_at = 0.0
        self._last_status_log_at = 0.0

    def run(self) -> int:
        self.http_thread.start()
        LOGGER.info("HTTP server listening on http://%s:%s/", self.config.http_host, self.config.http_port)
        track: Track | None = None
        pool = self.pool_gate.evaluate(None)
        decision = self._empty_decision()
        alert = self.alerts.snapshot()
        message = "starting"
        self._publish_placeholder(message, decision, alert, track, pool)

        try:
            try:
                self._initialize_pipeline(decision, alert, track, pool)
            except Exception as exc:  # noqa: BLE001 - keep HTTP diagnostics alive
                message = f"startup failed: {exc}"
                LOGGER.exception("pipeline startup failed")
                self._publish_placeholder(message, decision, alert, track, pool)
                self._wait_for_shutdown(message)
                return 2

            LOGGER.info("capture loop started")
            while True:
                try:
                    frame = self._read_frame()
                except Exception as exc:  # noqa: BLE001 - keep stream/status visible
                    message = f"source read failed: {exc}"
                    self._log_exception_throttled("_last_source_error_log_at", "video source read failed")
                    self._publish_placeholder_if_due(message, decision, alert, track, pool)
                    time.sleep(0.25)
                    continue

                if frame is None:
                    message = "source returned no frame"
                    self._read_failures += 1
                    self._log_no_frame()
                    self._publish_placeholder_if_due(message, decision, alert, track, pool)
                    time.sleep(0.1)
                    continue

                if self._read_failures:
                    LOGGER.info("source recovered after %s empty reads", self._read_failures)
                    self._read_failures = 0
                capture_fps = self.capture_rate.tick(frame.timestamp)
                if self._should_process_ai(frame.timestamp):
                    try:
                        track, pool, decision, alert = self._process_ai_frame(frame)
                        message = ""
                    except Exception as exc:  # noqa: BLE001 - publish video even if AI fails
                        self._last_ai_at = frame.timestamp
                        message = f"inference failed: {exc}"
                        self._log_exception_throttled("_last_inference_error_log_at", "AI processing failed")

                self._publish_frame(frame.image, decision, alert, track, pool, capture_fps, message)
                self._log_runtime_status(frame.index, decision, alert, capture_fps, message)
        except KeyboardInterrupt:
            LOGGER.info("shutdown requested")
            return 0
        finally:
            self.close()

    def close(self) -> None:
        self.shared_state.stop()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.http_thread.join(timeout=2.0)
        if self.source is not None:
            self.source.close()
        LOGGER.info("shutdown complete")

    def _initialize_pipeline(
        self,
        decision: DecisionState,
        alert: AlertState,
        track: Track | None,
        pool: PoolState,
    ) -> None:
        LOGGER.info(
            "config source=%s backend=%s rtsp_backend=%s gstreamer_pipeline=%s pool_gate=%s roi=%s roi_reference=%s ai_fps=%s alerts_enabled=%s",
            self.config.source,
            self.config.inference_backend,
            self.config.rtsp_backend,
            self.config.gstreamer_pipeline,
            self.config.pool_gate,
            self.config.inference_roi or "disabled",
            self.config.inference_roi_reference_size or "frame",
            self.config.ai_fps,
            self.config.alerts_enabled,
        )

        self._publish_placeholder("loading inference backend", decision, alert, track, pool)
        self.backend = create_backend(self.config)
        LOGGER.info("inference backend ready: %s", self.backend.__class__.__name__)

        self._publish_placeholder("opening video source", decision, alert, track, pool)
        self.source = create_source(self.config)
        LOGGER.info("video source ready: %s", self.source.__class__.__name__)

    def _read_frame(self):
        if self.source is None:
            raise RuntimeError("video source is not initialized")
        return self.source.read()

    def _should_process_ai(self, timestamp: float) -> bool:
        return timestamp - self._last_ai_at >= self._min_ai_interval

    def _process_ai_frame(self, frame) -> tuple[Track | None, PoolState, DecisionState, AlertState]:
        if self.backend is None:
            raise RuntimeError("inference backend is not initialized")
        result = self.backend.infer(frame)
        track = self.tracker.update(result.detections)
        pool = self.pool_gate.evaluate(track)
        decision = self.risk.process(frame, track, pool)
        alert = self.alerts.maybe_send(decision)
        self._last_ai_at = frame.timestamp
        return track, pool, decision, alert

    def _publish_frame(
        self,
        image,
        decision: DecisionState,
        alert: AlertState,
        track: Track | None,
        pool: PoolState,
        capture_fps: float,
        message: str,
    ) -> None:
        status = build_status(self.config, decision, alert, track, capture_fps, message)
        try:
            annotated = draw_overlay(image, status, track, pool)
        except Exception as exc:  # noqa: BLE001 - preserve status if overlay rendering fails
            LOGGER.exception("overlay render failed")
            self.shared_state.update_status({**status.as_dict(), "message": f"overlay failed: {exc}"})
            return
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            self.shared_state.update(encoded.tobytes(), status.as_dict())
            return
        LOGGER.error("JPEG encode failed")
        self.shared_state.update_status({**status.as_dict(), "message": "jpeg encode failed"})

    def _publish_status(
        self,
        decision: DecisionState,
        alert: AlertState,
        track: Track | None,
        capture_fps: float,
        message: str,
    ) -> None:
        status = build_status(self.config, decision, alert, track, capture_fps, message)
        self.shared_state.update_status(status.as_dict())

    def _publish_placeholder(
        self,
        message: str,
        decision: DecisionState,
        alert: AlertState,
        track: Track | None,
        pool: PoolState,
    ) -> None:
        image = self._placeholder_image(message)
        self._publish_frame(image, decision, alert, track, pool, 0.0, message)
        self._last_placeholder_at = time.time()

    def _publish_placeholder_if_due(
        self,
        message: str,
        decision: DecisionState,
        alert: AlertState,
        track: Track | None,
        pool: PoolState,
    ) -> None:
        now = time.time()
        if now - self._last_placeholder_at >= PLACEHOLDER_INTERVAL_SECONDS:
            self._publish_placeholder(message, decision, alert, track, pool)
            return
        self._publish_status(decision, alert, track, 0.0, message)

    def _placeholder_image(self, message: str):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:] = (20, 24, 26)
        rows = [
            "Leuco live demo",
            message[:110],
            "Check the terminal logs and /status for details.",
        ]
        y = 270
        for idx, row in enumerate(rows):
            scale = 1.15 if idx == 0 else 0.82
            thickness = 2 if idx == 0 else 1
            cv2.putText(
                image,
                row,
                (80, y + idx * 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (235, 238, 232),
                thickness,
                cv2.LINE_AA,
            )
        return image

    def _wait_for_shutdown(self, message: str) -> None:
        LOGGER.error("%s; HTTP diagnostics remain available until Ctrl-C", message)
        while True:
            time.sleep(1.0)

    def _log_no_frame(self) -> None:
        now = time.time()
        if now - self._last_no_frame_log_at < ERROR_LOG_INTERVAL_SECONDS:
            return
        LOGGER.warning("source returned no frame (%s consecutive empty reads)", self._read_failures)
        self._last_no_frame_log_at = now

    def _log_exception_throttled(self, timestamp_attr: str, message: str) -> None:
        now = time.time()
        if now - getattr(self, timestamp_attr) < ERROR_LOG_INTERVAL_SECONDS:
            return
        LOGGER.exception(message)
        setattr(self, timestamp_attr, now)

    def _log_runtime_status(
        self,
        frame_index: int,
        decision: DecisionState,
        alert: AlertState,
        capture_fps: float,
        message: str,
    ) -> None:
        now = time.time()
        if now - self._last_status_log_at < STATUS_LOG_INTERVAL_SECONDS:
            return
        LOGGER.info(
            "frame=%s capture_fps=%.1f person=%s risk=%s high_risk=%s/%s alert=%s message=%s",
            frame_index,
            capture_fps,
            decision.person_detected,
            decision.risk_state,
            decision.high_risk_frames,
            decision.window_size,
            alert.state,
            message or "ok",
        )
        self._last_status_log_at = now

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
