from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .models import AlertState, DecisionState, PoolState, RuntimeStatus, Track
from .roi import scale_roi_to_frame


def build_status(
    config: AppConfig,
    decision: DecisionState,
    alert: AlertState,
    track: Track | None,
    pool: PoolState,
    capture_fps: float,
    message: str = "",
) -> RuntimeStatus:
    return RuntimeStatus(
        source=config.source,
        backend=config.inference_backend,
        pool_gate=config.pool_gate,
        person_detected=decision.person_detected,
        in_pool=decision.in_pool,
        risk_active=decision.risk_active,
        risk_state=decision.risk_state,
        high_risk_frames=decision.high_risk_frames,
        window_size=decision.window_size,
        alert_state=alert.state,
        alert_cooldown_seconds=alert.cooldown_seconds,
        capture_fps=round(capture_fps, 2),
        ai_fps=decision.ai_fps,
        last_alert_at=alert.last_alert_at,
        tracked_id=track.track_id if track and track.missed_frames == 0 else None,
        bbox=track.bbox if track and track.missed_frames == 0 else None,
        pool_gate_active=pool.active,
        pool_polygon=pool.polygon,
        pool_test_point=pool.test_point,
        inference_roi=config.inference_roi,
        inference_roi_reference_size=config.inference_roi_reference_size,
        upper_activity=round(decision.metrics.upper_activity, 4),
        forward_progress=round(decision.metrics.forward_progress, 4),
        lost_visibility_frames=decision.metrics.lost_visibility_frames,
        distress_candidate_frames=decision.metrics.distress_candidate_frames,
        alert_reason=decision.alert_reason,
        message=message or alert.last_error or "",
    )


def draw_overlay(
    image,
    status: RuntimeStatus,
    track: Track | None,
    pool: PoolState,
) -> object:
    canvas = image.copy()
    height, width = canvas.shape[:2]

    if status.pool_gate == "full_frame_stub":
        cv2.rectangle(canvas, (4, 4), (width - 5, height - 5), (0, 220, 255), 3)
        _label(canvas, "FULL-FRAME POOL STUB", (18, 32), (0, 220, 255))

    if status.pool_polygon:
        polygon_color = (0, 220, 255) if status.in_pool else (0, 170, 255)
        polygon_points = np.array(
            [[(int(round(x)), int(round(y))) for x, y in status.pool_polygon]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, polygon_points, isClosed=True, color=polygon_color, thickness=3)
        first_x, first_y = polygon_points[0][0]
        _label(canvas, "POOL POLYGON", (int(first_x), max(22, int(first_y) - 8)), polygon_color)

    if status.inference_roi is not None:
        x1, y1, x2, y2 = scale_roi_to_frame(
            status.inference_roi,
            status.inference_roi_reference_size,
            width,
            height,
        )
        if x1 < width and y1 < height and x2 > 0 and y2 > 0:
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            roi_color = (255, 160, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), roi_color, 2)
            _label(canvas, "INFERENCE ROI", (x1, max(22, y1 - 8)), roi_color)

    if track is not None and track.missed_frames == 0:
        x1, y1, x2, y2 = track.bbox
        color = (0, 220, 0) if status.risk_state == "normal" else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _label(canvas, f"id:{track.track_id}", (x1, max(22, y1 - 8)), color)
        for x, y in track.keypoints.values():
            cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1)

    if status.pool_test_point is not None:
        x, y = status.pool_test_point
        test_color = (0, 220, 0) if status.in_pool else (0, 0, 255)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 7, test_color, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 10, (255, 255, 255), 2)

    rows = [
        f"source {status.source} | backend {status.backend}",
        f"person {'yes' if status.person_detected else 'no'} | in-pool {'yes' if status.in_pool else 'no'}",
        f"pool gate {'active' if status.pool_gate_active else 'inactive'} | {pool.label}",
        f"risk {'active' if status.risk_active else 'inactive'} | state {status.risk_state}",
        f"high-risk {status.high_risk_frames}/{status.window_size} | ai {status.ai_fps:g} fps",
        f"activity {status.upper_activity:.3f} | progress {status.forward_progress:.3f}",
        f"alert {status.alert_state} | capture {status.capture_fps:.1f} fps",
    ]
    if status.lost_visibility_frames or status.distress_candidate_frames:
        rows.append(
            f"lost visibility {status.lost_visibility_frames} | distress {status.distress_candidate_frames}"
        )
    if status.alert_reason:
        rows.append(f"reason {status.alert_reason[:70]}")
    if status.message:
        rows.append(f"message {status.message[:70]}")
    _panel(canvas, rows)
    return canvas


def _panel(canvas, rows: list[str]) -> None:
    x, y = 18, 52
    line_height = 26
    width = max(360, max(cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0] for row in rows) + 24)
    height = (len(rows) * line_height) + 20
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 10, y - 24), (x + width, y + height - 24), (12, 16, 18), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    for idx, row in enumerate(rows):
        cv2.putText(
            canvas,
            row,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 240),
            1,
            cv2.LINE_AA,
        )


def _label(canvas, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.rectangle(canvas, (x - 5, y - 20), (x + text_size[0] + 8, y + 7), (12, 16, 18), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
