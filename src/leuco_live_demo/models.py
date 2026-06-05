from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = tuple[int, int, int, int]
Point = tuple[float, float]
Keypoints = dict[str, Point]


@dataclass(frozen=True)
class Frame:
    image: Any
    timestamp: float
    index: int


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    label: str = "person"
    keypoints: Keypoints = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    detections: list[Detection]


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    confidence: float
    keypoints: Keypoints
    center: Point
    missed_frames: int = 0


@dataclass(frozen=True)
class PoolState:
    mode: str
    in_pool: bool
    active: bool
    label: str


@dataclass(frozen=True)
class RiskMetrics:
    upper_activity: float = 0.0
    forward_progress: float = 0.0
    high_activity: bool = False
    low_progress: bool = False


@dataclass(frozen=True)
class DecisionState:
    person_detected: bool
    in_pool: bool
    risk_active: bool
    risk_state: str
    high_risk_frames: int
    window_size: int
    window_seconds: float
    ai_fps: float
    should_alert: bool
    metrics: RiskMetrics = field(default_factory=RiskMetrics)


@dataclass(frozen=True)
class AlertState:
    state: str
    enabled: bool
    cooldown_seconds: float
    last_alert_at: float | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    source: str
    backend: str
    pool_gate: str
    person_detected: bool
    in_pool: bool
    risk_active: bool
    risk_state: str
    high_risk_frames: int
    window_size: int
    alert_state: str
    alert_cooldown_seconds: float
    capture_fps: float
    ai_fps: float
    last_alert_at: float | None
    tracked_id: int | None
    bbox: BBox | None
    upper_activity: float
    forward_progress: float
    message: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "backend": self.backend,
            "pool_gate": self.pool_gate,
            "person_detected": self.person_detected,
            "in_pool": self.in_pool,
            "risk_active": self.risk_active,
            "risk_state": self.risk_state,
            "high_risk_frames": self.high_risk_frames,
            "window_size": self.window_size,
            "alert_state": self.alert_state,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "capture_fps": self.capture_fps,
            "ai_fps": self.ai_fps,
            "last_alert_at": self.last_alert_at,
            "tracked_id": self.tracked_id,
            "bbox": self.bbox,
            "upper_activity": self.upper_activity,
            "forward_progress": self.forward_progress,
            "message": self.message,
        }
