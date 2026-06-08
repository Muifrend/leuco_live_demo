from __future__ import annotations

from collections import deque
from dataclasses import replace
import math

import cv2

from .config import AppConfig
from .models import DecisionState, Frame, PoolState, RiskMetrics, Track

UPPER_BODY_KEYS = {
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}
POSE_ACTIVITY_THRESHOLD = 0.045
FALLBACK_MOTION_THRESHOLD = 0.045
FORWARD_PROGRESS_THRESHOLD = 0.025
FORWARD_PROGRESS_LOOKBACK_SECONDS = 2.0


def _bbox_diag(track: Track) -> float:
    x1, y1, x2, y2 = track.bbox
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


def _frame_diag(frame: Frame) -> float:
    height, width = frame.image.shape[:2]
    return max(1.0, math.hypot(width, height))


class RiskEngine:
    """Frame-level cue scoring plus temporal in-pool distress rules."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.window: deque[bool] = deque(maxlen=config.window_size)
        self.centers: deque[tuple[float, float]] = deque(maxlen=config.window_size)
        self._prev_keypoints: dict[str, tuple[float, float]] | None = None
        self._prev_center: tuple[float, float] | None = None
        self._prev_gray = None
        self._inside_pool_session = False
        self._lost_visibility_armed = False
        self._outside_pool_frames = 0
        self._lost_visibility_frames = 0
        self._distress_candidate_frames = 0

    def process(self, frame: Frame, track: Track | None, pool: PoolState) -> DecisionState:
        person_present = track is not None and track.missed_frames == 0
        current_in_pool = bool(pool.active and pool.in_pool)
        reliable_in_pool_detection = self._reliable_in_pool_detection(track, person_present, current_in_pool)
        self._update_pool_session(pool, person_present, reliable_in_pool_detection)
        session_in_pool = bool(pool.active and self._inside_pool_session)
        risk_active = session_in_pool

        metrics = RiskMetrics()
        high_risk = False
        if person_present and current_in_pool and track is not None:
            self.centers.append(track.center)
            metrics = self._metrics(frame, track)
            high_risk = metrics.high_activity and metrics.low_progress
        else:
            self._prev_keypoints = None
            self._prev_center = None

        self._update_visibility_loss(track, person_present, current_in_pool, reliable_in_pool_detection)
        self.window.append(high_risk)
        high_count = sum(1 for item in self.window if item)
        pose_candidate = bool(
            session_in_pool
            and high_risk
            and high_count >= self.config.pose_risk_frames
        )
        lost_candidate = bool(
            session_in_pool
            and self._lost_visibility_frames >= self.config.lost_swimmer_frames
        )
        if pose_candidate or lost_candidate:
            self._distress_candidate_frames += 1
        else:
            self._distress_candidate_frames = 0

        high_risk_count_alert = bool(
            len(self.window) == self.window.maxlen
            and high_count >= self.config.high_risk_frames
        )
        persistent_alert = self._distress_candidate_frames >= self.config.distress_persist_frames
        should_alert = bool(session_in_pool and (high_risk_count_alert or persistent_alert))
        alert_reason = self._alert_reason(high_risk_count_alert, pose_candidate, lost_candidate)
        risk_state = self._risk_state(high_risk, pose_candidate, lost_candidate, should_alert)

        metrics = replace(
            metrics,
            lost_visibility_frames=self._lost_visibility_frames,
            distress_candidate_frames=self._distress_candidate_frames,
        )

        self._prev_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        return DecisionState(
            person_detected=person_present,
            in_pool=session_in_pool,
            risk_active=risk_active,
            risk_state=risk_state,
            high_risk_frames=high_count,
            window_size=self.config.window_size,
            window_seconds=self.config.decision_window_seconds,
            ai_fps=self.config.ai_fps,
            should_alert=should_alert,
            metrics=metrics,
            alert_reason=alert_reason,
        )

    def _update_pool_session(
        self,
        pool: PoolState,
        person_present: bool,
        reliable_in_pool_detection: bool,
    ) -> None:
        if not pool.active:
            self._clear_pool_session(clear_window=True)
            return
        if reliable_in_pool_detection:
            self._inside_pool_session = True
            self._lost_visibility_armed = True
            self._outside_pool_frames = 0
            return
        if person_present and not pool.in_pool:
            if self._inside_pool_session:
                self._outside_pool_frames += 1
                if self._outside_pool_frames >= self.config.confirmed_exit_frames:
                    self._clear_pool_session(clear_window=True)
            return
        self._outside_pool_frames = 0

    def _update_visibility_loss(
        self,
        track: Track | None,
        person_present: bool,
        current_in_pool: bool,
        reliable_in_pool_detection: bool,
    ) -> None:
        if not self._inside_pool_session or not self._lost_visibility_armed:
            self._lost_visibility_frames = 0
            return
        if reliable_in_pool_detection:
            self._lost_visibility_frames = 0
            return

        pose_missing = bool(
            person_present
            and current_in_pool
            and self.config.inference_backend == "yolo_pose"
            and track is not None
            and self._missing_body_pose(track)
        )
        if not person_present or pose_missing:
            self._lost_visibility_frames += 1
            return
        if current_in_pool:
            self._lost_visibility_frames = 0

    def _clear_pool_session(self, clear_window: bool = False) -> None:
        self._inside_pool_session = False
        self._lost_visibility_armed = False
        self._outside_pool_frames = 0
        self._lost_visibility_frames = 0
        self._distress_candidate_frames = 0
        self.centers.clear()
        self._prev_keypoints = None
        self._prev_center = None
        if clear_window:
            self.window.clear()

    def _alert_reason(
        self,
        high_risk_count_alert: bool,
        pose_candidate: bool,
        lost_candidate: bool,
    ) -> str:
        if lost_candidate:
            return "lost swimmer after in-pool track"
        if high_risk_count_alert:
            return "high-risk frame count"
        if pose_candidate:
            return "pose risk accumulation"
        return ""

    def _risk_state(
        self,
        high_risk: bool,
        pose_candidate: bool,
        lost_candidate: bool,
        should_alert: bool,
    ) -> str:
        if should_alert:
            return "high_risk"
        if lost_candidate:
            return "lost_swimmer"
        if pose_candidate:
            return "possible_distress"
        if high_risk:
            return "high_risk"
        return "normal"

    def _missing_body_pose(self, track: Track) -> bool:
        return len(UPPER_BODY_KEYS.intersection(track.keypoints)) < 3

    def _reliable_in_pool_detection(
        self,
        track: Track | None,
        person_present: bool,
        current_in_pool: bool,
    ) -> bool:
        if not person_present or not current_in_pool or track is None:
            return False
        if self.config.inference_backend == "yolo_pose":
            return not self._missing_body_pose(track)
        return True

    def _metrics(self, frame: Frame, track: Track) -> RiskMetrics:
        if track.keypoints:
            upper_activity = self._pose_activity(track)
            activity_threshold = POSE_ACTIVITY_THRESHOLD
        else:
            upper_activity = self._motion_activity(frame, track)
            activity_threshold = FALLBACK_MOTION_THRESHOLD
        forward_progress = self._forward_progress(frame)

        high_activity = upper_activity >= activity_threshold
        low_progress = forward_progress <= FORWARD_PROGRESS_THRESHOLD
        return RiskMetrics(
            upper_activity=upper_activity,
            forward_progress=forward_progress,
            high_activity=high_activity,
            low_progress=low_progress,
        )

    def _pose_activity(self, track: Track) -> float:
        activity = 0.0
        if self._prev_keypoints and self._prev_center:
            center_dx = track.center[0] - self._prev_center[0]
            center_dy = track.center[1] - self._prev_center[1]
            displacements: list[float] = []
            for name in UPPER_BODY_KEYS:
                if name not in track.keypoints or name not in self._prev_keypoints:
                    continue
                x, y = track.keypoints[name]
                px, py = self._prev_keypoints[name]
                adjusted_dx = (x - px) - center_dx
                adjusted_dy = (y - py) - center_dy
                displacements.append(math.hypot(adjusted_dx, adjusted_dy))
            if displacements:
                activity = sum(displacements) / len(displacements) / _bbox_diag(track)
        self._prev_keypoints = dict(track.keypoints)
        self._prev_center = track.center
        return activity

    def _motion_activity(self, frame: Frame, track: Track) -> float:
        if self._prev_gray is None:
            return 0.0
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = track.bbox
        height = max(1, y2 - y1)
        upper_y2 = y1 + max(1, height // 2)
        frame_h, frame_w = gray.shape[:2]
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y1 = max(0, min(frame_h - 1, y1))
        upper_y2 = max(y1 + 1, min(frame_h, upper_y2))
        current = gray[y1:upper_y2, x1:x2]
        previous = self._prev_gray[y1:upper_y2, x1:x2]
        if current.size == 0 or previous.size == 0:
            return 0.0
        diff = cv2.absdiff(current, previous)
        return float(diff.mean() / 255.0)

    def _forward_progress(self, frame: Frame) -> float:
        if len(self.centers) < 2:
            return 0.0
        lookback = min(
            len(self.centers),
            max(2, int(round(self.config.ai_fps * FORWARD_PROGRESS_LOOKBACK_SECONDS))),
        )
        start = self.centers[-lookback]
        end = self.centers[-1]
        return math.hypot(end[0] - start[0], end[1] - start[1]) / _frame_diag(frame)
