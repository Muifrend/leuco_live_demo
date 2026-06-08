from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from leuco_live_demo.alerts import AlertManager
from leuco_live_demo.config import load_config
from leuco_live_demo.inference import MockInferenceBackend
from leuco_live_demo.models import DecisionState, Frame, PoolState, RiskMetrics, Track
from leuco_live_demo.pool_gate import PoolGate
from leuco_live_demo.risk import RiskEngine
from leuco_live_demo.sources import MockVideoSource
from leuco_live_demo.tracker import OnePersonTracker


def config_for(**overrides: str):
    env = {key: str(value) for key, value in overrides.items()}
    with tempfile.TemporaryDirectory() as tmp:
        return load_config(argv=[], environ=env, demo_env=Path(tmp) / ".env", amcrest_env=Path(tmp) / ".amcrest")


def run_mock_scenario(scenario: str, pool_gate: str, frames: int = 48):
    config = config_for(
        LEUCO_SOURCE="mock",
        LEUCO_INFERENCE_BACKEND="mock",
        LEUCO_MOCK_SCENARIO=scenario,
        LEUCO_POOL_GATE=pool_gate,
        LEUCO_ALERTS_ENABLED="0",
    )
    source = MockVideoSource(scenario=scenario)
    backend = MockInferenceBackend(scenario)
    tracker = OnePersonTracker()
    gate = PoolGate(pool_gate)
    risk = RiskEngine(config)
    alerts = AlertManager(config)
    decision = None
    alert = None

    for index in range(frames):
        image = source._render(index)  # deterministic test fixture
        frame = Frame(image=image, timestamp=float(index) / config.ai_fps, index=index)
        result = backend.infer(frame)
        track = tracker.update(result.detections)
        pool = gate.evaluate(track)
        decision = risk.process(frame, track, pool)
        alert = alerts.maybe_send(decision)

    assert decision is not None
    assert alert is not None
    return decision, alert


def track_with_bbox(bbox, keypoints=None) -> Track:
    return Track(
        track_id=1,
        bbox=bbox,
        confidence=0.9,
        keypoints=keypoints or {},
        center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
    )


def frame_at(index: int, ai_fps: float = 8.0) -> Frame:
    return Frame(
        image=np.zeros((80, 80, 3), dtype=np.uint8),
        timestamp=float(index) / ai_fps,
        index=index,
    )


class RiskAndAlertTests(unittest.TestCase):
    def test_disabled_gate_never_activates_risk(self) -> None:
        decision, alert = run_mock_scenario("alert", "disabled")

        self.assertFalse(decision.risk_active)
        self.assertEqual(decision.high_risk_frames, 0)
        self.assertEqual(alert.state, "idle")

    def test_normal_motion_does_not_reach_alert_threshold(self) -> None:
        decision, alert = run_mock_scenario("normal", "full_frame_stub")

        self.assertTrue(decision.risk_active)
        self.assertLess(decision.high_risk_frames, 34)
        self.assertEqual(alert.state, "idle")

    def test_high_activity_low_progress_reaches_dry_run_alert(self) -> None:
        decision, alert = run_mock_scenario("alert", "full_frame_stub")

        self.assertTrue(decision.risk_active)
        self.assertGreaterEqual(decision.high_risk_frames, 15)
        self.assertTrue(decision.should_alert)
        self.assertEqual(alert.state, "dry_run")

    def test_fifteen_high_risk_frames_in_window_triggers_alert_even_if_current_normal(self) -> None:
        config = config_for(
            LEUCO_POOL_GATE="full_frame_stub",
            LEUCO_HIGH_RISK_FRAMES="15",
            LEUCO_ALERTS_ENABLED="0",
        )
        risk = RiskEngine(config)
        risk.window.extend(([True] * 15) + ([False] * 32))
        pool = PoolState(
            mode="full_frame_stub",
            in_pool=True,
            active=True,
            label="Full-frame pool stub",
        )
        track = track_with_bbox((20, 20, 40, 50))

        decision = risk.process(frame_at(0, config.ai_fps), track, pool)
        alert = AlertManager(config).maybe_send(decision)

        self.assertEqual(decision.high_risk_frames, 15)
        self.assertTrue(decision.should_alert)
        self.assertEqual(decision.alert_reason, "high-risk frame count")
        self.assertEqual(alert.state, "dry_run")

    def test_polygon_gate_outside_track_keeps_risk_inactive(self) -> None:
        config = config_for(LEUCO_POOL_GATE="polygon")
        gate = PoolGate("polygon", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        risk = RiskEngine(config)
        frame = Frame(image=np.zeros((80, 80, 3), dtype=np.uint8), timestamp=0.0, index=0)
        track = track_with_bbox((20, 1, 24, 8))

        pool = gate.evaluate(track, frame_size=(80, 80))
        decision = risk.process(frame, track, pool)

        self.assertFalse(pool.in_pool)
        self.assertFalse(decision.risk_active)

    def test_polygon_gate_inside_track_activates_risk(self) -> None:
        config = config_for(LEUCO_POOL_GATE="polygon")
        gate = PoolGate("polygon", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        risk = RiskEngine(config)
        frame = Frame(image=np.zeros((80, 80, 3), dtype=np.uint8), timestamp=0.0, index=0)
        track = track_with_bbox((4, 1, 6, 8))

        pool = gate.evaluate(track, frame_size=(80, 80))
        decision = risk.process(frame, track, pool)

        self.assertTrue(pool.in_pool)
        self.assertTrue(decision.risk_active)

    def test_lost_swimmer_after_inside_track_reaches_alert(self) -> None:
        config = config_for(
            LEUCO_POOL_GATE="full_frame_stub",
            LEUCO_LOST_SWIMMER_FRAMES="3",
            LEUCO_DISTRESS_PERSIST_SECONDS="0.25",
            LEUCO_ALERTS_ENABLED="0",
        )
        risk = RiskEngine(config)
        inside_pool = PoolState(
            mode="full_frame_stub",
            in_pool=True,
            active=True,
            label="Full-frame pool stub",
        )
        missing_pool = PoolState(
            mode="full_frame_stub",
            in_pool=False,
            active=True,
            label="Full-frame pool stub",
        )

        decision = risk.process(
            frame_at(0, config.ai_fps),
            track_with_bbox((20, 20, 40, 50)),
            inside_pool,
        )
        self.assertFalse(decision.should_alert)

        for index in range(1, 6):
            decision = risk.process(frame_at(index, config.ai_fps), None, missing_pool)

        self.assertTrue(decision.risk_active)
        self.assertTrue(decision.in_pool)
        self.assertTrue(decision.should_alert)
        self.assertEqual(decision.alert_reason, "lost swimmer after in-pool track")
        self.assertGreaterEqual(decision.metrics.lost_visibility_frames, 3)

    def test_confirmed_exit_clears_lost_swimmer_alert_path(self) -> None:
        config = config_for(
            LEUCO_POOL_GATE="polygon",
            LEUCO_LOST_SWIMMER_FRAMES="2",
            LEUCO_DISTRESS_PERSIST_SECONDS="0.25",
            LEUCO_CONFIRMED_EXIT_SECONDS="0.25",
            LEUCO_ALERTS_ENABLED="0",
        )
        risk = RiskEngine(config)
        inside_pool = PoolState(mode="polygon", in_pool=True, active=True, label="Inside pool polygon")
        outside_pool = PoolState(mode="polygon", in_pool=False, active=True, label="Outside pool polygon")

        risk.process(frame_at(0, config.ai_fps), track_with_bbox((20, 20, 40, 50)), inside_pool)
        risk.process(frame_at(1, config.ai_fps), track_with_bbox((60, 20, 70, 50)), outside_pool)
        decision = risk.process(frame_at(2, config.ai_fps), track_with_bbox((62, 20, 72, 50)), outside_pool)
        self.assertFalse(decision.in_pool)
        self.assertFalse(decision.risk_active)

        for index in range(3, 7):
            decision = risk.process(frame_at(index, config.ai_fps), None, outside_pool)

        self.assertFalse(decision.should_alert)
        self.assertEqual(decision.metrics.lost_visibility_frames, 0)

    def test_yolo_partial_pose_does_not_arm_lost_visibility_by_itself(self) -> None:
        config = config_for(
            LEUCO_INFERENCE_BACKEND="yolo_pose",
            LEUCO_POOL_GATE="polygon",
            LEUCO_LOST_SWIMMER_FRAMES="2",
            LEUCO_DISTRESS_PERSIST_SECONDS="0.25",
            LEUCO_ALERTS_ENABLED="0",
        )
        risk = RiskEngine(config)
        inside_pool = PoolState(mode="polygon", in_pool=True, active=True, label="Inside pool polygon")
        partial_pose_track = track_with_bbox((20, 20, 40, 50), {"nose": (30.0, 18.0)})

        for index in range(4):
            decision = risk.process(frame_at(index, config.ai_fps), partial_pose_track, inside_pool)

        self.assertFalse(decision.in_pool)
        self.assertFalse(decision.risk_active)
        self.assertFalse(decision.should_alert)
        self.assertEqual(decision.metrics.lost_visibility_frames, 0)
        self.assertEqual(decision.metrics.distress_candidate_frames, 0)
        self.assertEqual(decision.alert_reason, "")

    def test_yolo_partial_pose_counts_after_reliable_in_pool_detection(self) -> None:
        config = config_for(
            LEUCO_INFERENCE_BACKEND="yolo_pose",
            LEUCO_POOL_GATE="polygon",
            LEUCO_LOST_SWIMMER_FRAMES="2",
            LEUCO_DISTRESS_PERSIST_SECONDS="0.25",
            LEUCO_ALERTS_ENABLED="0",
        )
        risk = RiskEngine(config)
        inside_pool = PoolState(mode="polygon", in_pool=True, active=True, label="Inside pool polygon")
        reliable_pose_track = track_with_bbox(
            (20, 20, 40, 50),
            {
                "nose": (30.0, 18.0),
                "left_shoulder": (25.0, 28.0),
                "right_shoulder": (35.0, 28.0),
            },
        )
        partial_pose_track = track_with_bbox((20, 20, 40, 50), {"nose": (30.0, 18.0)})

        risk.process(frame_at(0, config.ai_fps), reliable_pose_track, inside_pool)
        for index in range(1, 5):
            decision = risk.process(frame_at(index, config.ai_fps), partial_pose_track, inside_pool)

        self.assertTrue(decision.should_alert)
        self.assertEqual(decision.alert_reason, "lost swimmer after in-pool track")
        self.assertGreaterEqual(decision.metrics.lost_visibility_frames, 2)

    def test_alert_cooldown_and_idle_reset(self) -> None:
        config = config_for(LEUCO_ALERTS_ENABLED="0")
        now = [100.0]
        manager = AlertManager(config=config, clock=lambda: now[0])
        decision = DecisionState(
            person_detected=True,
            in_pool=True,
            risk_active=True,
            risk_state="high_risk",
            high_risk_frames=34,
            window_size=48,
            window_seconds=6,
            ai_fps=8,
            should_alert=True,
            metrics=RiskMetrics(high_activity=True, low_progress=True),
        )

        self.assertEqual(manager.maybe_send(decision).state, "dry_run")
        now[0] = 110.0
        self.assertEqual(manager.maybe_send(decision).state, "cooldown")

        no_alert = DecisionState(
            person_detected=True,
            in_pool=True,
            risk_active=True,
            risk_state="normal",
            high_risk_frames=0,
            window_size=48,
            window_seconds=6,
            ai_fps=8,
            should_alert=False,
            metrics=RiskMetrics(),
        )
        self.assertEqual(manager.maybe_send(no_alert).state, "idle")

    def test_failed_alert_attempt_uses_cooldown(self) -> None:
        config = config_for(LEUCO_ALERTS_ENABLED="1")
        now = [100.0]

        def failing_opener(_request, _timeout):
            raise OSError("network unavailable")

        manager = AlertManager(config=config, opener=failing_opener, clock=lambda: now[0])
        decision = DecisionState(
            person_detected=True,
            in_pool=True,
            risk_active=True,
            risk_state="high_risk",
            high_risk_frames=34,
            window_size=48,
            window_seconds=6,
            ai_fps=8,
            should_alert=True,
            metrics=RiskMetrics(high_activity=True, low_progress=True),
        )

        failed = manager.maybe_send(decision)
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.last_error, "network unavailable")

        now[0] = 101.0
        cooldown = manager.maybe_send(decision)
        self.assertEqual(cooldown.state, "cooldown")
        self.assertEqual(cooldown.last_error, "network unavailable")


if __name__ == "__main__":
    unittest.main()
