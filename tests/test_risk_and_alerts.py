from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from leuco_live_demo.alerts import AlertManager
from leuco_live_demo.config import load_config
from leuco_live_demo.inference import MockInferenceBackend
from leuco_live_demo.models import DecisionState, Frame, RiskMetrics, Track
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


def track_with_bbox(bbox) -> Track:
    return Track(
        track_id=1,
        bbox=bbox,
        confidence=0.9,
        keypoints={},
        center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
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
        self.assertGreaterEqual(decision.high_risk_frames, 34)
        self.assertTrue(decision.should_alert)
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
        self.assertEqual(manager.maybe_send(decision).state, "cooldown")


if __name__ == "__main__":
    unittest.main()
