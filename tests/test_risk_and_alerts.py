from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from leuco_live_demo.alerts import AlertManager
from leuco_live_demo.config import load_config
from leuco_live_demo.inference import MockInferenceBackend
from leuco_live_demo.models import DecisionState, Frame, RiskMetrics
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


if __name__ == "__main__":
    unittest.main()
