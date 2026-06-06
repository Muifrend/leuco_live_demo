from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from leuco_live_demo.config import load_config
from leuco_live_demo.models import AlertState, DecisionState, PoolState, RiskMetrics, Track
from leuco_live_demo.overlay import build_status, draw_overlay


class OverlayTests(unittest.TestCase):
    def test_status_exposes_polygon_gate_fields_and_overlay_draws(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[
                    "--pool-gate",
                    "polygon",
                    "--pool-polygon",
                    "10,10;100,10;100,80;10,80",
                    "--inference-roi",
                    "5,5,120,90",
                ],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )
        decision = DecisionState(
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
        alert = AlertState(state="idle", enabled=False, cooldown_seconds=60)
        track = Track(
            track_id=7,
            bbox=(30, 20, 50, 70),
            confidence=0.8,
            keypoints={},
            center=(40.0, 45.0),
        )
        pool = PoolState(
            mode="polygon",
            in_pool=True,
            active=True,
            label="Inside pool polygon",
            polygon=((10.0, 10.0), (100.0, 10.0), (100.0, 80.0), (10.0, 80.0)),
            test_point=(40.0, 70.0),
        )

        status = build_status(config, decision, alert, track, pool, capture_fps=7.5)
        payload = status.as_dict()
        canvas = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = draw_overlay(canvas, status, track, pool)

        self.assertTrue(payload["pool_gate_active"])
        self.assertEqual(payload["pool_polygon"], pool.polygon)
        self.assertEqual(payload["pool_test_point"], pool.test_point)
        self.assertGreater(int(annotated.sum()), 0)


if __name__ == "__main__":
    unittest.main()
