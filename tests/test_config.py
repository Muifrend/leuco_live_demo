from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from leuco_live_demo.config import load_config


class ConfigTests(unittest.TestCase):
    def test_precedence_cli_over_env_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            amcrest = root / ".amcrest"
            demo_env = root / ".env"
            amcrest.write_text("AMCREST_USER=cam\nAMCREST_PASS=secret\nAMCREST_IP=10.0.0.2\n", encoding="utf-8")
            demo_env.write_text("LEUCO_HTTP_PORT=9000\nLEUCO_POOL_GATE=full_frame_stub\n", encoding="utf-8")

            config = load_config(
                argv=["--port", "9002"],
                environ={"LEUCO_HTTP_PORT": "9001"},
                demo_env=demo_env,
                amcrest_env=amcrest,
            )

            self.assertEqual(config.http_port, 9002)
            self.assertEqual(config.pool_gate, "full_frame_stub")
            self.assertEqual(config.amcrest_ip, "10.0.0.2")

    def test_rtsp_url_encodes_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            amcrest = Path(tmp) / ".amcrest"
            amcrest.write_text("AMCREST_USER=admin\nAMCREST_PASS=p@ss/word\nAMCREST_IP=192.0.2.10\n", encoding="utf-8")

            config = load_config(argv=[], environ={}, demo_env=Path(tmp) / "missing.env", amcrest_env=amcrest)

            self.assertEqual(
                config.rtsp_url,
                "rtsp://admin:p%40ss%2Fword@192.0.2.10:554/cam/realmonitor?channel=1&subtype=0",
            )

    def test_ntfy_url_alias_can_override_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[],
                environ={"NTFY_URL": "https://ntfy.sh/local-test"},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertEqual(config.ntfy_url, "https://ntfy.sh/local-test")

    def test_high_risk_threshold_cannot_exceed_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                load_config(
                    argv=[],
                    environ={
                        "LEUCO_AI_FPS": "2",
                        "LEUCO_DECISION_WINDOW_SECONDS": "2",
                        "LEUCO_HIGH_RISK_FRAMES": "5",
                    },
                    demo_env=Path(tmp) / ".env",
                    amcrest_env=Path(tmp) / ".amcrest",
                )

    def test_inference_roi_defaults_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertIsNone(config.inference_roi)

    def test_pool_polygon_defaults_to_roboflow_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertEqual(config.pool_polygon_reference_size, (1920, 1080))
            self.assertEqual(len(config.pool_polygon or ()), 25)
            self.assertEqual((config.pool_polygon or ())[0], (515.0, 515.0))

    def test_inference_roi_parses_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=["--inference-roi", "300,220,1080,670"],
                environ={"LEUCO_INFERENCE_ROI": "disabled"},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertEqual(config.inference_roi, (300, 220, 1080, 670))

    def test_inference_roi_reference_size_parses_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=["--inference-roi-reference-size", "1554x882"],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertEqual(config.inference_roi_reference_size, (1554, 882))

    def test_inference_roi_rejects_bad_values(self) -> None:
        cases = [
            ("1,2,3", "x1,y1,x2,y2"),
            ("1,two,3,4", "integer"),
            ("1,-2,3,4", "nonnegative"),
            ("10,20,10,40", "x2 > x1"),
            ("10,20,50,20", "y2 > y1"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for value, pattern in cases:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_config(
                            argv=[],
                            environ={"LEUCO_INFERENCE_ROI": value},
                            demo_env=Path(tmp) / ".env",
                            amcrest_env=Path(tmp) / ".amcrest",
                        )

    def test_inference_roi_reference_size_rejects_bad_values(self) -> None:
        cases = [
            ("1554", "WIDTHxHEIGHT"),
            ("wide,882", "integer"),
            ("1554,0", "greater than 0"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for value, pattern in cases:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_config(
                            argv=[],
                            environ={"LEUCO_INFERENCE_ROI_REFERENCE_SIZE": value},
                            demo_env=Path(tmp) / ".env",
                            amcrest_env=Path(tmp) / ".amcrest",
                        )

    def test_log_level_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "LEUCO_LOG_LEVEL"):
                load_config(
                    argv=[],
                    environ={"LEUCO_LOG_LEVEL": "chatty"},
                    demo_env=Path(tmp) / ".env",
                    amcrest_env=Path(tmp) / ".amcrest",
                )

    def test_pool_polygon_parses_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[
                    "--pool-gate",
                    "polygon",
                    "--pool-polygon",
                    "1,2;30,4;5,60",
                    "--pool-polygon-reference-size",
                    "100x80",
                ],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

            self.assertEqual(config.pool_gate, "polygon")
            self.assertEqual(config.pool_polygon, ((1.0, 2.0), (30.0, 4.0), (5.0, 60.0)))
            self.assertEqual(config.pool_polygon_reference_size, (100, 80))

    def test_pool_polygon_rejects_bad_values(self) -> None:
        cases = [
            ("1,2;3,4", "at least 3"),
            ("1,2;3;4,5", "x,y;x,y"),
            ("1,2;3,nope;4,5", "numeric"),
            ("1,2;-3,4;5,6", "nonnegative"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for value, pattern in cases:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_config(
                            argv=[],
                            environ={"LEUCO_POOL_POLYGON": value},
                            demo_env=Path(tmp) / ".env",
                            amcrest_env=Path(tmp) / ".amcrest",
                        )

    def test_polygon_gate_requires_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "LEUCO_POOL_POLYGON"):
                load_config(
                    argv=["--pool-gate", "polygon", "--pool-polygon", "disabled"],
                    environ={},
                    demo_env=Path(tmp) / ".env",
                    amcrest_env=Path(tmp) / ".amcrest",
                )

    def test_rtsp_backend_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=["--rtsp-backend", "ffmpeg"],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )
            self.assertEqual(config.rtsp_backend, "ffmpeg")

            with self.assertRaisesRegex(ValueError, "LEUCO_RTSP_BACKEND"):
                load_config(
                    argv=[],
                    environ={"LEUCO_RTSP_BACKEND": "other"},
                    demo_env=Path(tmp) / ".env",
                    amcrest_env=Path(tmp) / ".amcrest",
                )

    def test_gstreamer_pipeline_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=["--gstreamer-pipeline", "h265"],
                environ={},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )
            self.assertEqual(config.gstreamer_pipeline, "h265")

            with self.assertRaisesRegex(ValueError, "LEUCO_GSTREAMER_PIPELINE"):
                load_config(
                    argv=[],
                    environ={"LEUCO_GSTREAMER_PIPELINE": "magic"},
                    demo_env=Path(tmp) / ".env",
                    amcrest_env=Path(tmp) / ".amcrest",
                )


if __name__ == "__main__":
    unittest.main()
