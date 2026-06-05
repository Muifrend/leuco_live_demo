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


if __name__ == "__main__":
    unittest.main()
