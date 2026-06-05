from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping
from urllib.parse import quote

CONTROL_AMCREST_ENV = Path("/mnt/ssd/projects/drowning_detection/.amcrest")
DEMO_ENV = Path(".env")

DEFAULTS: dict[str, str] = {
    "LEUCO_HTTP_HOST": "0.0.0.0",
    "LEUCO_HTTP_PORT": "8080",
    "LEUCO_SOURCE": "mock",
    "LEUCO_INFERENCE_BACKEND": "mock",
    "LEUCO_POOL_GATE": "disabled",
    "LEUCO_AI_FPS": "8",
    "LEUCO_DECISION_WINDOW_SECONDS": "6",
    "LEUCO_HIGH_RISK_FRAMES": "34",
    "LEUCO_ALERT_COOLDOWN_SECONDS": "60",
    "LEUCO_ALERTS_ENABLED": "0",
    "LEUCO_NTFY_URL": "https://ntfy.sh/leuco",
    "LEUCO_NTFY_TITLE": "Leuco demo alert",
    "LEUCO_MODEL_PATH": "/mnt/ssd/models/leuco/yolo11n-pose.pt",
    "LEUCO_VIDEO_PATH": "",
    "LEUCO_MOCK_SCENARIO": "normal",
    "LEUCO_RTSP_SUBTYPE": "0",
}

VALID_SOURCES = {"mock", "rtsp", "video"}
VALID_BACKENDS = {"mock", "yolo_pose", "motion_box"}
VALID_POOL_GATES = {"disabled", "full_frame_stub"}


@dataclass(frozen=True)
class AppConfig:
    http_host: str
    http_port: int
    source: str
    inference_backend: str
    pool_gate: str
    ai_fps: float
    decision_window_seconds: float
    high_risk_frames: int
    alert_cooldown_seconds: float
    alerts_enabled: bool
    ntfy_url: str
    ntfy_title: str
    model_path: Path
    video_path: Path | None
    mock_scenario: str
    amcrest_user: str
    amcrest_pass: str
    amcrest_ip: str
    amcrest_rtsp_main: str
    amcrest_rtsp_sub: str
    rtsp_subtype: str

    @property
    def window_size(self) -> int:
        return max(1, int(round(self.ai_fps * self.decision_window_seconds)))

    @property
    def rtsp_url(self) -> str:
        if self.rtsp_subtype == "1" and self.amcrest_rtsp_sub:
            return self.amcrest_rtsp_sub
        if self.rtsp_subtype == "0" and self.amcrest_rtsp_main:
            return self.amcrest_rtsp_main
        if not (self.amcrest_user and self.amcrest_pass and self.amcrest_ip):
            raise ValueError("RTSP source requires AMCREST_USER, AMCREST_PASS, and AMCREST_IP")
        user = quote(self.amcrest_user, safe="")
        password = quote(self.amcrest_pass, safe="")
        ip = self.amcrest_ip
        subtype = quote(self.rtsp_subtype, safe="")
        return f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype={subtype}"


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the Leuco live demo")
    parser.add_argument("--host", dest="LEUCO_HTTP_HOST")
    parser.add_argument("--port", dest="LEUCO_HTTP_PORT", type=int)
    parser.add_argument("--source", dest="LEUCO_SOURCE", choices=sorted(VALID_SOURCES))
    parser.add_argument("--backend", dest="LEUCO_INFERENCE_BACKEND", choices=sorted(VALID_BACKENDS))
    parser.add_argument("--pool-gate", dest="LEUCO_POOL_GATE", choices=sorted(VALID_POOL_GATES))
    parser.add_argument("--alerts-enabled", dest="LEUCO_ALERTS_ENABLED", action="store_true")
    parser.add_argument("--alerts-disabled", dest="LEUCO_ALERTS_ENABLED", action="store_false")
    parser.set_defaults(LEUCO_ALERTS_ENABLED=None)
    parser.add_argument("--mock-scenario", dest="LEUCO_MOCK_SCENARIO")
    parser.add_argument("--video-path", dest="LEUCO_VIDEO_PATH")
    parser.add_argument("--model-path", dest="LEUCO_MODEL_PATH")
    parser.add_argument("--rtsp-subtype", dest="LEUCO_RTSP_SUBTYPE", choices=["0", "1"])
    return parser


def _namespace_values(args: Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in vars(args).items():
        if value is None:
            continue
        values[key] = "1" if isinstance(value, bool) and value else str(value)
    return values


def load_config(
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    demo_env: Path = DEMO_ENV,
    amcrest_env: Path = CONTROL_AMCREST_ENV,
) -> AppConfig:
    env = dict(environ if environ is not None else os.environ)
    parser = build_arg_parser()
    cli_values = _namespace_values(parser.parse_args(argv))

    values: dict[str, str] = {}
    values.update(DEFAULTS)
    values.update(load_env_file(amcrest_env))
    values.update(load_env_file(demo_env))
    values.update(env)
    values.update(cli_values)

    source = values["LEUCO_SOURCE"].strip()
    backend = values["LEUCO_INFERENCE_BACKEND"].strip()
    pool_gate = values["LEUCO_POOL_GATE"].strip()
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid LEUCO_SOURCE: {source}")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid LEUCO_INFERENCE_BACKEND: {backend}")
    if pool_gate not in VALID_POOL_GATES:
        raise ValueError(f"Invalid LEUCO_POOL_GATE: {pool_gate}")

    video_raw = values.get("LEUCO_VIDEO_PATH", "").strip()
    ntfy_url = (values.get("LEUCO_NTFY_URL", "") or values.get("NTFY_URL", "")).strip()
    topic = (values.get("LEUCO_NTFY_TOPIC", "") or values.get("NTFY_TOPIC", "")).strip()
    if not ntfy_url and topic:
        ntfy_url = f"https://ntfy.sh/{topic}"

    return AppConfig(
        http_host=values["LEUCO_HTTP_HOST"].strip(),
        http_port=int(values["LEUCO_HTTP_PORT"]),
        source=source,
        inference_backend=backend,
        pool_gate=pool_gate,
        ai_fps=float(values["LEUCO_AI_FPS"]),
        decision_window_seconds=float(values["LEUCO_DECISION_WINDOW_SECONDS"]),
        high_risk_frames=int(values["LEUCO_HIGH_RISK_FRAMES"]),
        alert_cooldown_seconds=float(values["LEUCO_ALERT_COOLDOWN_SECONDS"]),
        alerts_enabled=parse_bool(values["LEUCO_ALERTS_ENABLED"]),
        ntfy_url=ntfy_url,
        ntfy_title=values.get("LEUCO_NTFY_TITLE", "Leuco demo alert"),
        model_path=Path(values["LEUCO_MODEL_PATH"]),
        video_path=Path(video_raw) if video_raw else None,
        mock_scenario=values["LEUCO_MOCK_SCENARIO"].strip(),
        amcrest_user=values.get("AMCREST_USER", "").strip(),
        amcrest_pass=values.get("AMCREST_PASS", "").strip(),
        amcrest_ip=values.get("AMCREST_IP", "").strip(),
        amcrest_rtsp_main=values.get("AMCREST_RTSP_MAIN", "").strip(),
        amcrest_rtsp_sub=values.get("AMCREST_RTSP_SUB", "").strip(),
        rtsp_subtype=values.get("LEUCO_RTSP_SUBTYPE", "0").strip(),
    )
