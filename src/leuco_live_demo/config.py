from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping
from urllib.parse import quote

from .models import BBox, FrameSize

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
    "LEUCO_RTSP_BACKEND": "auto",
    "LEUCO_GSTREAMER_PIPELINE": "auto",
    "LEUCO_INFERENCE_ROI": "",
    "LEUCO_INFERENCE_ROI_REFERENCE_SIZE": "",
    "LEUCO_LOG_LEVEL": "INFO",
}

VALID_SOURCES = {"mock", "rtsp", "video"}
VALID_BACKENDS = {"mock", "yolo_pose", "motion_box"}
VALID_POOL_GATES = {"disabled", "full_frame_stub"}
VALID_RTSP_BACKENDS = {"auto", "gstreamer", "ffmpeg"}
VALID_GSTREAMER_PIPELINES = {"auto", "h264", "h265", "decodebin", "uridecodebin"}
DEFAULT_NTFY_URL = DEFAULTS["LEUCO_NTFY_URL"]


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
    rtsp_backend: str
    gstreamer_pipeline: str
    inference_roi: BBox | None
    inference_roi_reference_size: FrameSize | None
    log_level: str

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


def parse_positive_float(name: str, value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def parse_nonnegative_float(name: str, value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be 0 or greater")
    return parsed


def parse_positive_int(name: str, value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def parse_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("LEUCO_HTTP_PORT must be between 1 and 65535")
    return port


def parse_inference_roi(value: str | None) -> BBox | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "disabled":
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("LEUCO_INFERENCE_ROI must be x1,y1,x2,y2 or disabled")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("LEUCO_INFERENCE_ROI must contain four integer coordinates") from exc
    if min(x1, y1, x2, y2) < 0:
        raise ValueError("LEUCO_INFERENCE_ROI coordinates must be nonnegative")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("LEUCO_INFERENCE_ROI must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def parse_frame_size(name: str, value: str | None) -> FrameSize | None:
    raw = str(value or "").strip().lower()
    if not raw or raw == "disabled":
        return None
    normalized = raw.replace("x", ",")
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be WIDTHxHEIGHT, WIDTH,HEIGHT, or disabled")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must contain integer width and height") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} width and height must be greater than 0")
    return (width, height)


def parse_log_level(value: str) -> str:
    parsed = value.strip().upper()
    if parsed not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("LEUCO_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    return parsed


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


def _first_non_empty(values: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key, "").strip()
        if value:
            return value
    return ""


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
    parser.add_argument("--rtsp-backend", dest="LEUCO_RTSP_BACKEND", choices=sorted(VALID_RTSP_BACKENDS))
    parser.add_argument(
        "--gstreamer-pipeline",
        dest="LEUCO_GSTREAMER_PIPELINE",
        choices=sorted(VALID_GSTREAMER_PIPELINES),
    )
    parser.add_argument("--inference-roi", dest="LEUCO_INFERENCE_ROI")
    parser.add_argument("--inference-roi-reference-size", dest="LEUCO_INFERENCE_ROI_REFERENCE_SIZE")
    parser.add_argument("--log-level", dest="LEUCO_LOG_LEVEL")
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
    rtsp_backend = values["LEUCO_RTSP_BACKEND"].strip().lower()
    gstreamer_pipeline = values["LEUCO_GSTREAMER_PIPELINE"].strip().lower()
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid LEUCO_SOURCE: {source}")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid LEUCO_INFERENCE_BACKEND: {backend}")
    if pool_gate not in VALID_POOL_GATES:
        raise ValueError(f"Invalid LEUCO_POOL_GATE: {pool_gate}")
    if rtsp_backend not in VALID_RTSP_BACKENDS:
        raise ValueError(f"Invalid LEUCO_RTSP_BACKEND: {rtsp_backend}")
    if gstreamer_pipeline not in VALID_GSTREAMER_PIPELINES:
        raise ValueError(f"Invalid LEUCO_GSTREAMER_PIPELINE: {gstreamer_pipeline}")

    http_port = parse_port(values["LEUCO_HTTP_PORT"])
    ai_fps = parse_positive_float("LEUCO_AI_FPS", values["LEUCO_AI_FPS"])
    decision_window_seconds = parse_positive_float(
        "LEUCO_DECISION_WINDOW_SECONDS",
        values["LEUCO_DECISION_WINDOW_SECONDS"],
    )
    window_size = max(1, int(round(ai_fps * decision_window_seconds)))
    high_risk_frames = parse_positive_int("LEUCO_HIGH_RISK_FRAMES", values["LEUCO_HIGH_RISK_FRAMES"])
    if high_risk_frames > window_size:
        raise ValueError(
            "LEUCO_HIGH_RISK_FRAMES cannot exceed the processed decision window "
            f"({high_risk_frames} > {window_size})"
        )
    alert_cooldown_seconds = parse_nonnegative_float(
        "LEUCO_ALERT_COOLDOWN_SECONDS",
        values["LEUCO_ALERT_COOLDOWN_SECONDS"],
    )

    video_raw = values.get("LEUCO_VIDEO_PATH", "").strip()
    ntfy_url = _first_non_empty(values, "LEUCO_NTFY_URL")
    if ntfy_url == DEFAULT_NTFY_URL:
        ntfy_url = _first_non_empty(values, "NTFY_URL", "LEUCO_NTFY_URL")
    topic = _first_non_empty(values, "LEUCO_NTFY_TOPIC", "NTFY_TOPIC")
    if not ntfy_url and topic:
        ntfy_url = f"https://ntfy.sh/{topic}"
    inference_roi = parse_inference_roi(values.get("LEUCO_INFERENCE_ROI", ""))
    inference_roi_reference_size = parse_frame_size(
        "LEUCO_INFERENCE_ROI_REFERENCE_SIZE",
        values.get("LEUCO_INFERENCE_ROI_REFERENCE_SIZE", ""),
    )
    log_level = parse_log_level(values.get("LEUCO_LOG_LEVEL", "INFO"))

    return AppConfig(
        http_host=values["LEUCO_HTTP_HOST"].strip(),
        http_port=http_port,
        source=source,
        inference_backend=backend,
        pool_gate=pool_gate,
        ai_fps=ai_fps,
        decision_window_seconds=decision_window_seconds,
        high_risk_frames=high_risk_frames,
        alert_cooldown_seconds=alert_cooldown_seconds,
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
        rtsp_backend=rtsp_backend,
        gstreamer_pipeline=gstreamer_pipeline,
        inference_roi=inference_roi,
        inference_roi_reference_size=inference_roi_reference_size,
        log_level=log_level,
    )
