from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .config import AppConfig
from .models import Frame


class VideoSource:
    def read(self) -> Frame | None:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class MockVideoSource(VideoSource):
    scenario: str = "normal"
    width: int = 960
    height: int = 540
    fps: float = 20.0

    def __post_init__(self) -> None:
        self._index = 0
        self._next_at = time.time()

    def read(self) -> Frame:
        now = time.time()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        timestamp = time.time()
        image = self._render(self._index)
        frame = Frame(image=image, timestamp=timestamp, index=self._index)
        self._index += 1
        self._next_at = timestamp + (1.0 / self.fps)
        return frame

    def _render(self, index: int):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (55, 52, 48)
        cv2.rectangle(image, (80, 70), (880, 470), (170, 104, 36), thickness=-1)
        cv2.rectangle(image, (80, 70), (880, 470), (230, 180, 90), thickness=5)
        cv2.putText(
            image,
            f"mock:{self.scenario}",
            (24, self.height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        if self.scenario == "empty":
            return image
        center, arm_phase = mock_person_pose(index, self.scenario, self.width, self.height)
        self._draw_person(image, center, arm_phase)
        return image

    def _draw_person(self, image, center: tuple[float, float], arm_phase: float) -> None:
        cx, cy = int(center[0]), int(center[1])
        shoulder_y = cy - 44
        head = (cx, cy - 75)
        left_shoulder = (cx - 28, shoulder_y)
        right_shoulder = (cx + 28, shoulder_y)
        left_hand = (int(cx - 62), int(shoulder_y + math.sin(arm_phase) * 52))
        right_hand = (int(cx + 62), int(shoulder_y - math.sin(arm_phase) * 52))
        torso_bottom = (cx, cy + 58)
        color = (238, 238, 236)
        cv2.circle(image, head, 15, color, -1)
        cv2.line(image, left_shoulder, right_shoulder, color, 6)
        cv2.line(image, (cx, shoulder_y), torso_bottom, color, 6)
        cv2.line(image, left_shoulder, left_hand, color, 5)
        cv2.line(image, right_shoulder, right_hand, color, 5)
        cv2.ellipse(image, (cx, cy + 55), (58, 14), 0, 0, 360, (205, 130, 60), 2)


class VideoFileSource(VideoSource):
    def __init__(self, path: str) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {path}")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 20.0
        self._index = 0
        self._next_at = time.time()

    def read(self) -> Frame | None:
        now = time.time()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        ok, image = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.capture.read()
            if not ok:
                return None
        timestamp = time.time()
        frame = Frame(image=image, timestamp=timestamp, index=self._index)
        self._index += 1
        self._next_at = timestamp + (1.0 / self.fps)
        return frame

    def close(self) -> None:
        self.capture.release()


class RTSPSource(VideoSource):
    def __init__(self, rtsp_url: str) -> None:
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(self._gstreamer_pipeline(rtsp_url), cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open RTSP stream with GStreamer or FFmpeg")
        self._index = 0

    def read(self) -> Frame | None:
        ok, image = self.capture.read()
        if not ok:
            return None
        frame = Frame(image=image, timestamp=time.time(), index=self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self.capture.release()

    @staticmethod
    def _gstreamer_pipeline(rtsp_url: str) -> str:
        return (
            f'rtspsrc location="{rtsp_url}" protocols=tcp latency=200 ! '
            "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )


def mock_person_pose(
    index: int,
    scenario: str,
    width: int,
    height: int,
) -> tuple[tuple[float, float], float]:
    if scenario == "high_risk" or scenario == "alert":
        return (width * 0.52, height * 0.52), index * 0.9
    if scenario == "outside":
        return (width * 0.5, height * 0.9), index * 0.15
    travel = (index * 4) % 560
    return (200 + travel, height * 0.55), index * 0.18


def create_source(config: AppConfig) -> VideoSource:
    if config.source == "mock":
        return MockVideoSource(scenario=config.mock_scenario)
    if config.source == "video":
        if config.video_path is None:
            raise ValueError("LEUCO_SOURCE=video requires LEUCO_VIDEO_PATH")
        return VideoFileSource(str(config.video_path))
    if config.source == "rtsp":
        return RTSPSource(config.rtsp_url)
    raise ValueError(f"Unsupported source: {config.source}")
