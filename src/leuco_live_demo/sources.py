from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time

import cv2
import numpy as np

from .config import AppConfig
from .mock_scene import MockPerson, mock_person
from .models import Frame

LOGGER = logging.getLogger(__name__)


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
        person = mock_person(index, self.scenario, self.width, self.height)
        if person is None:
            return image
        self._draw_person(image, person)
        return image

    def _draw_person(self, image, person: MockPerson) -> None:
        cx, cy = int(person.center[0]), int(person.center[1])
        color = (238, 238, 236)
        points = {name: (int(x), int(y)) for name, (x, y) in person.keypoints.items()}
        cv2.circle(image, points["nose"], 15, color, -1)
        cv2.line(image, points["left_shoulder"], points["right_shoulder"], color, 6)
        cv2.line(image, (cx, points["left_shoulder"][1]), (cx, cy + 58), color, 6)
        cv2.line(image, points["left_shoulder"], points["left_wrist"], color, 5)
        cv2.line(image, points["right_shoulder"], points["right_wrist"], color, 5)
        cv2.ellipse(image, (cx, cy + 55), (58, 14), 0, 0, 360, (205, 130, 60), 2)


class VideoFileSource(VideoSource):
    def __init__(self, path: str) -> None:
        self.path = path
        LOGGER.info("opening video file source: %s", path)
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
    def __init__(
        self,
        rtsp_url: str,
        backend: str = "auto",
        gstreamer_pipeline: str = "auto",
        capture_factory=None,
        threaded: bool = True,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.backend = backend
        self.gstreamer_pipeline = gstreamer_pipeline
        self._capture_factory = capture_factory or cv2.VideoCapture
        self.capture, self.backend_name, initial_image = self._open_verified_capture()
        self._threaded = threaded
        self._closed = False
        self._lock = threading.Lock()
        self._latest_image = initial_image
        self._latest_timestamp = time.time()
        self._latest_index = 0
        self._last_returned_index = -1
        self._reader_errors = 0
        if self._threaded:
            self._thread = threading.Thread(target=self._reader_loop, name="leuco-rtsp-reader", daemon=True)
            self._thread.start()
            LOGGER.info("RTSP latest-frame reader started using %s", self.backend_name)
        else:
            self._thread = None

    def read(self) -> Frame | None:
        if self._threaded:
            return self._read_threaded()
        return self._read_direct()

    def close(self) -> None:
        self._closed = True
        self.capture.release()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _read_threaded(self) -> Frame | None:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            with self._lock:
                if self._latest_index > self._last_returned_index:
                    self._last_returned_index = self._latest_index
                    return Frame(
                        image=self._latest_image,
                        timestamp=self._latest_timestamp,
                        index=self._latest_index,
                    )
                if self._closed:
                    return None
            time.sleep(0.01)
        return None

    def _read_direct(self) -> Frame | None:
        with self._lock:
            if self._latest_index > self._last_returned_index:
                self._last_returned_index = self._latest_index
                return Frame(
                    image=self._latest_image,
                    timestamp=self._latest_timestamp,
                    index=self._latest_index,
                )
        ok, image = self.capture.read()
        if not ok or image is None:
            return None
        with self._lock:
            self._latest_index += 1
            self._latest_timestamp = time.time()
            self._latest_image = image
            self._last_returned_index = self._latest_index
            return Frame(image=image, timestamp=self._latest_timestamp, index=self._latest_index)

    def _reader_loop(self) -> None:
        next_index = 1
        while not self._closed:
            ok, image = self.capture.read()
            if not ok or image is None:
                self._reader_errors += 1
                if self._reader_errors in {1, 30, 120}:
                    LOGGER.warning(
                        "RTSP reader has not received a frame (%s consecutive failed reads)",
                        self._reader_errors,
                    )
                time.sleep(0.05)
                continue

            if self._reader_errors:
                LOGGER.info("RTSP reader recovered after %s failed reads", self._reader_errors)
                self._reader_errors = 0
            with self._lock:
                self._latest_image = image
                self._latest_timestamp = time.time()
                self._latest_index = next_index
            next_index += 1

    def _open_verified_capture(self):
        failures: list[str] = []
        for name, target, api_preference in self._capture_candidates():
            LOGGER.info("opening RTSP source with %s", name)
            capture = self._capture_factory(target, api_preference)
            if not capture.isOpened():
                LOGGER.warning("%s RTSP open failed", name)
                self._release_capture(capture)
                failures.append(f"{name}: open failed")
                continue

            self._set_low_latency_buffer(capture, name)
            ok, image = capture.read()
            if ok and image is not None and getattr(image, "size", 0) > 0:
                LOGGER.info("%s RTSP delivered initial frame shape=%s", name, image.shape[:2])
                return capture, name, image

            LOGGER.warning("%s RTSP opened but delivered no initial frame", name)
            self._release_capture(capture)
            failures.append(f"{name}: no initial frame")

        detail = "; ".join(failures) if failures else f"unsupported backend {self.backend}"
        raise RuntimeError(f"Could not read initial RTSP frame ({detail})")

    def _capture_candidates(self):
        gstreamer_candidates = self._gstreamer_candidates()
        ffmpeg = ("FFmpeg", self.rtsp_url, cv2.CAP_FFMPEG)
        opencv_direct = ("OpenCV direct", self.rtsp_url, cv2.CAP_ANY)
        if self.backend == "gstreamer":
            return gstreamer_candidates
        if self.backend == "ffmpeg":
            return [ffmpeg]
        return [*gstreamer_candidates, opencv_direct]

    def _gstreamer_candidates(self):
        specs = self._gstreamer_candidate_specs()
        return [
            (
                f"GStreamer {mode} {transport}",
                self._gstreamer_pipeline(self.rtsp_url, mode, transport),
                cv2.CAP_GSTREAMER,
            )
            for mode, transport in specs
        ]

    def _gstreamer_candidate_specs(self) -> list[tuple[str, str]]:
        if self.gstreamer_pipeline == "auto":
            return [
                ("h264", "forced-tcp"),
                ("h264", "default"),
                ("h265", "forced-tcp"),
                ("h265", "default"),
                ("decodebin", "forced-tcp"),
                ("decodebin", "default"),
            ]
        if self.gstreamer_pipeline == "uridecodebin":
            return [("uridecodebin", "default")]
        return [
            (self.gstreamer_pipeline, "forced-tcp"),
            (self.gstreamer_pipeline, "default"),
        ]

    @staticmethod
    def _set_low_latency_buffer(capture, backend_name: str) -> None:
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:  # noqa: BLE001 - not all backends support this property
            LOGGER.debug("%s did not accept CAP_PROP_BUFFERSIZE=1: %s", backend_name, exc)

    @staticmethod
    def _release_capture(capture) -> None:
        try:
            capture.release()
        except Exception:  # noqa: BLE001 - release is best-effort cleanup
            LOGGER.debug("capture release failed", exc_info=True)

    @staticmethod
    def _gstreamer_pipeline(rtsp_url: str, mode: str, transport: str) -> str:
        safe_url = rtsp_url.replace('"', '\\"')
        appsink = "appsink drop=true sync=false max-buffers=1"
        if transport not in {"forced-tcp", "default"}:
            raise ValueError(f"Unsupported GStreamer transport: {transport}")
        rtspsrc_transport = " protocols=tcp" if transport == "forced-tcp" else ""
        rtspsrc = f'rtspsrc location="{safe_url}"{rtspsrc_transport} latency=200'
        if mode == "h264":
            return (
                f"{rtspsrc} ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
                f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
            )
        if mode == "h265":
            return (
                f"{rtspsrc} ! "
                "rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! "
                f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
            )
        if mode == "decodebin":
            return (
                f"{rtspsrc} ! "
                f"decodebin ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
            )
        if mode == "uridecodebin":
            return f'uridecodebin uri="{safe_url}" ! videoconvert ! video/x-raw,format=BGR ! {appsink}'
        raise ValueError(f"Unsupported GStreamer pipeline mode: {mode}")


def create_source(config: AppConfig) -> VideoSource:
    if config.source == "mock":
        LOGGER.info("creating mock video source scenario=%s", config.mock_scenario)
        return MockVideoSource(scenario=config.mock_scenario)
    if config.source == "video":
        if config.video_path is None:
            raise ValueError("LEUCO_SOURCE=video requires LEUCO_VIDEO_PATH")
        return VideoFileSource(str(config.video_path))
    if config.source == "rtsp":
        LOGGER.info(
            "creating RTSP video source subtype=%s backend=%s gstreamer_pipeline=%s",
            config.rtsp_subtype,
            config.rtsp_backend,
            config.gstreamer_pipeline,
        )
        return RTSPSource(
            config.rtsp_url,
            backend=config.rtsp_backend,
            gstreamer_pipeline=config.gstreamer_pipeline,
        )
    raise ValueError(f"Unsupported source: {config.source}")
