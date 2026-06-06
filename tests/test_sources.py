from __future__ import annotations

import logging
import unittest

import cv2
import numpy as np

from leuco_live_demo.sources import RTSPSource


class FakeCapture:
    def __init__(self, opened: bool, frames) -> None:
        self.opened = opened
        self.frames = list(frames)
        self.released = False
        self.properties: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, prop: int, value: float) -> bool:
        self.properties.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class RTSPSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._logger = logging.getLogger("leuco_live_demo.sources")
        self._logger_disabled = self._logger.disabled
        self._logger.disabled = True

    def tearDown(self) -> None:
        self._logger.disabled = self._logger_disabled

    def test_auto_tries_ordered_gstreamer_candidates_then_opencv_direct(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        calls: list[tuple[str, int]] = []
        gstreamer_captures: list[FakeCapture] = []
        direct_capture = None

        def factory(target, api_preference):
            nonlocal direct_capture
            calls.append((target, api_preference))
            capture = FakeCapture(opened=True, frames=[] if api_preference == cv2.CAP_GSTREAMER else [image])
            if api_preference == cv2.CAP_GSTREAMER:
                gstreamer_captures.append(capture)
            else:
                direct_capture = capture
            return capture

        source = RTSPSource("rtsp://camera", backend="auto", capture_factory=factory, threaded=False)

        self.assertEqual(source.backend_name, "OpenCV direct")
        self.assertEqual(len(gstreamer_captures), 6)
        self.assertTrue(all(capture.released for capture in gstreamer_captures))
        self.assertIsNotNone(direct_capture)
        self.assertFalse(direct_capture.released)
        self.assertIn("rtph264depay", calls[0][0])
        self.assertIn("protocols=tcp", calls[0][0])
        self.assertIn("rtph264depay", calls[1][0])
        self.assertNotIn("protocols=tcp", calls[1][0])
        self.assertIn("rtph265depay", calls[2][0])
        self.assertIn("protocols=tcp", calls[2][0])
        self.assertIn("rtph265depay", calls[3][0])
        self.assertNotIn("protocols=tcp", calls[3][0])
        self.assertIn("decodebin", calls[4][0])
        self.assertIn("protocols=tcp", calls[4][0])
        self.assertIn("decodebin", calls[5][0])
        self.assertNotIn("protocols=tcp", calls[5][0])
        self.assertEqual(calls[-1], ("rtsp://camera", cv2.CAP_ANY))
        frame = source.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.index, 0)
        self.assertIs(frame.image, image)

    def test_gstreamer_h265_mode_skips_other_backends(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        calls: list[tuple[str, int]] = []

        def factory(target, api_preference):
            calls.append((target, api_preference))
            return FakeCapture(opened=True, frames=[image])

        source = RTSPSource(
            "rtsp://camera",
            backend="gstreamer",
            gstreamer_pipeline="h265",
            capture_factory=factory,
            threaded=False,
        )

        self.assertEqual(source.backend_name, "GStreamer h265 forced-tcp")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], cv2.CAP_GSTREAMER)
        self.assertIn("rtph265depay", calls[0][0])
        self.assertNotIn("rtph264depay", calls[0][0])
        self.assertIn("protocols=tcp", calls[0][0])

    def test_gstreamer_selected_mode_falls_back_from_tcp_to_default_transport(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        calls: list[tuple[str, int]] = []

        def factory(target, api_preference):
            calls.append((target, api_preference))
            frames = [] if len(calls) == 1 else [image]
            return FakeCapture(opened=True, frames=frames)

        source = RTSPSource(
            "rtsp://camera",
            backend="gstreamer",
            gstreamer_pipeline="h264",
            capture_factory=factory,
            threaded=False,
        )

        self.assertEqual(source.backend_name, "GStreamer h264 default")
        self.assertEqual(len(calls), 2)
        self.assertIn("protocols=tcp", calls[0][0])
        self.assertNotIn("protocols=tcp", calls[1][0])

    def test_ffmpeg_backend_skips_gstreamer(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        calls: list[int] = []

        def factory(_target, api_preference):
            calls.append(api_preference)
            return FakeCapture(opened=True, frames=[image])

        source = RTSPSource("rtsp://camera", backend="ffmpeg", capture_factory=factory, threaded=False)

        self.assertEqual(source.backend_name, "FFmpeg")
        self.assertEqual(calls, [cv2.CAP_FFMPEG])

    def test_raises_when_no_backend_delivers_initial_frame(self) -> None:
        def factory(_target, _api_preference):
            return FakeCapture(opened=True, frames=[])

        with self.assertRaisesRegex(RuntimeError, "initial RTSP frame"):
            RTSPSource("rtsp://camera", backend="auto", capture_factory=factory, threaded=False)


if __name__ == "__main__":
    unittest.main()
