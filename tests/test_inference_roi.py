from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from leuco_live_demo.config import load_config
from leuco_live_demo.inference import (
    InferenceBackend,
    MockInferenceBackend,
    ROICropBackend,
    create_backend,
)
from leuco_live_demo.models import Detection, Frame, InferenceResult


class RecordingBackend(InferenceBackend):
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.shapes: list[tuple[int, int]] = []

    def infer(self, frame: Frame) -> InferenceResult:
        height, width = frame.image.shape[:2]
        self.shapes.append((height, width))
        return InferenceResult(detections=list(self.detections))


class ROICropBackendTests(unittest.TestCase):
    def test_crops_frame_and_maps_detections_to_full_frame(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        wrapped = RecordingBackend(
            [
                Detection(
                    bbox=(5, 7, 40, 50),
                    confidence=0.82,
                    keypoints={"nose": (10.5, 12.25), "left_wrist": (20.0, 30.0)},
                )
            ]
        )
        backend = ROICropBackend(wrapped, (30, 20, 130, 80))

        result = backend.infer(Frame(image=image, timestamp=12.5, index=7))

        self.assertEqual(wrapped.shapes, [(60, 100)])
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.bbox, (35, 27, 70, 70))
        self.assertEqual(detection.confidence, 0.82)
        self.assertEqual(detection.keypoints["nose"], (40.5, 32.25))
        self.assertEqual(detection.keypoints["left_wrist"], (50.0, 50.0))

    def test_scales_roi_from_reference_size_before_mapping_detections(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        wrapped = RecordingBackend(
            [
                Detection(
                    bbox=(1, 2, 5, 6),
                    confidence=0.77,
                    keypoints={"nose": (3.0, 4.0)},
                )
            ]
        )
        backend = ROICropBackend(wrapped, (10, 5, 60, 25), reference_size=(100, 50))

        result = backend.infer(Frame(image=image, timestamp=12.5, index=7))

        self.assertEqual(wrapped.shapes, [(40, 100)])
        self.assertEqual(result.detections[0].bbox, (21, 12, 25, 16))
        self.assertEqual(result.detections[0].keypoints["nose"], (23.0, 14.0))

    def test_empty_roi_result_still_calls_backend_on_crop(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        wrapped = RecordingBackend([])
        backend = ROICropBackend(wrapped, (30, 20, 130, 80))

        result = backend.infer(Frame(image=image, timestamp=12.5, index=7))

        self.assertEqual(wrapped.shapes, [(60, 100)])
        self.assertEqual(result.detections, [])

    def test_roi_outside_frame_returns_no_detections(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        wrapped = RecordingBackend([Detection(bbox=(1, 1, 10, 10), confidence=0.5)])
        backend = ROICropBackend(wrapped, (300, 20, 400, 80))

        result = backend.infer(Frame(image=image, timestamp=12.5, index=7))

        self.assertEqual(wrapped.shapes, [])
        self.assertEqual(result.detections, [])

    def test_disabled_roi_preserves_plain_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                argv=[],
                environ={"LEUCO_INFERENCE_BACKEND": "mock", "LEUCO_INFERENCE_ROI": "disabled"},
                demo_env=Path(tmp) / ".env",
                amcrest_env=Path(tmp) / ".amcrest",
            )

        self.assertIsInstance(create_backend(config), MockInferenceBackend)


if __name__ == "__main__":
    unittest.main()
