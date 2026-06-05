from __future__ import annotations

from pathlib import Path
import math

import cv2

from .config import AppConfig
from .models import Detection, Frame, InferenceResult, Keypoints
from .sources import mock_person_pose

COCO_POSE_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class InferenceBackend:
    def infer(self, frame: Frame) -> InferenceResult:
        raise NotImplementedError


class MockInferenceBackend(InferenceBackend):
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario

    def infer(self, frame: Frame) -> InferenceResult:
        if self.scenario == "empty":
            return InferenceResult(detections=[])
        height, width = frame.image.shape[:2]
        center, phase = mock_person_pose(frame.index, self.scenario, width, height)
        bbox, keypoints = _mock_detection_geometry(center, phase)
        return InferenceResult(detections=[Detection(bbox=bbox, confidence=0.94, keypoints=keypoints)])


class YOLOPoseBackend(InferenceBackend):
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. Place weights there or set LEUCO_MODEL_PATH; "
                "the demo does not download model files automatically."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))

    def infer(self, frame: Frame) -> InferenceResult:
        results = self.model(frame.image, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            keypoints_obj = getattr(result, "keypoints", None)
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = [int(round(float(value))) for value in box.xyxy[0]]
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                keypoints: Keypoints = {}
                if keypoints_obj is not None and keypoints_obj.xy is not None:
                    points = keypoints_obj.xy[idx].tolist()
                    for name, point in zip(COCO_POSE_NAMES, points):
                        x, y = point[:2]
                        if x > 0 and y > 0:
                            keypoints[name] = (float(x), float(y))
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        keypoints=keypoints,
                    )
                )
        return InferenceResult(detections=detections)


class MotionBoxBackend(InferenceBackend):
    """Simple frame-difference person-box fallback for static camera demos."""

    def __init__(self) -> None:
        self._prev_gray = None

    def infer(self, frame: Frame) -> InferenceResult:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return InferenceResult(detections=[])
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return InferenceResult(detections=[])
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 600:
            return InferenceResult(detections=[])
        x, y, w, h = cv2.boundingRect(contour)
        pad = 16
        frame_h, frame_w = gray.shape[:2]
        bbox = (
            max(0, x - pad),
            max(0, y - pad),
            min(frame_w, x + w + pad),
            min(frame_h, y + h + pad),
        )
        return InferenceResult(detections=[Detection(bbox=bbox, confidence=0.50)])


def _mock_detection_geometry(center: tuple[float, float], phase: float) -> tuple[tuple[int, int, int, int], Keypoints]:
    cx, cy = center
    shoulder_y = cy - 44
    head_y = cy - 75
    arm_swing = math.sin(phase) * 52
    bbox = (int(cx - 82), int(cy - 96), int(cx + 82), int(cy + 82))
    keypoints: Keypoints = {
        "nose": (cx, head_y),
        "left_shoulder": (cx - 28, shoulder_y),
        "right_shoulder": (cx + 28, shoulder_y),
        "left_elbow": (cx - 48, shoulder_y + arm_swing * 0.7),
        "right_elbow": (cx + 48, shoulder_y - arm_swing * 0.7),
        "left_wrist": (cx - 66, shoulder_y + arm_swing),
        "right_wrist": (cx + 66, shoulder_y - arm_swing),
    }
    return bbox, keypoints


def create_backend(config: AppConfig) -> InferenceBackend:
    if config.inference_backend == "mock":
        return MockInferenceBackend(config.mock_scenario)
    if config.inference_backend == "yolo_pose":
        return YOLOPoseBackend(config.model_path)
    if config.inference_backend == "motion_box":
        return MotionBoxBackend()
    raise ValueError(f"Unsupported inference backend: {config.inference_backend}")
