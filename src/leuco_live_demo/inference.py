from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .config import AppConfig
from .mock_scene import mock_person
from .models import BBox, Detection, Frame, FrameSize, InferenceResult, Keypoints
from .roi import scale_roi_to_frame

LOGGER = logging.getLogger(__name__)

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


class ROICropBackend(InferenceBackend):
    """Runs another backend on a configured crop and maps detections back."""

    def __init__(
        self,
        backend: InferenceBackend,
        roi: BBox,
        reference_size: FrameSize | None = None,
    ) -> None:
        self.backend = backend
        self.roi = roi
        self.reference_size = reference_size
        self._logged_roi: BBox | None = None

    def infer(self, frame: Frame) -> InferenceResult:
        height, width = frame.image.shape[:2]
        effective_roi = scale_roi_to_frame(self.roi, self.reference_size, width, height)
        if self._logged_roi != effective_roi:
            LOGGER.info(
                "effective inference ROI %s for frame=%sx%s reference=%s",
                effective_roi,
                width,
                height,
                self.reference_size or "frame",
            )
            self._logged_roi = effective_roi
        bounded_roi = _clip_roi(effective_roi, width, height)
        if bounded_roi is None:
            return InferenceResult(detections=[])

        x1, y1, x2, y2 = bounded_roi
        crop = frame.image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return InferenceResult(detections=[])

        crop_frame = Frame(image=crop, timestamp=frame.timestamp, index=frame.index)
        result = self.backend.infer(crop_frame)
        return InferenceResult(
            detections=[
                Detection(
                    bbox=_clip_bbox(_offset_bbox(detection.bbox, x1, y1), width, height),
                    confidence=detection.confidence,
                    label=detection.label,
                    keypoints=_offset_keypoints(detection.keypoints, x1, y1),
                )
                for detection in result.detections
            ]
        )


class MockInferenceBackend(InferenceBackend):
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario

    def infer(self, frame: Frame) -> InferenceResult:
        if self.scenario == "empty":
            return InferenceResult(detections=[])
        height, width = frame.image.shape[:2]
        person = mock_person(frame.index, self.scenario, width, height)
        if person is None:
            return InferenceResult(detections=[])
        return InferenceResult(
            detections=[
                Detection(
                    bbox=_clip_bbox(person.bbox, width, height),
                    confidence=0.94,
                    keypoints=person.keypoints,
                )
            ]
        )


class YOLOPoseBackend(InferenceBackend):
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. Place weights there or set LEUCO_MODEL_PATH; "
                "the demo does not download model files automatically."
            )
        from ultralytics import YOLO

        LOGGER.info("loading YOLO pose model: %s", model_path)
        self.model = YOLO(str(model_path))
        LOGGER.info("YOLO pose model loaded")

    def infer(self, frame: Frame) -> InferenceResult:
        height, width = frame.image.shape[:2]
        results = self.model(frame.image, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if cls_id != 0:
                    continue
                bbox = _clip_bbox(tuple(int(round(float(value))) for value in box.xyxy[0]), width, height)
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=confidence,
                        keypoints=_extract_pose_keypoints(result, idx),
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


def _extract_pose_keypoints(result, detection_index: int) -> Keypoints:
    keypoints_obj = getattr(result, "keypoints", None)
    xy = getattr(keypoints_obj, "xy", None)
    if xy is None or detection_index >= len(xy):
        return {}
    keypoints: Keypoints = {}
    for name, point in zip(COCO_POSE_NAMES, xy[detection_index].tolist()):
        if len(point) < 2:
            continue
        x, y = point[:2]
        if x > 0 and y > 0:
            keypoints[name] = (float(x), float(y))
    return keypoints


def _clip_bbox(bbox: tuple[int, ...], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    clipped_x1 = max(0, min(width - 1, x1))
    clipped_y1 = max(0, min(height - 1, y1))
    clipped_x2 = max(clipped_x1 + 1, min(width, x2))
    clipped_y2 = max(clipped_y1 + 1, min(height, y2))
    return (
        clipped_x1,
        clipped_y1,
        clipped_x2,
        clipped_y2,
    )


def _clip_roi(roi: BBox, width: int, height: int) -> BBox | None:
    x1, y1, x2, y2 = roi
    if width <= 0 or height <= 0 or x1 >= width or y1 >= height:
        return None
    clipped_x1 = max(0, min(width - 1, x1))
    clipped_y1 = max(0, min(height - 1, y1))
    clipped_x2 = min(width, x2)
    clipped_y2 = min(height, y2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return None
    return (clipped_x1, clipped_y1, clipped_x2, clipped_y2)


def _offset_bbox(bbox: BBox, offset_x: int, offset_y: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)


def _offset_keypoints(keypoints: Keypoints, offset_x: int, offset_y: int) -> Keypoints:
    return {name: (x + offset_x, y + offset_y) for name, (x, y) in keypoints.items()}


def create_backend(config: AppConfig) -> InferenceBackend:
    backend: InferenceBackend
    if config.inference_backend == "mock":
        LOGGER.info("creating mock inference backend scenario=%s", config.mock_scenario)
        backend = MockInferenceBackend(config.mock_scenario)
    elif config.inference_backend == "yolo_pose":
        backend = YOLOPoseBackend(config.model_path)
    elif config.inference_backend == "motion_box":
        LOGGER.info("creating motion_box inference backend")
        backend = MotionBoxBackend()
    else:
        raise ValueError(f"Unsupported inference backend: {config.inference_backend}")

    if config.inference_roi is None:
        return backend
    LOGGER.info(
        "wrapping inference backend with ROI crop %s reference_size=%s",
        config.inference_roi,
        config.inference_roi_reference_size or "frame",
    )
    return ROICropBackend(backend, config.inference_roi, config.inference_roi_reference_size)
