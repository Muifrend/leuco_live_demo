from __future__ import annotations

from dataclasses import dataclass
import math

from .models import BBox, Keypoints, Point


@dataclass(frozen=True)
class MockPerson:
    center: Point
    arm_phase: float
    bbox: BBox
    keypoints: Keypoints


def mock_person(index: int, scenario: str, width: int, height: int) -> MockPerson | None:
    if scenario == "empty":
        return None
    center, phase = _mock_motion(index, scenario, width, height)
    bbox, keypoints = _mock_geometry(center, phase)
    return MockPerson(center=center, arm_phase=phase, bbox=bbox, keypoints=keypoints)


def _mock_motion(index: int, scenario: str, width: int, height: int) -> tuple[Point, float]:
    if scenario in {"high_risk", "alert"}:
        return (width * 0.52, height * 0.52), index * 0.9
    if scenario == "outside":
        return (width * 0.5, height * 0.9), index * 0.15
    travel = (index * 4) % 560
    return (200 + travel, height * 0.55), index * 0.18


def _mock_geometry(center: Point, phase: float) -> tuple[BBox, Keypoints]:
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
