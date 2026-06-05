from __future__ import annotations

from .models import BBox, Detection, Track


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


class OnePersonTracker:
    def __init__(self, max_missed_frames: int = 12) -> None:
        self.max_missed_frames = max_missed_frames
        self._track: Track | None = None

    def update(self, detections: list[Detection]) -> Track | None:
        person_detections = [det for det in detections if det.label == "person"]
        if not person_detections:
            if self._track is None:
                return None
            missed = self._track.missed_frames + 1
            if missed > self.max_missed_frames:
                self._track = None
                return None
            self._track = Track(
                track_id=self._track.track_id,
                bbox=self._track.bbox,
                confidence=self._track.confidence,
                keypoints=self._track.keypoints,
                center=self._track.center,
                missed_frames=missed,
            )
            return self._track

        if self._track is None:
            chosen = max(person_detections, key=lambda det: det.confidence)
        else:
            chosen = max(
                person_detections,
                key=lambda det: (0.75 * iou(self._track.bbox, det.bbox)) + (0.25 * det.confidence),
            )

        self._track = Track(
            track_id=1,
            bbox=chosen.bbox,
            confidence=chosen.confidence,
            keypoints=chosen.keypoints,
            center=bbox_center(chosen.bbox),
            missed_frames=0,
        )
        return self._track
