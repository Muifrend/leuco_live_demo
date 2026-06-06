from __future__ import annotations

from .models import BBox, FrameSize


def scale_roi_to_frame(roi: BBox, reference_size: FrameSize | None, frame_width: int, frame_height: int) -> BBox:
    if reference_size is None:
        return roi
    reference_width, reference_height = reference_size
    scale_x = frame_width / reference_width
    scale_y = frame_height / reference_height
    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )
