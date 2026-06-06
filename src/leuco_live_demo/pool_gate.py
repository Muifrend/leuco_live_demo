from __future__ import annotations

from .models import FrameSize, Point, Polygon, PoolState, Track


def lower_center_point(track: Track) -> Point:
    x1, _y1, x2, y2 = track.bbox
    return ((x1 + x2) / 2.0, float(y2))


def scale_polygon_to_frame(
    polygon: Polygon,
    reference_size: FrameSize | None,
    frame_size: FrameSize | None,
) -> Polygon:
    if reference_size is None or frame_size is None:
        return polygon
    reference_width, reference_height = reference_size
    frame_width, frame_height = frame_size
    scale_x = frame_width / reference_width
    scale_y = frame_height / reference_height
    return tuple((x * scale_x, y * scale_y) for x, y in polygon)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersection = ((x2 - x1) * (y - y1) / (y2 - y1)) + x1
            if x_intersection >= x:
                inside = not inside
    return inside


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = ((y - y1) * (x2 - x1)) - ((x - x1) * (y2 - y1))
    if abs(cross) > 1e-6:
        return False
    within_x = min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6
    within_y = min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6
    return within_x and within_y


class PoolGate:
    def __init__(
        self,
        mode: str,
        polygon: Polygon | None = None,
        polygon_reference_size: FrameSize | None = None,
    ) -> None:
        if mode not in {"disabled", "full_frame_stub", "polygon"}:
            raise ValueError(f"Unsupported pool gate mode: {mode}")
        if mode == "polygon" and polygon is None:
            raise ValueError("Polygon pool gate requires a polygon")
        self.mode = mode
        self.polygon = polygon
        self.polygon_reference_size = polygon_reference_size

    def evaluate(self, track: Track | None, frame_size: FrameSize | None = None) -> PoolState:
        if self.mode == "disabled":
            return PoolState(
                mode=self.mode,
                in_pool=False,
                active=False,
                label="Pool gate disabled",
            )
        if self.mode == "full_frame_stub":
            if track is None:
                return PoolState(
                    mode=self.mode,
                    in_pool=False,
                    active=True,
                    label="Full-frame pool stub",
                )
            return PoolState(
                mode=self.mode,
                in_pool=True,
                active=True,
                label="Full-frame pool stub",
            )

        polygon = scale_polygon_to_frame(self.polygon or (), self.polygon_reference_size, frame_size)
        if track is None:
            return PoolState(
                mode=self.mode,
                in_pool=False,
                active=True,
                label="Pool polygon",
                polygon=polygon,
            )
        test_point = lower_center_point(track)
        in_pool = point_in_polygon(test_point, polygon)
        return PoolState(
            mode=self.mode,
            in_pool=in_pool,
            active=True,
            label="Inside pool polygon" if in_pool else "Outside pool polygon",
            polygon=polygon,
            test_point=test_point,
        )
