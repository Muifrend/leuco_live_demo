from __future__ import annotations

import unittest

from leuco_live_demo.models import Track
from leuco_live_demo.pool_gate import PoolGate, lower_center_point, point_in_polygon


def track_with_bbox(bbox) -> Track:
    return Track(
        track_id=1,
        bbox=bbox,
        confidence=0.9,
        keypoints={},
        center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
    )


class PoolGateTests(unittest.TestCase):
    def test_lower_center_uses_bottom_of_bbox(self) -> None:
        track = track_with_bbox((10, 20, 30, 80))

        self.assertEqual(lower_center_point(track), (20.0, 80.0))

    def test_point_in_polygon_counts_inside_outside_and_boundary(self) -> None:
        polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

        self.assertTrue(point_in_polygon((5.0, 5.0), polygon))
        self.assertFalse(point_in_polygon((11.0, 5.0), polygon))
        self.assertTrue(point_in_polygon((5.0, 10.0), polygon))

    def test_polygon_gate_marks_track_inside(self) -> None:
        gate = PoolGate("polygon", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

        pool = gate.evaluate(track_with_bbox((4, 1, 6, 8)))

        self.assertTrue(pool.active)
        self.assertTrue(pool.in_pool)
        self.assertEqual(pool.test_point, (5.0, 8.0))

    def test_polygon_gate_marks_track_outside(self) -> None:
        gate = PoolGate("polygon", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

        pool = gate.evaluate(track_with_bbox((20, 1, 24, 8)))

        self.assertTrue(pool.active)
        self.assertFalse(pool.in_pool)
        self.assertEqual(pool.test_point, (22.0, 8.0))

    def test_polygon_gate_no_track_is_enabled_but_not_in_pool(self) -> None:
        gate = PoolGate("polygon", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))

        pool = gate.evaluate(None)

        self.assertTrue(pool.active)
        self.assertFalse(pool.in_pool)
        self.assertIsNone(pool.test_point)

    def test_polygon_gate_scales_reference_polygon_to_frame(self) -> None:
        gate = PoolGate(
            "polygon",
            ((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
            polygon_reference_size=(100, 100),
        )

        pool = gate.evaluate(track_with_bbox((29, 1, 31, 15)), frame_size=(200, 100))

        self.assertEqual(pool.polygon, ((20.0, 10.0), (40.0, 10.0), (40.0, 20.0), (20.0, 20.0)))
        self.assertTrue(pool.in_pool)


if __name__ == "__main__":
    unittest.main()
