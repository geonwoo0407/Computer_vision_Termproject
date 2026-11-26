import unittest

import numpy as np

from camera import CameraIntrinsics
from depth import estimate_object_depth


class DepthEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            width=640,
            height=480,
            fx=600.0,
            fy=600.0,
            ppx=320.0,
            ppy=240.0,
            distortion_model="none",
            coefficients=(0.0,) * 5,
        )

    def test_median_and_manual_deprojection(self) -> None:
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        value, position, valid_count = estimate_object_depth(
            depth,
            [220, 140, 420, 340],
            self.intrinsics,
        )

        self.assertEqual(value, 2.0)
        self.assertEqual(position, [0.0, 0.0, 2.0])
        self.assertGreater(valid_count, 20)

    def test_invalid_values_and_saturation_return_none(self) -> None:
        for invalid_value in (0.0, np.nan, np.inf, 65.535):
            with self.subTest(value=invalid_value):
                depth = np.full((480, 640), invalid_value, dtype=np.float32)
                value, position, valid_count = estimate_object_depth(
                    depth,
                    [220, 140, 420, 340],
                    self.intrinsics,
                )
                self.assertIsNone(value)
                self.assertIsNone(position)
                self.assertEqual(valid_count, 0)

    def test_bbox_is_clipped_to_frame(self) -> None:
        depth = np.full((20, 30), 1.5, dtype=np.float32)
        value, _, valid_count = estimate_object_depth(
            depth,
            [-20, -20, 100, 100],
            self.intrinsics,
            min_valid_pixels=1,
        )
        self.assertEqual(value, 1.5)
        self.assertGreater(valid_count, 0)

    def test_realsense_native_deprojection(self) -> None:
        try:
            import pyrealsense2 as rs
        except (ImportError, OSError):
            self.skipTest("pyrealsense2 is not installed")

        native = rs.intrinsics()
        native.width = 640
        native.height = 480
        native.fx = 600.0
        native.fy = 600.0
        native.ppx = 320.0
        native.ppy = 240.0
        native.model = rs.distortion.none
        native.coeffs = [0.0] * 5
        depth = np.full((480, 640), 2.0, dtype=np.float32)

        value, position, _ = estimate_object_depth(
            depth,
            [220, 140, 420, 340],
            self.intrinsics,
            native_intrinsics=native,
        )
        self.assertEqual(value, 2.0)
        self.assertEqual(position, [0.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
