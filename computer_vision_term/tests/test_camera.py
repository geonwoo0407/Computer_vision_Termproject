import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera import FramePacket, InputSourceError, MultiRealSenseSource, list_realsense_devices


class CameraDataTests(unittest.TestCase):
    def test_discovery_recovers_from_a_usb_control_timeout(self) -> None:
        sdk = Mock()
        query = sdk.context.return_value.query_devices
        query.side_effect = [RuntimeError("Connection timed out"), []]
        with patch.dict("sys.modules", pyrealsense2=sdk), patch("camera.time.sleep"):
            self.assertEqual(list_realsense_devices(), [])
        self.assertEqual(query.call_count, 2)

    def test_persistent_usb_timeout_reports_sdk_error_after_retry(self) -> None:
        sdk = Mock()
        query = sdk.context.return_value.query_devices
        query.side_effect = RuntimeError("UVCIOC_CTRL_QUERY: Connection timed out")
        with patch.dict("sys.modules", pyrealsense2=sdk), patch("camera.time.sleep"):
            with self.assertRaisesRegex(InputSourceError, "after retry: UVCIOC_CTRL_QUERY"):
                list_realsense_devices()
        self.assertEqual(query.call_count, 2)

    def test_frame_packet_optional_rgbd_metadata(self) -> None:
        packet = FramePacket(
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp=1.0,
            camera_id="camera_1",
            frame_index=7,
            serial_number="1234",
            color_timestamp_ms=1000.0,
            depth_timestamp_ms=999.5,
        )
        self.assertEqual(packet.serial_number, "1234")
        self.assertEqual(packet.frame_index, 7)
        self.assertEqual(packet.color_timestamp_ms, 1000.0)
        self.assertEqual(packet.depth_timestamp_ms, 999.5)

    def test_multi_source_requires_two_unique_serials(self) -> None:
        with self.assertRaises(ValueError):
            MultiRealSenseSource(["one"])
        with self.assertRaises(ValueError):
            MultiRealSenseSource(["same", "same"])

    def test_multi_source_requires_matching_camera_ids(self) -> None:
        with self.assertRaises(ValueError):
            MultiRealSenseSource(["one", "two"], ["camera_1"])


if __name__ == "__main__":
    unittest.main()
