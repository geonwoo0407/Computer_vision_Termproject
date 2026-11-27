import io
import json
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from seat_fusion import Observation, SeatFusion, SeatView, combine, observe_view
from camera import FramePacket
from detector import VisionDetector
from multiview import record_observation


def seen(table="no_change", changes=(), person=False, known=False):
    return Observation(person, known, table, list(changes))


class TwoViewTests(unittest.TestCase):
    def test_live_log_preserves_frame_identity_even_without_detections(self):
        sdk = Mock()
        sdk.YOLO.return_value.names = {0: "person"}
        sdk.YOLO.return_value.predict.return_value = [Mock(boxes=None)]
        with patch.dict("sys.modules", ultralytics=sdk):
            detector = VisionDetector()
        packet = FramePacket(np.zeros((12, 16, 3), np.uint8), 12., "rear", 81,
                             serial_number="rear-device", color_timestamp_ms=12000.,
                             timestamp_domain="hardware_clock", received_monotonic=207.4)
        detected = detector.process_packet(packet)
        log = io.StringIO()
        record_observation(log, packet.serial_number, packet.received_monotonic, {}, {},
                           detector_frame=detected)
        row = json.loads(log.getvalue())
        frame = row["detector_frame"]
        self.assertEqual(frame["camera_id"], "rear")
        self.assertEqual(frame["serial_number"], row["serial"])
        self.assertEqual(frame["frame_id"], 81)
        self.assertEqual(frame["color_timestamp_ms"], 12000.)
        self.assertEqual(frame["timestamp_domain"], "hardware_clock")
        self.assertEqual(frame["capture_monotonic_s"], row["capture_monotonic_s"])
        self.assertEqual(frame["detections"], [])
        self.assertFalse(frame["depth_available"])

    def test_opposite_view_can_confirm_person_or_recognized_item(self):
        self.assertEqual(combine(seen(), seen(person=True)), (True, False))
        self.assertEqual(combine(seen(known=True), seen("occluded")), (False, True))

    def test_uncertain_and_spatially_different_changes_are_not_belongings(self):
        bag = seen("object_candidate", [(20, 20, 12, 12)])
        self.assertIsNone(combine(bag, seen()))
        self.assertIsNone(combine(bag, seen("object_candidate", [(31, 31, 12, 12)])))
        self.assertEqual(combine(bag, seen("object_candidate", [(22, 22, 11, 12)])), (False, True))
        self.assertIsNone(combine(seen(known=None), seen()))
        self.assertIsNone(combine(seen(person=None), seen()))

    def test_missing_stale_duplicate_and_out_of_order_frames(self):
        fusion = SeatFusion(["front", "rear"])
        observations = {138: seen()}
        def update(camera, when, now=None):
            return fusion.update(camera, when, observations, when if now is None else now)
        self.assertFalse(update("front", 1))
        self.assertFalse(update("rear", 2))
        self.assertFalse(update("front", 1.4, 2.5))
        self.assertEqual(update("front", 2.1)[138][0], "pending")
        self.assertFalse(update("rear", 2))
        self.assertFalse(update("rear", 1.5, 2.2))
        for when in range(3, 8):
            update("front", when)
            self.assertEqual(update("rear", when)[138][0], "pending")
        update("front", 8)
        self.assertEqual(update("rear", 8)[138], ("stable", (False, False)))
        update("front", 11)
        self.assertEqual(update("rear", 11)[138][0], "pending")

    def test_person_or_chair_can_occlude_table_but_chair_is_never_an_item(self):
        background = np.full((100, 140, 3), 130, np.uint8)
        view = SeatView(background, [(0, 0), (120, 0), (120, 80), (0, 80)], (4, 4, 110, 80))
        frame = background.copy()
        cv2.rectangle(frame, (30, 30), (58, 55), (10, 20, 210), -1)
        for _ in range(4):
            result = observe_view(frame, {1: view}, [{"bbox": [28, 28, 60, 57], "class_name": "chair"}])[1]
        self.assertEqual(result.table, "occluded")
        self.assertFalse(result.known_item)
        self.assertFalse(result.changes)

    def test_shared_detection_is_ambiguous_for_neighboring_seats(self):
        bg = np.full((90, 100, 3), 120, np.uint8)
        table = [(0, 40), (80, 40), (80, 80), (0, 80)]
        views = {1: SeatView(bg, table, (0, 0, 65, 45)), 2: SeatView(bg, table, (45, 0, 55, 45))}
        observed = observe_view(bg, views, [{"class_name": "person", "bbox": [45, 1, 55, 35]}])
        self.assertIsNone(observed[1].person)
        self.assertIsNone(observed[2].person)
