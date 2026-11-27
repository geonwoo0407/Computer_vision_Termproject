import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from seat_grid import SeatGrid, draw_grid, prepare_grid, read_layout, save_layout
from seat_app import GridPublisher


class GridTests(unittest.TestCase):
    def test_app_receives_only_confirmed_observations_at_mapped_seats(self):
        sent = []
        app = SimpleNamespace(send=lambda *args: sent.append(args))
        seat_ids = [127, 26, 25, 22, 23, 24]
        with self.assertRaises(ValueError):
            GridPublisher(app, [1, 2, 3, 4, 5, 5])
        publisher = GridPublisher(app, seat_ids)
        states = ["empty", "empty", "away", "occupied", "occupied", "away"]
        seats = [dict(slot=i, state=state, confirmed_this_frame=i in (2, 3, 4))
                 for i, state in enumerate(states, 1)]
        try:
            publisher.update(seats, 30.)
        finally:
            publisher.close()
        self.assertEqual([s["app_seat_id"] for s in seats], seat_ids)
        self.assertEqual(sent, [(26, (False, False), 30.), (25, (False, True), 30.),
                                (22, (True, False), 30.)])

    def setUp(self):
        self.frame = np.full((180, 320, 3), 160, np.uint8)
        self.regions = []
        for y in (5, 95):
            for x in (10, 110, 210):
                self.regions.append([(x, y, x+100, y+30), (x, y+35, x+100, y+75)])
        self.view = SimpleNamespace(frame=self.frame, serial_number="cam1", camera_id="rgb", timestamp=0)

    def test_six_slots_update_independently_and_show_evidence(self):
        grid = SeatGrid(self.frame, self.regions, [], 5)
        changed = self.frame.copy()
        changed[102:117, 230:250] = (25, 25, 25)
        detections = [{"class_name": "person", "bbox": (15, 26, 85, 75), "confidence": .88},
                      {"class_name": "book", "bbox": (135, 10, 165, 25), "confidence": .81}]
        for i in range(34):
            seats = grid.observe(changed, detections, i/5)
        self.assertEqual([s["state"] for s in seats], ["occupied", "away", "empty", "empty", "empty", "away"])
        self.assertEqual([s["source"] for s in seats], ["person", "book", "clear", "clear", "clear", "RGB"])
        self.assertEqual([(s["slot"], s["row"], s["column"]) for s in seats],
                         [(1,1,1), (2,1,2), (3,1,3), (4,2,1), (5,2,2), (6,2,3)])
        self.assertGreater(draw_grid(changed.copy(), seats).shape[0], changed.shape[0])

    def test_overlapping_seats_share_an_occluder_but_never_its_positive_vote(self):
        self.regions[0][1] = (10, 40, 135, 80)
        self.regions[1][1] = (100, 40, 210, 80)
        grid = SeatGrid(self.frame, self.regions, [], 5)
        for name, expected in (("person", "occupied"), ("backpack", "away")):
            changed = self.frame.copy()
            changed[44:76, 73:141] = 20
            detections = [{"class_name": name, "bbox": (73, 44, 141, 76), "confidence": .9}]
            for i in range(35):
                seats = grid.observe(changed, detections, i/5)
            self.assertEqual([s["state"] for s in seats], [expected] + ["empty"]*5)
            self.assertFalse(seats[1]["rgb_boxes"])

    def test_interactive_registration_then_headless_reuse(self):
        def packets():
            for i in range(25):
                yield SimpleNamespace(frame=self.frame, serial_number="cam1", camera_id="rgb",
                                      timestamp=i/5, color_timestamp_ms=i*200)
        detector = Mock()
        detector.process_frame.return_value = {"detections": []}
        picks = [(x,y,r-x,d-y) for pair in self.regions for x,y,r,d in pair]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seats.json"
            with patch("cv2.imshow"), patch("cv2.destroyWindow"), patch("cv2.waitKey", return_value=ord("f")), \
                    patch("cv2.selectROI", side_effect=picks) as select:
                grid = prepare_grid(packets(), detector, path, False, 5)
            self.assertEqual(select.call_count, 12)
            self.assertEqual(len(grid.seats), 6)
            self.assertEqual(read_layout(path, self.view), self.regions)
            self.assertEqual(len(prepare_grid(packets(), detector, path, True, 5).seats), 6)
            self.assertEqual(detector.process_frame.call_count, 2)

    def test_wrong_camera_and_malformed_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layout.json"
            save_layout(path, self.view, self.regions)
            self.view.serial_number = "cam2"
            with self.assertRaises(ValueError):
                read_layout(path, self.view)
            self.view.serial_number = None  # An offline replay of the original view.
            self.assertEqual(read_layout(path, self.view), self.regions)
            data = json.loads(path.read_text())
            data["seats"][2]["chair"][0] = -5
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_layout(path, self.view)
